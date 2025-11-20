# gecko/core/engine/react.py
from __future__ import annotations
import json
import logging
from typing import List, Dict, Any, Optional, Type, TypeVar, AsyncIterator

from pydantic import BaseModel

from gecko.core.engine.base import CognitiveEngine
from gecko.core.message import Message
from gecko.core.output import AgentOutput
from gecko.core.utils import ensure_awaitable
from gecko.core.prompt import PromptTemplate
from gecko.core.structure import StructureEngine

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# 默认 Prompt 模板
DEFAULT_REACT_TEMPLATE = """You are a helpful AI assistant.
Available Tools:
{% for tool in tools %}
- {{ tool.function.name }}: {{ tool.function.description }}
{% endfor %}

Answer the user's request. Use tools if necessary.
"""

class ReActEngine(CognitiveEngine):
    """
    Gecko v0.3 ReAct Engine
    支持：
    1. 多模态输入 (Multimodal)
    2. 结构化输出 (Structured Output with Retry)
    3. 流式响应 (Streaming)
    """
    def __init__(self, model, toolbox, memory, max_turns: int = 5, system_prompt: str | PromptTemplate = None):
        super().__init__(model, toolbox, memory)
        self.max_turns = max_turns
        
        # 初始化 Prompt
        if system_prompt is None:
            self.prompt_template = PromptTemplate(template=DEFAULT_REACT_TEMPLATE)
        elif isinstance(system_prompt, str):
            self.prompt_template = PromptTemplate(template=system_prompt)
        else:
            self.prompt_template = system_prompt

    async def _build_context(self, input_messages: List[Message]) -> List[Message]:
        """
        构建上下文：加载记忆 + 注入 System Prompt + 合并新消息
        """
        # 1. 加载并修剪历史
        if self.memory.storage and self.memory.session_id:
            raw_data = await self.memory.storage.get(self.memory.session_id)
            history_raw = raw_data.get("messages", []) if raw_data else []
            history = await self.memory.get_history(history_raw)
        else:
            history = []
            
        # 2. 注入 System Prompt
        has_system = any(m.role == "system" for m in history)
        if not has_system:
            # 获取工具定义 (OpenAI 格式)
            tools_schema = self.toolbox.to_openai_schema()
            sys_content = self.prompt_template.format(tools=tools_schema)
            history.insert(0, Message(role="system", content=sys_content))

        # 3. 合并
        return history + input_messages

    def _prepare_request_params(
        self, 
        response_model: Optional[Type[T]], 
        strategy: str
    ) -> Dict[str, Any]:
        """
        根据结构化输出策略准备 LLM 参数 (Tools / Response Format)
        """
        kwargs = {}
        # 默认注入通用工具
        tools_schema = self.toolbox.to_openai_schema()
        if tools_schema:
            kwargs["tools"] = tools_schema
            kwargs["tool_choice"] = "auto"

        # 如果需要结构化输出，覆盖上述设置
        if response_model:
            if strategy in ["auto", "function_calling"]:
                # 策略 1: 强制 Tool Call
                tool_schema = StructureEngine.to_openai_tool(response_model)
                # 注意：这里我们将提取工具加入到 tools 列表
                if "tools" not in kwargs: kwargs["tools"] = []
                kwargs["tools"].append(tool_schema)
                kwargs["tool_choice"] = {
                    "type": "function", 
                    "function": {"name": tool_schema["function"]["name"]}
                }
            elif strategy == "json_mode":
                # 策略 2: JSON Mode
                kwargs["response_format"] = {"type": "json_object"}
                # 注意：通常还需要在 Prompt 里提示 "Reply in JSON"
        
        return kwargs

    async def step(
        self, 
        input_messages: List[Message],
        response_model: Optional[Type[T]] = None,
        strategy: str = "auto",
        max_retries: int = 2
    ) -> AgentOutput | T:
        
        current_context = await self._build_context(input_messages)
        
        turn = 0
        final_response: AgentOutput | None = None
        
        # 准备基础参数
        base_kwargs = self._prepare_request_params(response_model, strategy)

        while turn < self.max_turns:
            turn += 1
            
            # [修改] 使用 to_api_payload() 而不是 model_dump()
            # 这样 LiteLLM 收到的是纯字典，而不是 Pydantic 对象
            messages_payload = [m.to_api_payload() for m in current_context]
            
            # 调用 LLM
            response = await ensure_awaitable(
                self.model.acompletion, 
                messages=messages_payload,
                **base_kwargs
            )
            
            # 包装响应
            choice = response.choices[0]
            
            # [优化] 增加防御性逻辑，处理 model_dump 可能缺失 role 的情况
            if hasattr(choice.message, "model_dump"):
                msg_data = choice.message.model_dump()
            else:
                # 兼容部分 Mock 对象或旧版本 litellm
                msg_data = {
                    "content": getattr(choice.message, "content", ""),
                    "tool_calls": getattr(choice.message, "tool_calls", None),
                    "role": getattr(choice.message, "role", "assistant") # 默认 assistant
                }
            
            # 确保 role 存在
            if "role" not in msg_data or msg_data["role"] is None:
                msg_data["role"] = "assistant"
                
            response_message = Message(**msg_data)
            
            current_context.append(response_message)

            # 1. 优先检查是否为结构化提取任务的 Tool Call
            if response_model and strategy in ["auto", "function_calling"]:
                tool_calls = response_message.tool_calls or []
                extraction_tool_name = StructureEngine.to_openai_tool(response_model)["function"]["name"]
                
                # 检查是否命中了提取工具
                extraction_call = next((tc for tc in tool_calls if tc["function"]["name"] == extraction_tool_name), None)
                
                if extraction_call:
                    # 直接解析并返回，不再继续 ReAct 循环
                    try:
                        return await StructureEngine.parse(
                            content=response_message.content,
                            model_class=response_model,
                            raw_tool_calls=[extraction_call]
                        )
                    except Exception as e:
                        # 结构化解析失败，进入重试逻辑（见循环外）
                        logger.warning(f"Initial structure extraction failed: {e}")
                        pass # Fallthrough to retry logic

            # 2. 处理常规工具调用 (ReAct Loop)
            if response_message.tool_calls:
                has_real_tool_execution = False
                for tc in response_message.tool_calls:
                    func_name = tc["function"]["name"]
                    
                    # 跳过结构化提取专用的工具（如果有的话）
                    if response_model and func_name == StructureEngine.to_openai_tool(response_model)["function"]["name"]:
                        continue

                    args_str = tc["function"]["arguments"]
                    call_id = tc["id"]
                    
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        content = await self.toolbox.execute(func_name, args)
                        has_real_tool_execution = True
                    except Exception as e:
                        content = f"Error executing {func_name}: {str(e)}"

                    tool_msg = Message(
                        role="tool",
                        content=str(content),
                        tool_call_id=call_id,
                        name=func_name
                    )
                    current_context.append(tool_msg)
                
                if has_real_tool_execution:
                    continue # 继续下一轮 LLM 思考
            
            # 3. 既没有工具调用，也不是提取工具，说明生成了自然语言或 JSON 文本
            final_response = AgentOutput(
                content=response_message.content or "",
                raw=response
            )
            break
        
        if not final_response:
            final_response = AgentOutput(content="Max turns reached.")

        # --- 结构化输出解析与重试 (Self-Correction) ---
        if response_model:
            retry_count = 0
            while retry_count <= max_retries:
                try:
                    # 尝试解析 (Tool Call -> JSON -> Regex)
                    tool_calls = getattr(final_response.raw.choices[0].message, "tool_calls", None)
                    return await StructureEngine.parse(
                        content=final_response.content,
                        model_class=response_model,
                        raw_tool_calls=tool_calls
                    )
                except ValueError as e:
                    retry_count += 1
                    if retry_count > max_retries:
                        logger.error(f"Structure parsing failed after {max_retries} retries: {e}")
                        raise e
                    
                    logger.info(f"Structure parsing failed, retrying ({retry_count}/{max_retries})...")
                    
                    # 将错误反馈给 LLM 进行自我修正
                    error_msg = Message(role="user", content=f"Error parsing your response: {str(e)}. Please format the output strictly as JSON matching the schema.")
                    current_context.append(error_msg)
                    
                    # 重新生成
                    messages_payload = [m.model_dump() for m in current_context]
                    response = await ensure_awaitable(
                        self.model.acompletion,
                        messages=messages_payload,
                        **base_kwargs
                    )
                    
                    # 更新 final_response
                    msg_data = response.choices[0].message.model_dump()
                    new_msg = Message(**msg_data)
                    current_context.append(new_msg)
                    final_response = AgentOutput(content=new_msg.content or "", raw=response)

        # 保存记忆
        await self._save_memory(current_context)
        return final_response

    async def step_stream(self, input_messages: List[Message]) -> AsyncIterator[str]:
        """
        流式执行：
        简化策略：如果涉及工具调用，先在后台执行（非流式），直到产生最终回复时才流式输出。
        """
        current_context = await self._build_context(input_messages)
        turn = 0
        
        # 流式暂时不支持结构化强制（因为结构化通常需要完整 JSON）
        base_kwargs = self._prepare_request_params(None, "auto")

        while turn < self.max_turns:
            turn += 1
            
            # 1. 先尝试非流式 Peek，检查是否需要调用工具
            # 这里为了用户体验，如果不需要工具，我们希望立刻看到字。
            # 但如果不运行模型，我们不知道是否需要工具。
            # 方案：使用 astream，但在遇到 Tool Call chunk 时中断流式，转为后台执行。
            
            messages_payload = [m.model_dump() for m in current_context]
            
            # 发起流式请求
            stream_generator = self.model.astream(
                messages=messages_payload,
                **base_kwargs
            )
            
            # 状态累积
            accumulated_content = []
            tool_call_accumulator = [] # 复杂：因为 tool call 是分片传的
            is_tool_call_mode = False
            
            async for chunk in stream_generator:
                delta = chunk.choices[0].delta
                
                # 检查是否开始 Tool Call
                if getattr(delta, "tool_calls", None):
                    is_tool_call_mode = True
                    # 简单处理：一旦发现是 Tool Call，流式对用户来说就没有意义了（除非显示“正在调用...”）
                    # 我们这里选择：收集完整的 Tool Call 信息，然后执行
                    # 注意：LiteLLM/OpenAI 的流式 Tool Call 拼接比较复杂，这里简化逻辑：
                    # 如果检测到 Tool Call，建议直接 break，用非流式重新请求一次 (Fallback) 
                    # 这样虽然浪费了一次 Token，但大大降低了代码复杂度。
                    break 
                
                if delta.content:
                    yield delta.content
                    accumulated_content.append(delta.content)

            if is_tool_call_mode:
                # 发现需要工具，改用非流式请求完整拿到 Tool Call
                # (为了代码健壮性，避免手动拼接流式 JSON)
                full_resp = await ensure_awaitable(
                    self.model.acompletion, 
                    messages=messages_payload, 
                    **base_kwargs
                )
                msg = Message(**full_resp.choices[0].message.model_dump())
                current_context.append(msg)
                
                # 执行工具
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        # ... 执行工具逻辑 (同 step) ...
                        func_name = tc["function"]["name"]
                        args_str = tc["function"]["arguments"]
                        try:
                            args = json.loads(args_str)
                            content = await self.toolbox.execute(func_name, args)
                        except Exception as e:
                            content = str(e)
                            
                        current_context.append(Message(
                            role="tool", content=str(content), 
                            tool_call_id=tc["id"], name=func_name
                        ))
                    # 工具执行完，进入下一轮循环 (continue while)
                    continue
            
            # 如果不是 Tool Call 模式，说明流式已经输出完了内容
            # 记录完整内容到上下文
            full_text = "".join(accumulated_content)
            current_context.append(Message(role="assistant", content=full_text))
            
            # 保存记忆
            await self._save_memory(current_context)
            break

    async def _save_memory(self, context: List[Message]):
        if self.memory.storage and self.memory.session_id:
            full_to_save = [m.model_dump() for m in context]
            await self.memory.storage.set(
                self.memory.session_id, 
                {"messages": full_to_save}
            )