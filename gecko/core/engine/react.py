# gecko/core/engine/react.py
"""
ReAct 推理引擎

实现了 ReAct (Reason + Act) 认知架构，负责协调 LLM 与工具箱的交互循环。

核心功能：
1. 推理循环：基于 Thought-Action-Observation 模式的自动执行
2. 流式输出：支持低延迟的 Token 级流式响应，同时保持工具调用的完整性
3. 结构化输出：支持将推理结果解析为强类型的 Pydantic 对象
4. 稳健性设计：内置死循环检测、观测值截断、错误反馈与自动重试

优化日志：
- [Fix] 重构 step_stream: 移除 Peek 机制，显著降低首字延迟 (TTFT)
- [Fix] 修复 Jinja2 模板语法，正确处理字典属性访问
- [Fix] 完善生命周期钩子 (on_turn_start/end) 在流式模式下的覆盖
- [Feat] 增加工具调用死循环检测 (Hash-based loop detection)
- [Feat] 增加工具观测值 (Observation) 智能截断
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel

from gecko.core.engine.base import CognitiveEngine
from gecko.core.exceptions import AgentError, ModelError
from gecko.core.logging import get_logger
from gecko.core.memory import TokenMemory
from gecko.core.message import Message
from gecko.core.output import AgentOutput
from gecko.core.prompt import PromptTemplate
from gecko.core.structure import StructureEngine, StructureParseError
from gecko.core.toolbox import ToolBox
from gecko.core.utils import ensure_awaitable

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

# 默认 ReAct 提示词模板
# 包含时间注入和工具列表渲染
DEFAULT_REACT_TEMPLATE = """You are a helpful AI assistant.
Current Time: {{ current_time }}

Available Tools:
{% for tool in tools %}
- {{ tool['function']['name'] }}: {{ tool['function']['description'] }}
{% endfor %}

Answer the user's request. Use tools if necessary.
If you use a tool, just output the tool call format.
"""


class ExecutionContext:
    """
    执行上下文
    
    封装每一轮 ReAct 循环的运行时状态，用于在 Engine 内部传递数据。
    """

    def __init__(self, messages: List[Message]):
        self.messages = messages.copy()  # 浅拷贝，避免污染原始列表
        self.turn = 0
        self.metadata: Dict[str, Any] = {}
        
        # 状态追踪：用于死循环检测
        self.last_tool_calls_hash: Optional[int] = None
        self.consecutive_tool_error_count: int = 0

    def add_message(self, message: Message):
        """追加消息到当前上下文"""
        self.messages.append(message)

    def get_last_message(self) -> Optional[Message]:
        """获取最后一条消息"""
        return self.messages[-1] if self.messages else None


class ReActEngine(CognitiveEngine):
    """
    ReAct (Reason + Act) 引擎实现
    """

    def __init__(
        self,
        model: Any,
        toolbox: ToolBox,
        memory: TokenMemory,
        max_turns: int = 5,
        max_observation_length: int = 2000,
        system_prompt: Union[str, PromptTemplate, None] = None,
        on_turn_start: Optional[Callable[[ExecutionContext], Any]] = None,
        on_turn_end: Optional[Callable[[ExecutionContext], Any]] = None,
        on_tool_execute: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        **kwargs,
    ):
        """
        初始化 ReAct 引擎
        
        参数:
            model: LLM 模型实例 (需实现 ModelProtocol)
            toolbox: 工具箱实例
            memory: 记忆管理器
            max_turns: 最大思考轮数 (防止无限循环)
            max_observation_length: 工具输出的最大字符数 (防止 Context 爆炸)
            system_prompt: 自定义系统提示词
            on_turn_start: 每一轮开始时的钩子
            on_turn_end: 每一轮结束时的钩子
            on_tool_execute: 工具执行前的钩子
        """
        super().__init__(model, toolbox, memory, **kwargs)
        self.max_turns = max_turns
        self.max_observation_length = max_observation_length
        
        # Hooks
        self.on_turn_start = on_turn_start
        self.on_turn_end = on_turn_end
        self.on_tool_execute = on_tool_execute

        # 初始化 Prompt 模板
        if system_prompt is None:
            self.prompt_template = PromptTemplate(template=DEFAULT_REACT_TEMPLATE)
        elif isinstance(system_prompt, str):
            self.prompt_template = PromptTemplate(template=system_prompt)
        else:
            self.prompt_template = system_prompt

        # 能力检测缓存
        self._supports_functions = self._check_function_calling_support()
        self._supports_stream = self._check_streaming_support()

    def _check_function_calling_support(self) -> bool:
        if hasattr(self.model, "_supports_function_calling"):
            return getattr(self.model, "_supports_function_calling")
        return True  # 默认假设支持，运行时报错再处理

    def _check_streaming_support(self) -> bool:
        return hasattr(self.model, "astream")

    # ===================== 核心接口实现 =====================

    async def step( # type: ignore
        self,
        input_messages: List[Message],
        response_model: Optional[Type[T]] = None,
        strategy: str = "auto",
        max_retries: int = 2,
        **kwargs,
    ) -> Union[AgentOutput, T]:
        """
        执行单次推理任务 (同步等待模式)
        
        流程:
        1. 验证输入 & Hook
        2. 构建上下文 (Context) & 参数
        3. 执行 ReAct 循环 (Reasoning Loop)
        4. 处理结构化输出 (如果需要)
        5. 保存记忆 & Hook
        """
        start_time = time.time()

        # 1. 验证与 Hook
        self.validate_input(input_messages)
        await self.before_step(input_messages, **kwargs)

        logger.info(
            "ReAct execution started",
            input_count=len(input_messages),
            has_structure=response_model is not None,
        )

        try:
            # 2. 预处理与上下文构建
            augmented_messages = self._augment_messages_for_structure(
                input_messages, response_model
            )
            context = await self._build_execution_context(augmented_messages)
            
            llm_params = self._build_llm_params(response_model, strategy)
            llm_params.update(kwargs)

            # 3. 运行推理循环
            final_output = await self._run_reasoning_loop(
                context, llm_params, response_model
            )

            # 4. 结构化输出处理
            result: Union[AgentOutput, T] = final_output
            if response_model:
                result = await self._handle_structured_output(
                    final_output, response_model, context, llm_params, max_retries
                )

            # 5. 保存与收尾
            await self._save_context(context)

            # 触发 after_step (需将 T 转换为 AgentOutput 以兼容 Hook 签名)
            hook_output = (
                result
                if isinstance(result, AgentOutput)
                else AgentOutput(content=str(result), raw=result)
            )
            await self.after_step(input_messages, hook_output, **kwargs)

            # 统计
            if self.stats:
                self.stats.add_step(time.time() - start_time)

            return result

        except Exception as e:
            if self.stats:
                self.stats.errors += 1
            logger.exception("ReAct execution failed")
            await self.on_error(e, input_messages, **kwargs)
            raise

    async def step_stream( # type: ignore
        self, input_messages: List[Message], **kwargs
    ) -> AsyncIterator[str]:
        """
        执行流式推理任务
        
        特点:
        - 立即返回首个 Token (Low Latency)
        - 遇到工具调用时暂停输出，执行工具后继续流式生成
        """
        if not self._supports_stream:
            raise AgentError("当前模型不支持流式输出")

        start_time = time.time()
        await self.before_step(input_messages, **kwargs)

        # 构建上下文
        context = await self._build_execution_context(input_messages)
        llm_params = self._build_llm_params(None, "auto")
        llm_params.update(kwargs)

        try:
            # 进入递归流式循环
            async for token in self._run_streaming_loop(context, llm_params):
                yield token

            # 保存记忆
            await self._save_context(context)

            if self.stats:
                self.stats.add_step(time.time() - start_time)

        except Exception as e:
            if self.stats:
                self.stats.errors += 1
            logger.exception("ReAct stream failed")
            await self.on_error(e, input_messages, **kwargs)
            raise

    # ===================== 推理循环逻辑 =====================

    # [新增辅助方法] 提取通用的回合处理逻辑
    async def _process_turn_results(
        self, 
        context: ExecutionContext, 
        assistant_msg: Message, 
        response_model: Optional[Type[T]] = None
    ) -> bool:
        """
        处理 LLM 生成后的通用逻辑：
        1. 死循环检测
        2. 更新上下文
        3. 判断是否终止 (结构化提取/无工具调用)
        4. 执行工具
        
        返回:
            should_continue: 是否继续下一轮循环 (True=继续, False=终止)
        """
        # 1. 死循环检测
        if self._detect_infinite_loop(assistant_msg, context):
            logger.warning("Detected infinite tool loop, breaking.")
            
            # [关键] 不要把错误的 tool call 加入 context 的历史
            # 如果把死循环的 tool call 加入历史，LLM 在重试时看到历史里自己刚调用了这个工具，
            # 可能会再次困惑。
            # 策略：
            # 1. 记录这个由 Assistant 发出的错误尝试
            context.add_message(assistant_msg) 
            # 2. 紧接着追加 User/System 的警告，强制打断它的思维惯性
            context.add_message(Message.user(
                f"System Alert: Execution stopped because you are"
                f" calling tool '{assistant_msg.tool_calls[0]['function']['name']}' " # type: ignore
                "repeatedly with identical arguments. " # type: ignore
                "Stop looping. Provide the final answer immediately using the correct format."
            ))
            
            return False

        context.add_message(assistant_msg)

        # 2. 终止条件检查
        # 条件 A: 是结构化提取 (Implicit Tool Call)
        if response_model and self._is_structure_extraction(
            assistant_msg, response_model
        ):
            await self._trigger_turn_end(context)
            return False

        # 条件 B: 无工具调用 (纯文本回复)
        if not assistant_msg.tool_calls:
            await self._trigger_turn_end(context)
            return False

        # 3. 执行工具
        if self.stats:
            self.stats.tool_calls += len(assistant_msg.tool_calls)

        await self._execute_tool_calls(assistant_msg.tool_calls, context)

        # Hook: Turn End
        await self._trigger_turn_end(context)
        
        return True

    # [修改方法] 重构同步循环，使用 _process_turn_results
    async def _run_reasoning_loop(
        self,
        context: ExecutionContext,
        llm_params: Dict[str, Any],
        response_model: Optional[Type[T]],
    ) -> AgentOutput:
        """
        ReAct 主循环 (同步模式)
        """
        while context.turn < self.max_turns:
            context.turn += 1

            # Hook: Turn Start
            if self.on_turn_start:
                await ensure_awaitable(self.on_turn_start, context)

            # 1. LLM 推理
            response = await self._call_llm(context, llm_params)
            assistant_msg = self._parse_llm_response(response)
            context.metadata["last_response"] = response

            # 2. 处理回合逻辑 (复用)
            should_continue = await self._process_turn_results(
                context, assistant_msg, response_model
            )
            
            if not should_continue:
                break

        # 循环结束，返回最后结果
        last_msg = context.get_last_message()
        if not last_msg:
            return AgentOutput(content="No response generated.")

        return AgentOutput(
            content=last_msg.content or "", # type: ignore
            raw=context.metadata.get("last_response"),
            tool_calls=last_msg.tool_calls or [],
        )

    # [修改方法] 重构流式循环，使用 _process_turn_results
    async def _run_streaming_loop(
        self, context: ExecutionContext, llm_params: Dict[str, Any]
    ) -> AsyncIterator[str]:
        """
        ReAct 流式循环 (递归模式)
        """
        # 循环控制：只要 turn 未达上限，且 should_continue 为 True，就一直循环
        while context.turn < self.max_turns:
            context.turn += 1
            
            # Hook: Turn Start
            if self.on_turn_start:
                await ensure_awaitable(self.on_turn_start, context)

            messages_payload = [m.to_openai_format() for m in context.messages]
            
            # 状态累积器 (每轮开始前重置)
            collected_content = []
            tool_calls_data: List[Dict[str, Any]] = []

            # 1. 消费流 (Inner Loop: Streaming Consumer)
            # 负责将 LLM 的 Token 实时透传给用户，并累积工具调用信息
            async for chunk in self.model.astream(messages=messages_payload, **llm_params): # type: ignore
                delta = self._extract_delta(chunk)

                # A. 文本内容：实时 Yield
                content = delta.get("content")
                if content:
                    collected_content.append(content)
                    yield content

                # B. 工具调用：后台累积
                if delta.get("tool_calls"):
                    self._accumulate_tool_chunks(tool_calls_data, delta["tool_calls"])

            # 2. 组装完整消息 (Turn Completion)
            final_text = "".join(collected_content)
            assistant_msg = Message.assistant(content=final_text)
            
            # 清洗并组装工具调用
            if tool_calls_data:
                valid_calls = [
                    tc for tc in tool_calls_data 
                    if tc["function"]["name"] or tc["function"]["arguments"]
                ]
                if valid_calls:
                    assistant_msg.tool_calls = valid_calls

            # 3. 处理回合逻辑 (Decision Making)
            # 复用基类的 _process_turn_results 方法
            # 返回 True 表示 "工具已执行完毕，状态已更新，请继续下一轮 LLM 推理"
            # 返回 False 表示 "任务完成" 或 "检测到死循环/无需工具"，应退出循环
            should_continue = await self._process_turn_results(
                context, assistant_msg, response_model=None
            )

            # 如果不需要继续，跳出 while 循环，结束流式生成
            if not should_continue:
                break
            
            # 如果 should_continue 为 True，while 循环会自动进入下一轮
            # context.turn 增加，context.messages 已包含工具结果

    # ===================== 辅助逻辑 =====================
    
    def _normalize_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        [New Helper] 规范化工具调用数据结构 (Adapter Pattern)
        
        目标：将 LLM 的原始输出统一转换为 ToolBox.execute_many 所需的扁平格式。
        兼容：
        1. OpenAI 嵌套格式: {"function": {"name": "...", "arguments": "..."}}
        2. 扁平格式: {"name": "...", "arguments": {...}}
        3. 参数类型: JSON String 或 Dict
        """
        # 1. 提取 Name 和 Arguments
        func_block = tool_call.get("function")
        
        # 优先尝试 OpenAI 嵌套结构
        if func_block and isinstance(func_block, dict):
            name = func_block.get("name")
            raw_args = func_block.get("arguments", "{}")
        else:
            # 降级尝试扁平结构
            name = tool_call.get("name")
            raw_args = tool_call.get("arguments", "{}")

        # 2. 安全解析参数 (JSON String -> Dict)
        if isinstance(raw_args, str):
            try:
                # 处理常见的 JSON 格式问题 (如包含换行符)
                parsed_args = json.loads(raw_args)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool arguments for '{name}': {raw_args}")
                # 解析失败返回空字典，让 ToolBox 抛出参数校验错误，而不是在这里崩溃
                parsed_args = {} 
        else:
            parsed_args = raw_args if isinstance(raw_args, dict) else {}

        # 3. 返回标准扁平格式
        return {
            "id": tool_call.get("id", ""), # ID 丢失也允许执行
            "name": name,
            "arguments": parsed_args,
        }

    async def _execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]], context: ExecutionContext
    ):
        """
        批量执行工具
        
        改进：使用 _normalize_tool_call 解耦数据清洗逻辑
        """
        # 1. 数据规范化 (Normalization)
        flat_tool_calls = [self._normalize_tool_call(tc) for tc in tool_calls]

        # 2. 批量执行 (Execution)
        results = await self.toolbox.execute_many(flat_tool_calls)

        # 3. 处理结果与副作用 (Side Effects)
        for res in results:
            if self.on_tool_execute:
                await ensure_awaitable(self.on_tool_execute, res.tool_name, {})

            content = res.result
            # 观测值截断
            if len(content) > self.max_observation_length:
                logger.warning(
                    "Truncating tool output",
                    tool=res.tool_name,
                    original_len=len(content),
                    limit=self.max_observation_length,
                )
                content = (
                    content[: self.max_observation_length]
                    + f"\n...(truncated, total {len(content)} chars)"
                )

            tool_msg = Message.tool_result(
                tool_call_id=res.call_id,
                content=content,
                tool_name=res.tool_name,
            )
            context.add_message(tool_msg)

            # 错误反馈策略
            if res.is_error:
                context.consecutive_tool_error_count += 1
                if context.consecutive_tool_error_count >= 3:
                    context.add_message(
                        Message.user(
                            "System: Too many tool errors. Please stop using this tool or change parameters."
                        )
                    )
            else:
                context.consecutive_tool_error_count = 0

    
    def _detect_infinite_loop(
        self, message: Message, context: ExecutionContext
    ) -> bool:
        """检测是否连续以相同参数调用同一工具"""
        if not message.tool_calls:
            return False

        try:
            # 计算工具调用的指纹 (Name + Args)
            calls_dump = json.dumps(
                [
                    {
                        "name": tc["function"]["name"],
                        "args": tc["function"]["arguments"],
                    }
                    for tc in message.tool_calls
                ],
                sort_keys=True,
            )
            current_hash = hash(calls_dump)

            if context.last_tool_calls_hash == current_hash:
                logger.warning("Infinite tool loop detected", calls=calls_dump)
                return True

            context.last_tool_calls_hash = current_hash
            return False
        except Exception:
            # 如果 JSON 解析失败或其他错误，保守放行
            return False

    async def _trigger_turn_end(self, context: ExecutionContext):
        """触发 Turn End Hook"""
        if self.on_turn_end:
            await ensure_awaitable(self.on_turn_end, context)

    def _accumulate_tool_chunks(self, target_list: List[Dict], chunks: List[Dict]):
        """
        合并流式工具调用片段 (OpenAI Protocol)
        """
        for tc_chunk in chunks:
            index = tc_chunk.get("index")
            
            # 确保列表长度足够
            if index is not None:
                while len(target_list) <= index:
                    target_list.append(
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    )
            
            # 获取目标引用
            target = (
                target_list[index]
                if index is not None and index < len(target_list)
                else (target_list[-1] if target_list else None)
            )

            if target:
                if tc_chunk.get("id"):
                    target["id"] += tc_chunk["id"]
                
                func_chunk = tc_chunk.get("function", {})
                if func_chunk.get("name"):
                    target["function"]["name"] += func_chunk["name"]
                if func_chunk.get("arguments"):
                    target["function"]["arguments"] += func_chunk["arguments"]

    # ----------------- 上下文构建 -----------------

    async def _build_execution_context(
        self, input_messages: List[Message]
    ) -> ExecutionContext:
        """加载历史并构建包含 System Prompt 的上下文"""
        history = await self._load_history()

        system_msg = None
        # 检查是否已存在 System Prompt
        has_system = any(m.role == "system" for m in input_messages) or any(
            m.role == "system" for m in history
        )

        if not has_system:
            # 动态渲染 System Prompt
            template_vars = {
                "tools": self.toolbox.to_openai_schema(),
                "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            system_content = self.prompt_template.format_safe(**template_vars)
            system_msg = Message.system(system_content)

        all_messages = []
        if system_msg:
            all_messages.append(system_msg)
        all_messages.extend(history)
        all_messages.extend(input_messages)

        return ExecutionContext(all_messages)

    async def _load_history(self) -> List[Message]:
        """从 Memory 加载历史记录"""
        if not self.memory.storage:
            return []
        try:
            data = await self.memory.storage.get(self.memory.session_id)
            if data and "messages" in data:
                return await self.memory.get_history(data["messages"])
        except Exception:
            return []
        return []

    async def _save_context(self, context: ExecutionContext):
        """保存上下文到 Memory"""
        if not self.memory.storage:
            return
        try:
            messages_data = [m.to_openai_format() for m in context.messages]
            await self.memory.storage.set(
                self.memory.session_id, {"messages": messages_data}
            )
        except Exception as e:
            logger.warning("Failed to save context", error=str(e))

    # ----------------- LLM 交互 -----------------

    def _build_llm_params(
        self, response_model: Optional[Type[T]], strategy: str
    ) -> Dict[str, Any]:
        """构建传递给 LLM 的参数"""
        params: Dict[str, Any] = {}
        tools_schema = self.toolbox.to_openai_schema()

        if tools_schema and self._supports_functions:
            params["tools"] = tools_schema
            params["tool_choice"] = "auto"

        if response_model:
            if strategy in {"auto", "function_calling"} and self._supports_functions:
                structure_tool = StructureEngine.to_openai_tool(response_model)
                existing_names = {
                    t["function"]["name"] for t in params.get("tools", [])
                }
                if structure_tool["function"]["name"] not in existing_names:
                    params.setdefault("tools", []).append(structure_tool)
                params["tool_choice"] = "auto"
            else:
                params["response_format"] = {"type": "json_object"}

        return params

    def _augment_messages_for_structure(
        self, messages: List[Message], response_model: Optional[Type[T]]
    ) -> List[Message]:
        """注入结构化输出引导指令"""
        if not response_model or not self._supports_functions:
            return messages

        structure_tool_name = StructureEngine.to_openai_tool(response_model)[
            "function"
        ]["name"]
        instruction = (
            f"\nIMPORTANT: You MUST call the '{structure_tool_name}' function to provide your final answer. "
            "Do not reply with text only."
        )

        augmented = messages.copy()
        # 尝试追加到最后一条 User 消息
        for i in range(len(augmented) - 1, -1, -1):
            if augmented[i].role == "user":
                original = augmented[i]
                new_content = str(original.content) + instruction
                augmented[i] = Message(
                    role="user", content=new_content, name=original.name
                )
                return augmented

        # 否则新增一条 User 消息
        augmented.append(Message.user(instruction))
        return augmented

    async def _call_llm(self, context: ExecutionContext, params: Dict[str, Any]) -> Any:
        """调用 LLM API"""
        messages_payload = [m.to_openai_format() for m in context.messages]
        try:
            return await ensure_awaitable(
                self.model.acompletion, messages=messages_payload, **params
            )
        except Exception as e:
            raise ModelError(f"LLM API call failed: {e}") from e

    def _parse_llm_response(self, response: Any) -> Message:
        """解析 LLM 响应为 Message 对象"""
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            message_data = choice.message
            if hasattr(message_data, "model_dump"):
                message_data = message_data.model_dump()
            elif hasattr(message_data, "to_dict"):
                message_data = message_data.to_dict()
            elif not isinstance(message_data, dict):
                message_data = {
                    "role": getattr(message_data, "role", "assistant"),
                    "content": getattr(message_data, "content", ""),
                    "tool_calls": getattr(message_data, "tool_calls", None),
                }

            if "tool_calls" in message_data and message_data["tool_calls"] is None:
                del message_data["tool_calls"]

            return Message(**message_data)

        raise ModelError("Invalid LLM response format")

    def _extract_delta(self, chunk: Any) -> Dict[str, Any]:
        """从 Stream Chunk 中提取 delta"""
        if hasattr(chunk, "choices") and chunk.choices:
            choice = chunk.choices[0]
            if isinstance(choice, dict):
                return choice.get("delta", {})
            return getattr(choice, "delta", {})
        return {}

    def _is_structure_extraction(self, message: Message, model_class: Type[T]) -> bool:
        """判断当前是否为结构化提取的工具调用"""
        if not message.tool_calls or not self._supports_functions:
            return False
        extraction_tool_name = StructureEngine.to_openai_tool(model_class)["function"][
            "name"
        ]
        return any(
            tc.get("function", {}).get("name") == extraction_tool_name
            for tc in message.tool_calls
        )

    async def _handle_structured_output(
        self,
        output: AgentOutput,
        response_model: Type[T],
        context: ExecutionContext,
        llm_params: Dict[str, Any],
        max_retries: int,
    ) -> T: # type: ignore
        """处理结构化输出解析与自动重试"""
        expected_tool_name = StructureEngine.to_openai_tool(response_model)["function"]["name"]

        for attempt in range(max_retries + 1):
            try:
                # ------------------------------------------------------------------
                # [增强逻辑] 智能定位目标工具调用
                # ------------------------------------------------------------------
                target_tool_call = None
                
                if output.tool_calls and self._supports_functions:
                    # 1. 尝试在所有工具调用中寻找匹配预期的那个
                    for tc in output.tool_calls:
                        name = tc.get("function", {}).get("name")
                        if name == expected_tool_name:
                            target_tool_call = tc
                            break
                    
                    # 2. 如果有工具调用，但没一个是目标工具 -> 判定为推理中断/死循环/逻辑错误
                    if not target_tool_call:
                        # 获取第一个工具名用于报错信息
                        first_tool = output.tool_calls[0].get("function", {}).get("name")
                        raise StructureParseError(
                            f"Incorrect tool used. Expected final tool '{expected_tool_name}', "
                            f"but detected intermediate tool(s) like '{first_tool}'. "
                            "Execution stopped prematurely (e.g., infinite loop or max turns reached). "
                            "Please directly call the final tool with the answer."
                        )
                
                # ------------------------------------------------------------------
                # 解析逻辑 (传入过滤后的 target_tool_call)
                # ------------------------------------------------------------------
                # 注意：如果是纯文本，target_tool_call 为 None，parse 会尝试从 content 提取
                return await StructureEngine.parse(
                    content=output.content,
                    model_class=response_model,
                    # 关键：只传给解析器它关心的那个工具调用，或者 None
                    raw_tool_calls=[target_tool_call] if target_tool_call else None,
                    auto_fix=True,
                )

            except StructureParseError as e:
                if attempt >= max_retries:
                    raise AgentError(f"Failed to parse structured output after {max_retries} retries: {e}")

                # 构造反馈消息 (保持不变)
                feedback_msg = Message.user(
                    f"Error parsing response: {e}. Please ensure you call the '{expected_tool_name}' function correctly."
                )
                context.add_message(feedback_msg)
                
                # 重新生成
                response = await self._call_llm(context, llm_params)
                msg = self._parse_llm_response(response)
                context.add_message(msg)
                output = AgentOutput(
                    content=msg.content or "", tool_calls=msg.tool_calls or [] # type: ignore
                )