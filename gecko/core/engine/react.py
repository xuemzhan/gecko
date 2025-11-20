# gecko/core/engine/react.py
"""
ReActEngine 重构版（Phase 2 增强）

改进：
1. 更细粒度的方法拆分
2. 每个方法职责单一
3. 便于单元测试
4. 支持自定义 Hook
"""
from __future__ import annotations
import asyncio
import json
from typing import List, Dict, Any, Optional, Type, TypeVar, AsyncIterator, Callable

from pydantic import BaseModel

from gecko.core.engine.base import CognitiveEngine
from gecko.core.message import Message
from gecko.core.output import AgentOutput
from gecko.core.utils import ensure_awaitable
from gecko.core.prompt import PromptTemplate
from gecko.core.structure import StructureEngine
from gecko.core.logging import get_logger
from gecko.core.exceptions import AgentError, ModelError

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_REACT_TEMPLATE = """You are a helpful AI assistant.
Available Tools:
{% for tool in tools %}
- {{ tool.function.name }}: {{ tool.function.description }}
{% endfor %}

Answer the user's request. Use tools if necessary.
"""

# ========== 执行上下文 ==========

class ExecutionContext:
    """
    执行上下文对象
    
    新增：将执行状态封装为对象，便于传递和扩展
    """
    def __init__(self, messages: List[Message]):
        self.messages = messages
        self.turn = 0
        self.metadata: Dict[str, Any] = {}
    
    def add_message(self, message: Message):
        """添加消息到上下文"""
        self.messages.append(message)
    
    def get_last_message(self) -> Optional[Message]:
        """获取最后一条消息"""
        return self.messages[-1] if self.messages else None

# ========== ReActEngine ==========

class ReActEngine(CognitiveEngine):
    """
    ReAct 引擎（Phase 2 重构版）
    
    核心改进：
    1. 方法平均长度 < 20 行
    2. 每个方法可独立测试
    3. 支持执行钩子（便于扩展）
    """
    
    def __init__(
        self,
        model,
        toolbox,
        memory,
        max_turns: int = 5,
        system_prompt: str | PromptTemplate = None,
        # ✅ 新增：钩子支持
        on_turn_start: Optional[Callable] = None,
        on_turn_end: Optional[Callable] = None,
        on_tool_execute: Optional[Callable] = None,
    ):
        super().__init__(model, toolbox, memory)
        self.max_turns = max_turns
        
        # 钩子函数
        self.on_turn_start = on_turn_start
        self.on_turn_end = on_turn_end
        self.on_tool_execute = on_tool_execute
        
        # 初始化 Prompt
        if system_prompt is None:
            self.prompt_template = PromptTemplate(template=DEFAULT_REACT_TEMPLATE)
        elif isinstance(system_prompt, str):
            self.prompt_template = PromptTemplate(template=system_prompt)
        else:
            self.prompt_template = system_prompt

    # ========== 公开 API ==========

    async def step(
        self,
        input_messages: List[Message],
        response_model: Optional[Type[T]] = None,
        strategy: str = "auto",
        max_retries: int = 2
    ) -> AgentOutput | T:
        """
        执行推理步骤
        
        流程：
        1. 构建上下文
        2. 执行推理循环
        3. 提取结构化输出（如需要）
        4. 保存记忆
        """
        logger.info(
            "ReAct execution started",
            input_count=len(input_messages),
            has_structure=response_model is not None,
        )
        
        try:
            # 1. 构建初始上下文
            context = await self._build_execution_context(input_messages)
            
            # 2. 准备 LLM 参数
            llm_params = self._build_llm_params(response_model, strategy)
            
            # 3. 执行推理循环
            final_output = await self._run_reasoning_loop(
                context,
                llm_params,
                response_model,
                strategy
            )
            
            # 4. 处理结构化输出
            if response_model:
                structured_result = await self._extract_and_retry(
                    final_output,
                    response_model,
                    context,
                    llm_params,
                    max_retries
                )
                await self._save_context(context)
                return structured_result
            
            # 5. 保存记忆
            await self._save_context(context)
            
            logger.info("ReAct execution completed")
            return final_output
        
        except Exception as e:
            logger.exception("ReAct execution failed")
            if isinstance(e, AgentError):
                raise
            raise AgentError(f"ReAct execution failed: {e}") from e

    async def step_stream(
        self,
        input_messages: List[Message]
    ) -> AsyncIterator[str]:
        """
        流式执行
        
        策略：
        1. 快速检测是否需要工具
        2. 如需要，完成工具调用后再流式输出
        3. 如不需要，直接流式输出
        """
        context = await self._build_execution_context(input_messages)
        llm_params = self._build_llm_params(None, "auto")
        
        turn = 0
        while turn < self.max_turns:
            turn += 1
            
            # 检测是否需要工具
            needs_tools = await self._check_needs_tools(context, llm_params)
            
            if needs_tools:
                # 执行工具后继续
                await self._execute_one_turn(context, llm_params, None, "auto")
                continue
            else:
                # 流式输出最终回复
                async for chunk in self._stream_final_response(context, llm_params):
                    yield chunk
                break
        
        await self._save_context(context)

    # ========== 上下文构建 ==========

    async def _build_execution_context(
        self,
        input_messages: List[Message]
    ) -> ExecutionContext:
        """
        构建执行上下文
        
        步骤：
        1. 加载历史记忆
        2. 注入 System Prompt
        3. 添加用户输入
        """
        # 1. 加载历史
        history = await self._load_history()
        
        # 2. 确保有 System Prompt
        if not any(m.role == "system" for m in history):
            system_msg = self._create_system_message()
            history.insert(0, system_msg)
        
        # 3. 合并
        all_messages = history + input_messages
        
        return ExecutionContext(all_messages)

    async def _load_history(self) -> List[Message]:
        """加载历史记忆"""
        if not self.memory.storage or not self.memory.session_id:
            return []
        
        try:
            raw_data = await self.memory.storage.get(self.memory.session_id)
            if not raw_data:
                return []
            
            history_raw = raw_data.get("messages", [])
            return await self.memory.get_history(history_raw)
        except Exception as e:
            logger.warning("Failed to load history", error=str(e))
            return []

    def _create_system_message(self) -> Message:
        """创建 System 消息"""
        tools_schema = self.toolbox.to_openai_schema()
        content = self.prompt_template.format(tools=tools_schema)
        return Message.system(content)

    # ========== 参数构建 ==========

    def _build_llm_params(
        self,
        response_model: Optional[Type[T]],
        strategy: str
    ) -> Dict[str, Any]:
        """
        构建 LLM 调用参数
        
        根据是否需要结构化输出调整参数
        """
        params: Dict[str, Any] = {}
        
        # 基础工具
        tools = self.toolbox.to_openai_schema()
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"
        
        # 结构化输出参数
        if response_model:
            self._add_structure_params(params, response_model, strategy)
        
        return params

    def _add_structure_params(
        self,
        params: Dict[str, Any],
        response_model: Type[T],
        strategy: str
    ):
        """添加结构化输出参数"""
        if strategy in ["auto", "function_calling"]:
            # Function Calling 模式
            structure_tool = StructureEngine.to_openai_tool(response_model)
            if "tools" not in params:
                params["tools"] = []
            params["tools"].append(structure_tool)
            params["tool_choice"] = {
                "type": "function",
                "function": {"name": structure_tool["function"]["name"]}
            }
        elif strategy == "json_mode":
            # JSON Mode
            params["response_format"] = {"type": "json_object"}

    # ========== 推理循环 ==========

    async def _run_reasoning_loop(
        self,
        context: ExecutionContext,
        llm_params: Dict[str, Any],
        response_model: Optional[Type[T]],
        strategy: str
    ) -> AgentOutput:
        """
        主推理循环
        
        每轮：
        1. 调用 LLM
        2. 检查工具调用
        3. 执行工具（如有）
        4. 判断是否继续
        """
        while context.turn < self.max_turns:
            context.turn += 1
            
            # 调用钩子
            if self.on_turn_start:
                await self._safe_call_hook(self.on_turn_start, context)
            
            logger.debug("Turn started", turn=context.turn)
            
            # 执行一轮
            should_continue = await self._execute_one_turn(
                context,
                llm_params,
                response_model,
                strategy
            )
            
            # 调用钩子
            if self.on_turn_end:
                await self._safe_call_hook(self.on_turn_end, context)
            
            if not should_continue:
                break
        
        # 返回最终输出
        last_msg = context.get_last_message()
        if last_msg and last_msg.role == "assistant":
            return AgentOutput(
                content=last_msg.content or "",
                raw=context.metadata.get("last_response")
            )
        
        logger.warning("Max turns reached", max_turns=self.max_turns)
        return AgentOutput(content="Max iterations reached.")

    async def _execute_one_turn(
        self,
        context: ExecutionContext,
        llm_params: Dict[str, Any],
        response_model: Optional[Type[T]],
        strategy: str
    ) -> bool:
        """
        执行单轮推理
        
        返回：是否应该继续下一轮
        """
        # 1. 调用 LLM
        response = await self._call_llm(context, llm_params)
        
        # 2. 解析响应
        assistant_msg = self._parse_llm_response(response)
        context.add_message(assistant_msg)
        context.metadata["last_response"] = response
        
        # 3. 检查结构化提取
        if response_model and strategy in ["auto", "function_calling"]:
            if self._is_structure_extraction(assistant_msg, response_model):
                return False  # 提取完成，结束循环
        
        # 4. 执行工具
        if assistant_msg.tool_calls:
            has_tools = await self._execute_tools(
                assistant_msg.tool_calls,
                context,
                response_model
            )
            return has_tools  # 有工具执行则继续
        
        # 5. 生成了最终回复
        return False

    # ========== LLM 调用 ==========

    async def _call_llm(
        self,
        context: ExecutionContext,
        params: Dict[str, Any]
    ) -> Any:
        """
        调用 LLM API
        
        处理：
        1. 序列化消息
        2. 调用模型
        3. 错误处理
        """
        messages_payload = [m.to_openai_format() for m in context.messages]
        
        logger.debug("Calling LLM", message_count=len(messages_payload))
        
        try:
            response = await ensure_awaitable(
                self.model.acompletion,
                messages=messages_payload,
                **params
            )
            return response
        except Exception as e:
            logger.error("LLM call failed", error=str(e))
            raise ModelError(f"LLM API call failed: {e}") from e

    def _parse_llm_response(self, response: Any) -> Message:
        """
        解析 LLM 响应为 Message
        
        处理：
        1. 提取消息数据
        2. 确保必需字段
        3. 构建 Message 对象
        """
        choice = response.choices[0]
        raw_msg = choice.message
        
        # 安全提取数据
        msg_data: Dict[str, Any] = {}
        
        if hasattr(raw_msg, "model_dump"):
            try:
                msg_data = raw_msg.model_dump()
            except Exception:
                pass
        
        if not msg_data:
            msg_data = {
                "content": getattr(raw_msg, "content", ""),
                "tool_calls": getattr(raw_msg, "tool_calls", None),
            }
        
        # 确保 role 存在
        if "role" not in msg_data or not msg_data["role"]:
            logger.warning("Missing role in LLM response")
            msg_data["role"] = "assistant"
        
        return Message(**msg_data)

    # ========== 工具执行 ==========

    async def _execute_tools(
        self,
        tool_calls: List[Dict],
        context: ExecutionContext,
        response_model: Optional[Type[T]]
    ) -> bool:
        """
        执行工具调用
        
        返回：是否有真实工具执行（排除结构化提取工具）
        """
        # 获取结构化提取工具名（如果有）
        extraction_tool_name = None
        if response_model:
            extraction_tool_name = StructureEngine.to_openai_tool(
                response_model
            )["function"]["name"]
        
        has_real_execution = False
        
        for tool_call in tool_calls:
            # 跳过结构化提取工具
            if tool_call["function"]["name"] == extraction_tool_name:
                continue
            
            # 执行工具
            result = await self._execute_single_tool(tool_call)
            
            # 添加结果到上下文
            tool_msg = Message.tool_result(
                tool_call_id=tool_call["id"],
                content=result["content"],
                tool_name=tool_call["function"]["name"]
            )
            context.add_message(tool_msg)
            
            has_real_execution = True
        
        return has_real_execution

    async def _execute_single_tool(self, tool_call: Dict) -> Dict[str, str]:
        """
        执行单个工具
        
        返回：{"content": str, "is_error": bool}
        """
        func_name = tool_call["function"]["name"]
        args_str = tool_call["function"]["arguments"]
        
        logger.debug("Executing tool", tool=func_name)
        
        try:
            # 解析参数
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
            
            # 调用钩子
            if self.on_tool_execute:
                await self._safe_call_hook(self.on_tool_execute, func_name, args)
            
            # 执行工具
            content = await self.toolbox.execute(func_name, args)
            
            logger.info("Tool executed", tool=func_name, success=True)
            return {"content": str(content), "is_error": False}
        
        except Exception as e:
            logger.error("Tool execution failed", tool=func_name, error=str(e))
            return {
                "content": f"Error executing {func_name}: {str(e)}",
                "is_error": True
            }

    # ========== 结构化输出 ==========

    def _is_structure_extraction(
        self,
        message: Message,
        model_class: Type[T]
    ) -> bool:
        """检查是否为结构化提取的 Tool Call"""
        if not message.tool_calls:
            return False
        
        extraction_name = StructureEngine.to_openai_tool(model_class)["function"]["name"]
        return any(
            tc["function"]["name"] == extraction_name
            for tc in message.tool_calls
        )

    async def _extract_and_retry(
        self,
        base_output: AgentOutput,
        model_class: Type[T],
        context: ExecutionContext,
        llm_params: Dict[str, Any],
        max_retries: int
    ) -> T:
        """
        提取结构化输出（带重试）
        
        流程：
        1. 尝试解析
        2. 失败则反馈错误给 LLM
        3. 重新生成
        4. 重复直到成功或达到最大重试次数
        """
        for retry in range(max_retries + 1):
            try:
                # 尝试解析
                tool_calls = self._extract_tool_calls(base_output)
                
                result = await StructureEngine.parse(
                    content=base_output.content,
                    model_class=model_class,
                    raw_tool_calls=tool_calls
                )
                
                logger.info("Structured output extracted", retries=retry)
                return result
            
            except ValueError as e:
                if retry >= max_retries:
                    logger.error("Structure extraction failed", retries=retry)
                    raise
                
                logger.info("Retrying structure extraction", attempt=retry + 1)
                
                # 反馈错误
                base_output = await self._retry_with_feedback(
                    str(e),
                    context,
                    llm_params
                )
        
        raise AgentError("Structure extraction failed unexpectedly")

    def _extract_tool_calls(self, output: AgentOutput) -> Optional[List[Dict]]:
        """从 AgentOutput 中提取 tool_calls"""
        if not output.raw:
            return None
        
        try:
            return getattr(
                output.raw.choices[0].message,
                "tool_calls",
                None
            )
        except (AttributeError, IndexError):
            return None

    async def _retry_with_feedback(
        self,
        error_message: str,
        context: ExecutionContext,
        llm_params: Dict[str, Any]
    ) -> AgentOutput:
        """带错误反馈的重试"""
        # 添加错误反馈
        feedback_msg = Message(
            role="user",
            content=f"Parsing error: {error_message}. Please format as valid JSON."
        )
        context.add_message(feedback_msg)
        
        # 重新生成
        response = await self._call_llm(context, llm_params)
        new_msg = self._parse_llm_response(response)
        context.add_message(new_msg)
        
        return AgentOutput(
            content=new_msg.content or "",
            raw=response
        )

    # ========== 流式相关 ==========

    async def _check_needs_tools(
        self,
        context: ExecutionContext,
        llm_params: Dict[str, Any]
    ) -> bool:
        """快速检测是否需要工具"""
        # 使用低 max_tokens 快速检测
        peek_params = {**llm_params, "max_tokens": 50}
        
        try:
            response = await self._call_llm(context, peek_params)
            msg = self._parse_llm_response(response)
            return bool(msg.tool_calls)
        except Exception:
            return False

    async def _stream_final_response(
        self,
        context: ExecutionContext,
        llm_params: Dict[str, Any]
    ) -> AsyncIterator[str]:
        """流式输出最终响应"""
        messages_payload = [m.to_openai_format() for m in context.messages]
        
        accumulated = []
        
        try:
            async for chunk in self.model.astream(messages=messages_payload, **llm_params):
                delta = chunk.choices[0].delta
                if delta.content:
                    accumulated.append(delta.content)
                    yield delta.content
        except Exception as e:
            logger.error("Streaming failed", error=str(e))
            raise
        
        # 保存完整内容
        full_content = "".join(accumulated)
        context.add_message(Message.assistant(full_content))

    # ========== 记忆保存 ==========

    async def _save_context(self, context: ExecutionContext):
        """保存执行上下文到记忆"""
        if not self.memory.storage or not self.memory.session_id:
            return
        
        try:
            messages_data = [m.model_dump() for m in context.messages]
            await self.memory.storage.set(
                self.memory.session_id,
                {"messages": messages_data}
            )
            logger.debug("Context saved", message_count=len(messages_data))
        except Exception as e:
            logger.error("Failed to save context", error=str(e))
            # 不抛出异常，避免中断主流程

    # ========== 工具方法 ==========

    async def _safe_call_hook(self, hook: Callable, *args, **kwargs):
        """安全调用钩子函数"""
        try:
            if asyncio.iscoroutinefunction(hook):
                await hook(*args, **kwargs)
            else:
                hook(*args, **kwargs)
        except Exception as e:
            logger.warning("Hook execution failed", error=str(e))
            # 钩子失败不应该中断主流程