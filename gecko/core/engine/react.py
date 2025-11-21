# gecko/core/engine/react.py
"""
ReActEngine - 推理与行动引擎

优化日志：
1. 集成 StructureEngine 的新 API，移除冗余的解析逻辑
2. 修复 ExecutionStats 统计缺失问题
3. 优化 ExecutionContext 上下文管理
4. 增强工具执行的错误反馈机制
5. 统一普通推理和流式推理的底层逻辑
6. 修复流式输出记忆丢失问题
7. 修复结构化输出时 tool_choice 过于严格导致无法调用其他工具的问题
8. 增加 Prompt 引导以强制结构化输出工具调用
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

from pydantic import BaseModel

from gecko.core.engine.base import CognitiveEngine
from gecko.core.output import AgentOutput
from gecko.core.exceptions import AgentError, ModelError
from gecko.core.logging import get_logger
from gecko.core.message import Message
from gecko.core.memory import TokenMemory
from gecko.core.prompt import PromptTemplate
from gecko.core.structure import StructureEngine, StructureParseError
from gecko.core.toolbox import ToolBox
from gecko.core.utils import ensure_awaitable

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

DEFAULT_REACT_TEMPLATE = """You are a helpful AI assistant.
Available Tools:
{% for tool in tools %}
- {{ tool.function.name }}: {{ tool.function.description }}
{% endfor %}

Answer the user's request. Use tools if necessary.
"""


class ExecutionContext:
    """
    执行上下文：封装每一轮 ReAct 循环的状态
    """

    def __init__(self, messages: List[Message]):
        self.messages = messages.copy()  # 浅拷贝，避免污染原始列表
        self.turn = 0
        self.metadata: Dict[str, Any] = {}

    def add_message(self, message: Message):
        self.messages.append(message)

    def get_last_message(self) -> Optional[Message]:
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
        system_prompt: str | PromptTemplate | None = None,
        on_turn_start: Optional[Callable[[ExecutionContext], Any]] = None,
        on_turn_end: Optional[Callable[[ExecutionContext], Any]] = None,
        on_tool_execute: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        **kwargs,
    ):
        super().__init__(model, toolbox, memory, **kwargs)
        self.max_turns = max_turns
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
        # 优先检查显式属性，其次检查方法
        if hasattr(self.model, "_supports_function_calling"):
            return getattr(self.model, "_supports_function_calling")
        # 默认假设实现了 ModelProtocol 且没有抛错
        return True

    def _check_streaming_support(self) -> bool:
        return hasattr(self.model, "astream")

    # ===================== 核心接口实现 =====================

    async def step(
        self,
        input_messages: List[Message],
        response_model: Optional[Type[T]] = None,
        strategy: str = "auto",
        max_retries: int = 2,
        **kwargs,
    ) -> AgentOutput | T:
        """
        核心推理入口
        """
        start_time = time.time()

        # 1. 验证输入
        self.validate_input(input_messages)

        # 2. Hook: Before Step
        await self.before_step(input_messages, **kwargs)

        logger.info(
            "ReAct execution started",
            input_count=len(input_messages),
            has_structure=response_model is not None,
        )

        try:
            # 增强结构化输出的 Prompt 引导
            augmented_messages = input_messages
            if response_model and self._supports_functions:
                structure_tool_name = StructureEngine.to_openai_tool(response_model)[
                    "function"
                ]["name"]
                instruction = (
                    f"\nIMPORTANT: You MUST call the '{structure_tool_name}' function to provide your final answer. "
                    "Do not reply with text only."
                )
                augmented_messages = input_messages.copy()
                augmented_messages.append(Message.user(instruction))

            # 3. 构建上下文
            context = await self._build_execution_context(augmented_messages)

            # 4. 构建 LLM 参数
            llm_params = self._build_llm_params(response_model, strategy)
            llm_params.update(kwargs)  # 合并额外参数

            # 5. 运行推理循环 (ReAct Loop)
            final_output = await self._run_reasoning_loop(
                context, llm_params, response_model
            )

            # 6. 结构化输出处理 (如果需要)
            result = final_output
            if response_model:
                result = await self._handle_structured_output(
                    final_output, response_model, context, llm_params, max_retries
                )

            # 7. 保存上下文到记忆
            # 注意：保存的是原始 context，包含为了引导而添加的指令
            # 如果不希望污染长期记忆，可以在这里做过滤，但为了简单起见暂且保留
            await self._save_context(context)

            # 8. Hook: After Step
            # 注意：如果返回的是结构化对象，我们需要包装成 AgentOutput 给 hook
            hook_output = (
                result
                if isinstance(result, AgentOutput)
                else AgentOutput(content=str(result), raw=result)
            )
            await self.after_step(input_messages, hook_output, **kwargs)

            # 9. 统计更新
            duration = time.time() - start_time
            if self.stats:
                self.stats.add_step(duration)

            return result

        except Exception as e:
            if self.stats:
                self.stats.errors += 1
            logger.exception("ReAct execution failed")
            await self.on_error(e, input_messages, **kwargs)
            raise

    async def step_stream(
        self, input_messages: List[Message], **kwargs
    ) -> AsyncIterator[str]:
        """
        流式推理入口
        """
        if not self._supports_stream:
            raise AgentError("当前模型不支持流式输出")

        start_time = time.time()
        await self.before_step(input_messages, **kwargs)

        context = await self._build_execution_context(input_messages)
        llm_params = self._build_llm_params(None, "auto")
        llm_params.update(kwargs)

        try:
            # 1. Peek 检查
            needs_tools, peek_response = await self._check_needs_tools(
                context, llm_params
            )

            if needs_tools:
                # 如果需要工具，先在内部跑完 ReAct 循环
                await self._execute_one_turn(context, llm_params, None)
                # 工具执行完后，生成最终回复，此时可以流式
            elif peek_response:
                # 如果 Peek 已经拿到了文本回复，直接 yield
                msg = self._parse_llm_response(peek_response)
                yield msg.content or ""
                return

            # 2. 流式输出最终回复
            async for chunk in self._stream_final_response(context, llm_params):
                yield chunk

            # 3. 统计与保存
            await self._save_context(context)

            duration = time.time() - start_time
            if self.stats:
                self.stats.add_step(duration)

        except Exception as e:
            if self.stats:
                self.stats.errors += 1
            logger.exception("ReAct stream failed")
            await self.on_error(e, input_messages, **kwargs)
            raise

    # ===================== 上下文与参数构建 =====================

    async def _build_execution_context(
        self, input_messages: List[Message]
    ) -> ExecutionContext:
        """构建包含历史记录和系统提示词的上下文"""
        # 加载历史
        history = await self._load_history()

        # 准备系统消息
        system_msg = None
        has_system = any(m.role == "system" for m in input_messages) or any(
            m.role == "system" for m in history
        )

        if not has_system:
            system_content = self.prompt_template.format(
                tools=self.toolbox.to_openai_schema()
            )
            system_msg = Message.system(system_content)

        # 组合消息: System + History + Input
        all_messages = []
        if system_msg:
            all_messages.append(system_msg)
        all_messages.extend(history)
        all_messages.extend(input_messages)

        return ExecutionContext(all_messages)

    def _build_llm_params(
        self, response_model: Optional[Type[T]], strategy: str
    ) -> Dict[str, Any]:
        """构建传递给 LLM 的参数 (Tools, Output Format)"""
        params: Dict[str, Any] = {}

        # 1. 注入工具
        tools_schema = self.toolbox.to_openai_schema()
        if tools_schema and self._supports_functions:
            params["tools"] = tools_schema
            params["tool_choice"] = "auto"

        # 2. 注入结构化输出要求
        if response_model:
            if strategy in {"auto", "function_calling"} and self._supports_functions:
                # 使用 Tool Calling 方式提取结构
                structure_tool = StructureEngine.to_openai_tool(response_model)
                params.setdefault("tools", []).append(structure_tool)
                params["tool_choice"] = "auto"
            else:
                # 使用 JSON Mode
                params["response_format"] = {"type": "json_object"}
                if not self._supports_functions:
                    logger.warning("模型不支持 Function Calling，降级为 JSON Mode")

        return params

    # ===================== 推理循环 (The Loop) =====================

    async def _run_reasoning_loop(
        self,
        context: ExecutionContext,
        llm_params: Dict[str, Any],
        response_model: Optional[Type[T]],
    ) -> AgentOutput:
        """ReAct 主循环"""

        while context.turn < self.max_turns:
            context.turn += 1

            if self.on_turn_start:
                await ensure_awaitable(self.on_turn_start, context)

            should_continue = await self._execute_one_turn(
                context, llm_params, response_model
            )

            if self.on_turn_end:
                await ensure_awaitable(self.on_turn_end, context)

            if not should_continue:
                break

        last_msg = context.get_last_message()
        if not last_msg:
            return AgentOutput(content="No response generated.")

        return AgentOutput(
            content=last_msg.content or "",
            raw=context.metadata.get("last_response"),
            tool_calls=last_msg.tool_calls or [],
        )

    async def _execute_one_turn(
        self,
        context: ExecutionContext,
        llm_params: Dict[str, Any],
        response_model: Optional[Type[T]],
    ) -> bool:
        """
        执行单轮推理
        返回: should_continue (bool)
        """
        response = await self._call_llm(context, llm_params)
        context.metadata["last_response"] = response

        assistant_msg = self._parse_llm_response(response)
        context.add_message(assistant_msg)

        if response_model and self._is_structure_extraction(
            assistant_msg, response_model
        ):
            return False

        if assistant_msg.tool_calls:
            if self.stats:
                self.stats.tool_calls += len(assistant_msg.tool_calls)

            tool_executed = await self._execute_tool_calls(
                assistant_msg.tool_calls, context
            )
            return tool_executed

        return False

    # ===================== 工具执行逻辑 =====================

    async def _execute_tool_calls(
        self, tool_calls: List[Dict[str, Any]], context: ExecutionContext
    ) -> bool:
        """
        执行工具列表并将结果作为 Tool Message 加入 Context
        """
        results = await self.toolbox.execute_many(tool_calls)

        has_execution = False
        for res in results:
            has_execution = True

            if self.on_tool_execute:
                await ensure_awaitable(self.on_tool_execute, res.tool_name, {})

            tool_msg = Message.tool_result(
                tool_call_id=res.call_id,
                content=res.result,
                tool_name=res.tool_name,
            )
            context.add_message(tool_msg)

            if res.is_error:
                feedback = f"Tool '{res.tool_name}' failed: {res.result}. Please try again or use another tool."
                context.add_message(Message.user(feedback))

        return has_execution

    # ===================== 结构化输出处理 =====================

    def _is_structure_extraction(self, message: Message, model_class: Type[T]) -> bool:
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
    ) -> T:

        for attempt in range(max_retries + 1):
            try:
                return await StructureEngine.parse(
                    content=output.content,
                    model_class=response_model,
                    raw_tool_calls=output.tool_calls,
                    auto_fix=True,
                )
            except StructureParseError as e:
                if attempt >= max_retries:
                    logger.error(
                        "Structured parsing failed after retries", error=str(e)
                    )
                    raise AgentError(f"Failed to parse structured output: {e}")

                logger.warning(
                    "Parsing failed, retrying with feedback", attempt=attempt
                )

                feedback_msg = Message.user(
                    f"The previous response could not be parsed into the expected format. "
                    f"Error: {e}. Please try again."
                )
                context.add_message(feedback_msg)

                response = await self._call_llm(context, llm_params)
                msg = self._parse_llm_response(response)
                context.add_message(msg)

                output = AgentOutput(
                    content=msg.content or "",
                    tool_calls=msg.tool_calls or [],
                    raw=response,
                )

        raise AgentError("Structured parsing failed unexpectedly")

    # ===================== 辅助方法 =====================

    async def _call_llm(self, context: ExecutionContext, params: Dict[str, Any]) -> Any:
        messages_payload = [m.to_openai_format() for m in context.messages]
        try:
            return await ensure_awaitable(
                self.model.acompletion, messages=messages_payload, **params
            )
        except Exception as e:
            raise ModelError(f"LLM API call failed: {e}") from e

    def _parse_llm_response(self, response: Any) -> Message:
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

    async def _check_needs_tools(
        self, context: ExecutionContext, llm_params: Dict[str, Any]
    ) -> Tuple[bool, Optional[Any]]:
        peek_params = llm_params.copy()

        try:
            response = await self._call_llm(context, peek_params)
            msg = self._parse_llm_response(response)

            if msg.tool_calls:
                context.add_message(msg)
                return True, None

            return False, response
        except Exception:
            return False, None

    async def _stream_final_response(
        self, context: ExecutionContext, llm_params: Dict[str, Any]
    ) -> AsyncIterator[str]:
        messages_payload = [m.to_openai_format() for m in context.messages]
        full_content = []

        try:
            async for chunk in self.model.astream(
                messages=messages_payload, **llm_params
            ):
                content = None
                if hasattr(chunk, "content"):
                    content = chunk.content
                elif hasattr(chunk, "delta"):
                    content = chunk.delta.get("content")
                elif hasattr(chunk, "choices") and chunk.choices:
                    delta = (
                        chunk.choices[0].get("delta", {})
                        if isinstance(chunk.choices[0], dict)
                        else chunk.choices[0].delta
                    )
                    content = (
                        getattr(delta, "content", None)
                        if not isinstance(delta, dict)
                        else delta.get("content")
                    )

                if content:
                    full_content.append(content)
                    yield content

        finally:
            if full_content:
                final_text = "".join(full_content)
                context.add_message(Message.assistant(content=final_text))
                context.metadata["last_response"] = final_text

    async def _save_context(self, context: ExecutionContext):
        if not self.memory.storage:
            return
        try:
            messages_data = [m.to_openai_format() for m in context.messages]
            await self.memory.storage.set(
                self.memory.session_id, {"messages": messages_data}
            )
        except Exception as e:
            logger.warning("Failed to save context", error=str(e))

    async def _load_history(self) -> List[Message]:
        if not self.memory.storage:
            return []
        try:
            data = await self.memory.storage.get(self.memory.session_id)
            if data and "messages" in data:
                return await self.memory.get_history(data["messages"])
        except Exception:
            return []
        return []