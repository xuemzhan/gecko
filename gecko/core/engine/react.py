# gecko/core/engine/react.py  
"""  
ReActEngine（增强版）  
  
核心能力：  
1. 模型能力自适应：检测是否支持 function calling / stream，自动降级或提示  
2. 工具执行更健壮：校验字段、捕获错误、反馈给 LLM，防止死循环  
3. 结构化输出解析与带反馈重试  
4. 流式/非流式路径统一管理上下文  
5. Hook 机制完善，异常不影响主流程  
"""  
  
from __future__ import annotations  
  
import asyncio  
import json  
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
from gecko.core.exceptions import AgentError, ModelError  
from gecko.core.logging import get_logger  
from gecko.core.message import Message  
from gecko.core.output import AgentOutput  
from gecko.core.prompt import PromptTemplate  
from gecko.core.structure import StructureEngine  
from gecko.core.toolbox import ToolBox  
from gecko.core.utils import ensure_awaitable  
from gecko.core.memory import TokenMemory  
  
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
    执行上下文：  
    - messages：当前对话（包含历史和用户输入）  
    - turn：已经执行的轮数  
    - metadata：可存储 last_response 等额外信息  
    """  
  
    def __init__(self, messages: List[Message]):  
        self.messages = messages  
        self.turn = 0  
        self.metadata: Dict[str, Any] = {}  
  
    def add_message(self, message: Message):  
        self.messages.append(message)  
  
    def get_last_message(self) -> Optional[Message]:  
        return self.messages[-1] if self.messages else None  
  
  
class ReActEngine(CognitiveEngine):  
    """  
    ReAct 引擎（完整实现）  
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
        supports_functions: Optional[bool] = None,  
        supports_stream: Optional[bool] = None,  
    ):  
        super().__init__(model, toolbox, memory)  
        self.max_turns = max_turns  
        self.on_turn_start = on_turn_start  
        self.on_turn_end = on_turn_end  
        self.on_tool_execute = on_tool_execute  
  
        # 模型能力检测：若未显式声明，则根据方法/属性推断  
        self.supports_functions = (  
            supports_functions  
            if supports_functions is not None  
            else hasattr(model, "acompletion")  
        )  
        self.supports_stream = (  
            supports_stream  
            if supports_stream is not None  
            else hasattr(model, "astream")  
        )  
  
        if system_prompt is None:  
            self.prompt_template = PromptTemplate(template=DEFAULT_REACT_TEMPLATE)  
        elif isinstance(system_prompt, str):  
            self.prompt_template = PromptTemplate(template=system_prompt)  
        else:  
            self.prompt_template = system_prompt  
  
    # ===================== 对外 API =====================  
    async def step(  
        self,  
        input_messages: List[Message],  
        response_model: Optional[Type[T]] = None,  
        strategy: str = "auto",  
        max_retries: int = 2,  
    ) -> AgentOutput | T:  
        """  
        核心推理入口。  
        1. 构建上下文（包含历史和 system prompt）  
        2. 构造模型调用参数（function calling / json mode 等）  
        3. 运行 ReAct 循环  
        4. 如需结构化输出，执行解析与带反馈重试  
        """  
        logger.info(  
            "ReAct execution started",  
            input_count=len(input_messages),  
            has_structure=response_model is not None,  
        )  
  
        try:  
            context = await self._build_execution_context(input_messages)  
            llm_params = self._build_llm_params(response_model, strategy)  
  
            final_output = await self._run_reasoning_loop(  
                context,  
                llm_params,  
                response_model,  
                strategy,  
            )  
  
            if response_model:  
                structured = await self._extract_and_retry(  
                    final_output,  
                    response_model,  
                    context,  
                    llm_params,  
                    max_retries,  
                )  
                await self._save_context(context)  
                return structured  
  
            await self._save_context(context)  
            logger.info("ReAct execution completed")  
            return final_output  
  
        except Exception as e:  
            logger.exception("ReAct execution failed")  
            if isinstance(e, AgentError):  
                raise  
            raise AgentError(f"ReAct execution failed: {e}") from e  
  
    async def step_stream(self, input_messages: List[Message]) -> AsyncIterator[str]:  
        """  
        流式执行入口：  
        1. 先尝试快速判断是否需要工具，若需要则走常规 ReAct  
        2. 若不需要，直接流式输出最终回复（仅在模型支持 stream 时）  
        """  
        if not self.supports_stream:  
            raise AgentError("当前模型不支持流式输出")  
  
        context = await self._build_execution_context(input_messages)  
        llm_params = self._build_llm_params(None, "auto")  
  
        turn = 0  
        while turn < self.max_turns:  
            turn += 1  
  
            needs_tools, peek_response = await self._check_needs_tools(context, llm_params)  
            if needs_tools:  
                await self._execute_one_turn(context, llm_params, None, "auto")  
                continue  
  
            if peek_response:  
                msg = self._parse_llm_response(peek_response)  
                context.add_message(msg)  
                for chunk in msg.content or []:  
                    yield chunk  
                break  
  
            async for chunk in self._stream_final_response(context, llm_params):  
                yield chunk  
            break  
  
        await self._save_context(context)  
  
    # ===================== 上下文构建 =====================  
    async def _build_execution_context(self, input_messages: List[Message]) -> ExecutionContext:  
        history = await self._load_history()  
  
        # 若历史中不存在 system prompt，则插入  
        if not any(m.role == "system" for m in history):  
            system_msg = self._create_system_message()  
            history.insert(0, system_msg)  
  
        # 用户输入中若包含 system 也应该去重/放到开头  
        user_system = [m for m in input_messages if m.role == "system"]  
        normal_inputs = [m for m in input_messages if m.role != "system"]  
        all_messages = history + user_system + normal_inputs  
  
        return ExecutionContext(all_messages)  
  
    async def _load_history(self) -> List[Message]:  
        if not self.memory.storage or not self.memory.session_id:  
            return []  
  
        try:  
            raw = await self.memory.storage.get(self.memory.session_id)  
            if not raw:  
                return []  
            history_raw = raw.get("messages", [])  
            return await self.memory.get_history(history_raw)  
        except Exception as e:  
            logger.warning("Failed to load history", session_id=self.memory.session_id, error=str(e))  
            return []  
  
    def _create_system_message(self) -> Message:  
        tools_schema = self.toolbox.to_openai_schema()  
        content = self.prompt_template.format(tools=tools_schema)  
        return Message.system(content)  
  
    # ===================== LLM 参数构建 =====================  
    def _build_llm_params(self, response_model: Optional[Type[T]], strategy: str) -> Dict[str, Any]:  
        params: Dict[str, Any] = {}  
  
        tools_schema = self.toolbox.to_openai_schema()  
        if tools_schema and self.supports_functions:  
            params["tools"] = tools_schema  
            params["tool_choice"] = "auto"  
  
        if response_model:  
            if strategy in {"auto", "function_calling"} and self.supports_functions:  
                self._add_structure_params(params, response_model)  
            else:  
                params["response_format"] = {"type": "json_object"}  
                if not self.supports_functions:  
                    logger.warning("模型不支持 function calling，已降级为 JSON Mode")  
  
        return params  
  
    def _add_structure_params(self, params: Dict[str, Any], response_model: Type[T]):  
        structure_tool = StructureEngine.to_openai_tool(response_model)  
        params.setdefault("tools", []).append(structure_tool)  
        params["tool_choice"] = {  
            "type": "function",  
            "function": {"name": structure_tool["function"]["name"]},  
        }  
  
    # ===================== 主推理循环 =====================  
    async def _run_reasoning_loop(  
        self,  
        context: ExecutionContext,  
        llm_params: Dict[str, Any],  
        response_model: Optional[Type[T]],  
        strategy: str,  
    ) -> AgentOutput:  
        while context.turn < self.max_turns:  
            context.turn += 1  
  
            if self.on_turn_start:  
                await self._safe_call_hook(self.on_turn_start, context)  
  
            should_continue = await self._execute_one_turn(  
                context,  
                llm_params,  
                response_model,  
                strategy,  
            )  
  
            if self.on_turn_end:  
                await self._safe_call_hook(self.on_turn_end, context)  
  
            if not should_continue:  
                break  
  
        last_msg = context.get_last_message()  
        if last_msg and last_msg.role == "assistant":  
            tool_calls = getattr(last_msg, "tool_calls", None) or []  
            return AgentOutput(  
                content=last_msg.content or "",  
                raw=context.metadata.get("last_response"),  
                tool_calls=tool_calls,  
            )  
  
        logger.warning("Max turns reached", max_turns=self.max_turns)  
        return AgentOutput(content="Max iterations reached.")  
  
    async def _execute_one_turn(  
        self,  
        context: ExecutionContext,  
        llm_params: Dict[str, Any],  
        response_model: Optional[Type[T]],  
        strategy: str,  
    ) -> bool:  
        response = await self._call_llm(context, llm_params)  
        assistant_msg = self._parse_llm_response(response)  
        context.add_message(assistant_msg)  
        context.metadata["last_response"] = response  
  
        # 结构化提取完成则无需继续  
        if (  
            response_model  
            and strategy in {"auto", "function_calling"}  
            and self._is_structure_extraction(assistant_msg, response_model)  
        ):  
            return False  
  
        if assistant_msg.tool_calls:  
            executed = await self._execute_tools(assistant_msg.tool_calls, context, response_model)  
            return executed  
  
        # 无工具调用，则视为最终回复  
        return False  
  
    # ===================== LLM 调用/解析 =====================  
    async def _call_llm(self, context: ExecutionContext, params: Dict[str, Any]) -> Any:  
        messages_payload = [m.to_openai_format() for m in context.messages]  
        logger.debug("Calling LLM", message_count=len(messages_payload))  
  
        try:  
            response = await ensure_awaitable(self.model.acompletion, messages=messages_payload, **params)  
            return response  
        except Exception as e:  
            logger.error("LLM call failed", error=str(e))  
            raise ModelError(f"LLM API call failed: {e}") from e  
  
    def _parse_llm_response(self, response: Any) -> Message:  
        choice = response.choices[0]  
        raw_msg = choice.message  
  
        if hasattr(raw_msg, "model_dump"):  
            try:  
                msg_data = raw_msg.model_dump()  
            except Exception:  
                msg_data = {}  
        else:  
            msg_data = {}  
  
        if not msg_data:  
            msg_data = {  
                "content": getattr(raw_msg, "content", ""),  
                "tool_calls": getattr(raw_msg, "tool_calls", None),  
            }  
  
        if not msg_data.get("role"):  
            logger.warning("Missing role in LLM response")  
            msg_data["role"] = "assistant"  
  
        return Message(**msg_data)  
  
    # ===================== 工具调用 =====================  
    async def _execute_tools(  
        self,  
        tool_calls: List[Dict[str, Any]],  
        context: ExecutionContext,  
        response_model: Optional[Type[T]],  
    ) -> bool:  
        extraction_name = None  
        if response_model and self.supports_functions:  
            extraction_name = StructureEngine.to_openai_tool(response_model)["function"]["name"]  
  
        executed_successfully = False  
  
        for tool_call in tool_calls:  
            func_info = tool_call.get("function") or {}  
            func_name = func_info.get("name")  
            arguments = func_info.get("arguments")  
  
            if not func_name:  
                logger.error("Tool call missing name", tool_call=tool_call)  
                continue  
  
            if func_name == extraction_name:  
                # 结构化提取工具由结构化流程处理，此处跳过  
                continue  
  
            if not isinstance(arguments, (str, dict)):  
                logger.warning("Tool call arguments invalid", tool=func_name)  
                continue  
  
            result = await self._execute_single_tool(func_name, arguments, tool_call.get("id") or "")  
  
            tool_msg = Message.tool_result(  
                tool_call_id=tool_call.get("id") or "",  
                content=result["content"],  
                tool_name=func_name,  
            )  
            context.add_message(tool_msg)  
  
            if result["is_error"]:  
                # 反馈给 LLM，促使其重新规划  
                context.add_message(Message.user(f"工具 {func_name} 执行失败：{result['content']}"))  
            else:  
                executed_successfully = True  
  
        return executed_successfully  
  
    async def _execute_single_tool(self, func_name: str, args_payload: str | dict, call_id: str) -> Dict[str, str]:  
        try:  
            args = json.loads(args_payload) if isinstance(args_payload, str) else args_payload  
        except json.JSONDecodeError as e:  
            logger.error("Tool arguments JSON decode failed", tool=func_name, error=str(e))  
            return {"content": f"参数解析失败：{e}", "is_error": True}  
  
        try:  
            if self.on_tool_execute:  
                await self._safe_call_hook(self.on_tool_execute, func_name, args)  
  
            result = await self.toolbox.execute(func_name, args)  
            return {"content": result if isinstance(result, str) else str(result), "is_error": False}  
        except Exception as e:  
            logger.exception("Tool execution failed", tool=func_name)  
            return {"content": f"执行异常：{e}", "is_error": True}  
  
    # ===================== 结构化输出 =====================  
    def _is_structure_extraction(self, message: Message, model_class: Type[T]) -> bool:  
        if not message.tool_calls or not self.supports_functions:  
            return False  
        extraction_name = StructureEngine.to_openai_tool(model_class)["function"]["name"]  
        return any(tc.get("function", {}).get("name") == extraction_name for tc in message.tool_calls)  
  
    async def _extract_and_retry(  
        self,  
        base_output: AgentOutput,  
        model_class: Type[T],  
        context: ExecutionContext,  
        llm_params: Dict[str, Any],  
        max_retries: int,  
    ) -> T:  
        for retry in range(max_retries + 1):  
            try:  
                tool_calls = self._extract_tool_calls(base_output)  
                result = await StructureEngine.parse(base_output.content, model_class, tool_calls)  
                logger.info("Structured output extracted", retries=retry)  
                return result  
            except ValueError as e:  
                if retry >= max_retries:  
                    logger.error("Structure extraction failed", retries=retry)  
                    raise AgentError(str(e))  
                logger.info("Retrying structure extraction", attempt=retry + 1)  
                base_output = await self._retry_with_feedback(str(e), context, llm_params)  
  
        raise AgentError("Structure extraction failed unexpectedly")  
  
    def _extract_tool_calls(self, output: AgentOutput) -> Optional[List[Dict[str, Any]]]:  
        if not output.raw:  
            return None  
        try:  
            return getattr(output.raw.choices[0].message, "tool_calls", None)  
        except (AttributeError, IndexError):  
            return None  
  
    async def _retry_with_feedback(  
        self,  
        error_message: str,  
        context: ExecutionContext,  
        llm_params: Dict[str, Any],  
    ) -> AgentOutput:  
        feedback = Message.user(f"Parsing error: {error_message}. Please format as valid JSON.")  
        context.add_message(feedback)  
  
        response = await self._call_llm(context, llm_params)  
        new_msg = self._parse_llm_response(response)  
        context.add_message(new_msg)  
  
        return AgentOutput(  
            content=new_msg.content or "",  
            raw=response,  
            tool_calls=getattr(new_msg, "tool_calls", None),  
        )  
  
    # ===================== 流式/peek =====================  
    async def _check_needs_tools(  
        self,  
        context: ExecutionContext,  
        llm_params: Dict[str, Any],  
    ) -> Tuple[bool, Optional[Any]]:  
        """  
        通过一次短输出的调用来判断是否需要工具。  
        返回 (需要工具, peek_response)，若模型已给出最终回复，则可复用 peek_response。  
        """  
        peek_params = dict(llm_params)  
        peek_params["max_tokens"] = min(50, peek_params.get("max_tokens", 50))  
  
        try:  
            response = await self._call_llm(context, peek_params)  
            msg = self._parse_llm_response(response)  
  
            if msg.tool_calls:  
                # 已经给出工具调用，则把该消息加入上下文  
                context.add_message(msg)  
                return True, None  
  
            # 无工具调用，直接返回 peek response 供上层复用  
            return False, response  
  
        except Exception as e:  
            logger.warning("Peek call failed, fallback to normal flow", error=str(e))  
            return False, None  
  
    async def _stream_final_response(  
        self,  
        context: ExecutionContext,  
        llm_params: Dict[str, Any],  
    ) -> AsyncIterator[str]:  
        messages_payload = [m.to_openai_format() for m in context.messages]  
        accumulated: List[str] = []  
  
        try:  
            async for chunk in self.model.astream(messages=messages_payload, **llm_params):  
                delta = chunk.choices[0].delta  
                if delta.content:  
                    accumulated.append(delta.content)  
                    yield delta.content  
        except Exception as e:  
            logger.error("Streaming failed", error=str(e))  
            raise  
  
        full_content = "".join(accumulated)  
        final_msg = Message.assistant(full_content)  
        context.add_message(final_msg)  
        context.metadata["last_response"] = accumulated  
  
    # ===================== 记忆保存 =====================  
    async def _save_context(self, context: ExecutionContext):  
        if not self.memory.storage or not self.memory.session_id:  
            return  
        try:  
            messages_data = [m.model_dump() for m in context.messages]  
            await self.memory.storage.set(self.memory.session_id, {"messages": messages_data})  
            logger.debug("Context saved", message_count=len(messages_data))  
        except Exception as e:  
            logger.error("Failed to save context", error=str(e))  
  
    # ===================== Hook/工具方法 =====================  
    async def _safe_call_hook(self, hook: Callable, *args, **kwargs):  
        try:  
            if asyncio.iscoroutinefunction(hook):  
                await hook(*args, **kwargs)  
            else:  
                hook(*args, **kwargs)  
        except Exception as e:  
            logger.warning("Hook execution failed", hook=getattr(hook, "__name__", str(hook)), error=str(e))  
