# gecko/core/engine/react.py

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import random
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set, Type, TypeVar, Union, cast

from pydantic import BaseModel

from gecko.config import get_settings
from gecko.core.engine.base import CognitiveEngine, get_pricing_for_model
from gecko.core.engine.buffer import StreamBuffer
from gecko.core.events.types import AgentStreamEvent
from gecko.core.exceptions import AgentError
from gecko.core.logging import get_logger
from gecko.core.memory import TokenMemory
from gecko.core.message import Message
from gecko.core.output import AgentOutput
from gecko.core.prompt import PromptTemplate
from gecko.core.structure import StructureEngine
from gecko.core.toolbox import ToolBox, ToolExecutionResult
from gecko.core.utils import ensure_awaitable

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

STRUCTURE_TOOL_PREFIX = "__gecko_structured_output_"

DEFAULT_REACT_TEMPLATE = """You are a helpful AI assistant.
Current Time: {{ current_time }}

Available Tools:
{% for tool in tools %}
- {{ tool['function']['name'] }}: {{ tool['function']['description'] }}
{% endfor %}

Answer the user's request. Use tools if necessary.
If you use a tool, just output the tool call format.
"""

MAX_TOOL_INDEX_GAP = 500
MAX_RETRY_DELAY_SECONDS = 5.0
DEFAULT_MAX_HISTORY = 50
DEFAULT_MAX_CONTEXT_CHARS = 100_000
TOOL_HASH_DEQUE_SIZE = 5


class ReActConfig(BaseModel):
    max_reflections: int = 2
    tool_error_threshold: int = 3
    loop_repeat_threshold: int = 2
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS


class ExecutionContext:
    __slots__ = (
        "messages", "max_history", "max_chars", "turn", "metadata",
        "consecutive_errors", "reflection_attempts", "last_tool_hash",
        "last_tool_hashes", "message_metadata", "_msg_lengths_cache"
    )

    def __init__(
        self,
        messages: List[Message],
        max_history: int = DEFAULT_MAX_HISTORY,
        max_chars: Optional[int] = None,
    ):
        self.messages: List[Message] = list(messages)
        self.max_history: int = max_history
        self.max_chars: int = max_chars or DEFAULT_MAX_CONTEXT_CHARS
        self.turn: int = 0
        self.metadata: Dict[str, Any] = {}

        self.consecutive_errors: int = 0
        self.reflection_attempts: int = 0

        self.last_tool_hash: Optional[str] = None
        self.last_tool_hashes: deque = deque(maxlen=TOOL_HASH_DEQUE_SIZE)

        self.message_metadata: Dict[str, Dict[str, Any]] = {}
        self._msg_lengths_cache: Dict[int, int] = {}

    def add_message(self, message: Message) -> None:
        self.messages.append(message)
        self._msg_lengths_cache[id(message)] = self._get_message_length(message)

        if len(self.messages) > self.max_history:
            self._trim_context()
            return

        current_chars = self._calculate_total_chars()

        if current_chars > self.max_chars:
            logger.warning(
                "Context size exceeded limit, trimming",
                current_chars=current_chars,
                limit=self.max_chars,
            )
            self._trim_context(target_chars=self.max_chars)

    def add_message_with_metadata(
        self,
        message: Message,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        msg_id = str(uuid.uuid4())
        self.messages.append(message)
        self._msg_lengths_cache[id(message)] = self._get_message_length(message)

        if metadata:
            self.message_metadata[msg_id] = metadata

        try:
            object.__setattr__(message, "_gecko_msg_id", msg_id)
        except Exception:
            pass

        if len(self.messages) > self.max_history:
            self._trim_context()

        return msg_id

    def _calculate_total_chars(self) -> int:
        total = 0
        for m in self.messages:
            msg_id = id(m)
            if msg_id in self._msg_lengths_cache:
                total += self._msg_lengths_cache[msg_id]
            else:
                length = self._get_message_length(m)
                self._msg_lengths_cache[msg_id] = length
                total += length
        return total

    def _trim_context(self, target_chars: Optional[int] = None) -> None:
        system_msgs = [m for m in self.messages if m.role == "system"]
        conversation_msgs = [m for m in self.messages if m.role != "system"]

        if not conversation_msgs:
            self.messages = system_msgs
            return

        if target_chars is None:
            self._trim_by_count(system_msgs, conversation_msgs)
        else:
            self._trim_by_size(system_msgs, conversation_msgs, target_chars)

    def _trim_by_count(self, system_msgs: List[Message], conversation_msgs: List[Message]) -> None:
        keep_count = max(1, self.max_history - len(system_msgs))
        
        remove_count = len(conversation_msgs) - keep_count
        if remove_count <= 0:
            self.messages = system_msgs + conversation_msgs
            return

        remove_indices: Set[int] = set()
        i = 0
        
        while len(remove_indices) < remove_count and i < len(conversation_msgs):
            if i in remove_indices:
                i += 1
                continue
                
            msg = conversation_msgs[i]
            
            if self._is_assistant_with_tools(msg):
                tool_ids = self._extract_tool_ids(msg)
                chain_indices = self._find_tool_chain(conversation_msgs, i, tool_ids)
                for idx in chain_indices:
                    remove_indices.add(idx)
            else:
                remove_indices.add(i)
            
            i += 1

        remaining = [m for idx, m in enumerate(conversation_msgs) if idx not in remove_indices]
        self.messages = system_msgs + remaining

    def _trim_by_size(
        self, 
        system_msgs: List[Message], 
        conversation_msgs: List[Message], 
        target_chars: int
    ) -> None:
        msg_lengths = []
        for m in conversation_msgs:
            msg_id = id(m)
            if msg_id in self._msg_lengths_cache:
                msg_lengths.append(self._msg_lengths_cache[msg_id])
            else:
                length = self._get_message_length(m)
                self._msg_lengths_cache[msg_id] = length
                msg_lengths.append(length)

        system_len = sum(self._get_message_length(m) for m in system_msgs)
        total_len = system_len + sum(msg_lengths)

        remove_indices: Set[int] = set()
        i = 0

        while total_len > target_chars and i < len(conversation_msgs):
            if i in remove_indices:
                i += 1
                continue

            msg = conversation_msgs[i]

            if self._is_assistant_with_tools(msg):
                tool_ids = self._extract_tool_ids(msg)
                chain_indices = self._find_tool_chain(conversation_msgs, i, tool_ids)

                chain_len = sum(msg_lengths[idx] for idx in chain_indices if idx < len(msg_lengths))
                for idx in chain_indices:
                    remove_indices.add(idx)
                total_len -= chain_len
            else:
                remove_indices.add(i)
                total_len -= msg_lengths[i]

            i += 1

        remaining = [m for idx, m in enumerate(conversation_msgs) if idx not in remove_indices]
        self.messages = system_msgs + remaining

        logger.debug(
            "Context trimmed",
            remaining_messages=len(self.messages),
            removed_count=len(remove_indices),
        )

    def _is_assistant_with_tools(self, msg: Message) -> bool:
        return getattr(msg, "role", None) == "assistant" and bool(getattr(msg, "tool_calls", None))

    def _extract_tool_ids(self, msg: Message) -> Set[str]:
        tool_calls = getattr(msg, "tool_calls", None) or []
        return {
            tc.get("id", "")
            for tc in tool_calls
            if isinstance(tc, dict) and tc.get("id")
        }

    def _find_tool_chain(
        self, 
        msgs: List[Message], 
        start_index: int, 
        tool_ids: Set[str]
    ) -> List[int]:
        indices = [start_index]
        for j in range(start_index + 1, len(msgs)):
            check_msg = msgs[j]
            if getattr(check_msg, "role", None) == "tool":
                tool_call_id = getattr(check_msg, "tool_call_id", None)
                if tool_call_id in tool_ids:
                    indices.append(j)
                else:
                    break
            else:
                break
        return indices

    def _get_message_length(self, msg: Message) -> int:
        try:
            return len(msg.get_text_content())
        except Exception:
            return 0

    @property
    def last_message(self) -> Message:
        if not self.messages:
            raise ValueError("Context is empty, cannot get last message")
        return self.messages[-1]


class ReActEngine(CognitiveEngine):
    def __init__(
        self,
        model: Any,
        toolbox: ToolBox,
        memory: TokenMemory,
        max_turns: int = 10,
        max_observation_length: int = 2000,
        system_prompt: Union[str, PromptTemplate, None] = None,
        on_turn_start: Optional[Callable[[ExecutionContext], Any]] = None,
        on_turn_end: Optional[Callable[[ExecutionContext], Any]] = None,
        config: Optional[ReActConfig] = None,
        **kwargs: Any,
    ):
        super().__init__(model, toolbox, memory, **kwargs)

        self.max_turns = int(max_turns)
        self.max_observation_length = int(max_observation_length)
        self.on_turn_start = on_turn_start
        self.on_turn_end = on_turn_end
        self.config = config or ReActConfig()

        if system_prompt is None:
            self.prompt_template = PromptTemplate(template=DEFAULT_REACT_TEMPLATE)
        elif isinstance(system_prompt, str):
            self.prompt_template = PromptTemplate(template=system_prompt)
        else:
            self.prompt_template = system_prompt

        self._supports_functions: bool = bool(getattr(self.model, "_supports_function_calling", True))

        logger.debug(
            "ReActEngine initialized",
            max_turns=self.max_turns,
            supports_functions=self._supports_functions,
        )

    def _safe_deep_copy_messages(self, messages: List[Message]) -> List[Message]:
        result: List[Message] = []
        for m in messages:
            try:
                if hasattr(m, "model_copy"):
                    result.append(m.model_copy(deep=True))
                else:
                    result.append(copy.deepcopy(m))
            except Exception as e:
                logger.warning(
                    "Message copy failed, using original reference",
                    error=str(e),
                    message_type=type(m).__name__,
                )
                result.append(m)
        return result

    async def step(
        self,
        input_messages: List[Message],
        response_model: Optional[Type[T]] = None,
        max_retries: int = 0,
        **kwargs: Any,
    ) -> Union[AgentOutput, T]:
        if response_model is not None:
            from inspect import isclass

            if not (isclass(response_model) and issubclass(response_model, BaseModel)):
                raise TypeError(
                    f"response_model must be a subclass of Pydantic BaseModel, got: {type(response_model).__name__}"
                )

            output = await self._execute_step(input_messages, response_model=response_model, **kwargs)

            current_messages = list(input_messages)
            attempts = 0

            while True:
                try:
                    if output.tool_calls:
                        return await StructureEngine.parse(
                            content="",
                            model_class=response_model,
                            raw_tool_calls=output.tool_calls,
                        )

                    return await StructureEngine.parse(output.content, response_model)

                except Exception as e:
                    if attempts >= max_retries:
                        raise AgentError(f"Structured parsing failed: {e}") from e

                    attempts += 1
                    logger.warning(
                        "Structure parse failed, retrying",
                        attempt=attempts,
                        max_retries=max_retries,
                        error=str(e),
                    )

                    current_messages.append(
                        Message.assistant(content=output.content, tool_calls=output.tool_calls)
                    )
                    current_messages.append(
                        Message.user(
                            f"Error parsing response: {e}. Please try again using the correct format."
                        )
                    )

                    output = await self._execute_step(
                        current_messages, response_model=response_model, **kwargs
                    )

        return await self._execute_step(input_messages, **kwargs)

    async def step_structured(
        self,
        input_messages: List[Message],
        response_model: Type[T],
        max_retries: int = 0,
        **kwargs: Any,
    ) -> T:
        result = await self.step(
            input_messages, response_model=response_model, max_retries=max_retries, **kwargs
        )
        return cast(T, result)

    async def step_stream(
        self,
        input_messages: List[Message],
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[AgentStreamEvent]:
        if timeout is None:
            timeout = get_settings().default_model_timeout

        self.validate_input(input_messages)
        await self.before_step(input_messages, **kwargs)

        context = await self._build_execution_context(input_messages)
        start_time = time.time()

        final_output: Optional[AgentOutput] = None
        saw_error_event: bool = False

        try:
            async for event in self._execute_lifecycle_with_timeout(context, float(timeout), **kwargs):
                if event.type == "result" and event.data:
                    try:
                        final_output = cast(AgentOutput, event.data.get("output"))
                    except Exception:
                        final_output = None

                if event.type == "error":
                    saw_error_event = True

                yield event

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(
                "step_stream timeout",
                elapsed_seconds=elapsed,
                max_timeout=timeout,
                current_turn=context.turn,
            )

            await self._save_context(context, force=True)

            yield AgentStreamEvent(
                type="error",
                content=f"Execution timeout after {timeout}s at turn {context.turn}",
            )
            raise

        except asyncio.CancelledError:
            logger.warning("step_stream was cancelled", current_turn=context.turn)
            await self._save_context(context, force=True)
            raise

        except AgentError as e:
            logger.error("step_stream caught AgentError", error=str(e))
            yield AgentStreamEvent(type="error", content=str(e))
            await self.on_error(e, input_messages, **kwargs)
            raise

        except Exception as e:
            logger.exception("Lifecycle execution failed", error=str(e))
            yield AgentStreamEvent(type="error", content=str(e))
            await self.on_error(e, input_messages, **kwargs)
            raise

        finally:
            try:
                await self._save_context(context)
            except Exception as save_error:
                logger.warning("Final context save failed", error=str(save_error))

            if final_output is not None and not saw_error_event:
                try:
                    await self.after_step(
                        context.messages,
                        final_output,
                        original_input=input_messages,
                        **kwargs
                    )
                except Exception as hook_err:
                    logger.warning("after_step failed", error=str(hook_err), exc_info=True)

    async def _execute_lifecycle_with_timeout(
        self,
        context: ExecutionContext,
        timeout: float,
        **kwargs: Any,
    ) -> AsyncIterator[AgentStreamEvent]:
        async with asyncio.timeout(timeout):
            async for event in self._execute_lifecycle(context, **kwargs):
                yield event

    async def _execute_step(
        self,
        input_messages: List[Message],
        response_model: Optional[Type[T]] = None,
        **kwargs: Any,
    ) -> AgentOutput:
        current_messages = self._safe_deep_copy_messages(input_messages)

        stream_kwargs: Dict[str, Any] = dict(kwargs)
        stream_kwargs["response_model"] = response_model

        final_result: Optional[AgentOutput] = None

        try:
            async for event in self.step_stream(current_messages, **stream_kwargs):
                if event.type == "result" and event.data:
                    final_result = cast(AgentOutput, event.data.get("output"))

                elif event.type == "error":
                    error_msg = str(event.content)
                    logger.error("Received error event during execution", error=error_msg)
                    raise AgentError(error_msg)

        except AgentError:
            raise
        except Exception as e:
            logger.exception("Event stream processing failed", error=str(e))
            raise AgentError(f"Step execution failed: {e}") from e

        if final_result is None:
            raise AgentError("Infinite loop detected: no result generated")

        return final_result

    async def _execute_lifecycle(self, context: ExecutionContext, **kwargs: Any) -> AsyncIterator[AgentStreamEvent]:
        response_model = kwargs.get("response_model")
        structure_tool_name: Optional[str] = None

        if response_model and self._supports_functions:
            schema = StructureEngine.to_openai_tool(response_model)
            structure_tool_name = schema["function"]["name"]

        while context.turn < self.max_turns:
            context.turn += 1
            logger.debug("Starting turn", turn=context.turn)

            if self.on_turn_start:
                await ensure_awaitable(self.on_turn_start, context)

            async for event in self._think_phase(context, **kwargs):
                yield event

            assistant_msg = context.last_message

            if self._detect_loop(context, assistant_msg):
                error_msg = "Infinite loop detected"
                yield AgentStreamEvent(type="error", content=error_msg)
                return

            tool_calls = assistant_msg.safe_tool_calls

            if structure_tool_name and tool_calls:
                target_call = next(
                    (
                        tc for tc in tool_calls
                        if tc.get("function", {}).get("name") == structure_tool_name
                    ),
                    None,
                )
                if target_call:
                    final_output = AgentOutput(
                        content="",
                        tool_calls=[target_call],
                        metadata={"is_structured": True},
                    )
                    yield AgentStreamEvent(type="result", data={"output": final_output})
                    return

            if not tool_calls:
                final_output = AgentOutput(content=str(assistant_msg.content or ""), tool_calls=[])
                yield AgentStreamEvent(type="result", data={"output": final_output})
                return

            yield AgentStreamEvent(type="tool_input", data={"tools": tool_calls})

            for _ in tool_calls:
                self.record_tool_call()

            async for event in self._act_phase(context, tool_calls):
                yield event

            should_continue = await self._observe_phase(context)

            if self.on_turn_end:
                await ensure_awaitable(self.on_turn_end, context)

            await self._save_context(context)

            if not should_continue:
                stop_output = AgentOutput(content="Task stopped by system monitor.")
                yield AgentStreamEvent(type="result", data={"output": stop_output})
                return

        logger.warning("Reached max turns limit, treating as infinite loop", max_turns=self.max_turns)
        yield AgentStreamEvent(type="error", content="Infinite loop detected")
        return

    async def _think_phase(self, context: ExecutionContext, **kwargs: Any) -> AsyncIterator[AgentStreamEvent]:
        messages_payload = self._build_messages_payload(context)
        llm_params = self._build_llm_params(kwargs.get("response_model"), strategy="auto")
        
        safe_kwargs = {k: v for k, v in kwargs.items() if k not in ["response_model"]}
        llm_params.update(safe_kwargs)
        llm_params["stream"] = True

        input_tokens = 0
        output_tokens = 0
        model_name = self._get_model_name()

        buffer = StreamBuffer()
        stream_gen = self.model.astream(messages=messages_payload, **llm_params)

        try:
            async for chunk in stream_gen:
                if not hasattr(chunk, "delta"):
                    delta = getattr(chunk, "choices", [{}])
                    if not delta:
                        logger.warning("Received chunk without delta, skipping", chunk_type=type(chunk).__name__)
                        continue

                usage = getattr(chunk, "usage", None)
                if usage:
                    input_tokens = max(input_tokens, int(getattr(usage, "prompt_tokens", 0) or 0))
                    output_tokens = max(output_tokens, int(getattr(usage, "completion_tokens", 0) or 0))

                text_delta = buffer.add_chunk(chunk)
                if text_delta:
                    yield AgentStreamEvent(type="token", content=text_delta)
        finally:
            if input_tokens > 0 or output_tokens > 0:
                self.record_tokens(input_tokens=input_tokens, output_tokens=output_tokens)
                self.record_cost(input_tokens, output_tokens, model_name)

        assistant_msg = buffer.build_message()
        context.add_message(assistant_msg)

    async def _act_phase(
        self, 
        context: ExecutionContext, 
        tool_calls: List[Dict[str, Any]]
    ) -> AsyncIterator[AgentStreamEvent]:
        normalized_calls = [self._normalize_tool_call(tc) for tc in tool_calls]
        tool_results = await self.toolbox.execute_many(normalized_calls)

        for result in tool_results:
            truncated_content = self._truncate_observation(result.result, result.tool_name)

            context.add_message(
                Message.tool_result(
                    result.call_id,
                    truncated_content,
                    result.tool_name,
                )
            )

            yield AgentStreamEvent(
                type="tool_output",
                content=result.result,
                data={"tool_name": result.tool_name, "is_error": result.is_error},
            )

        context.metadata["last_tool_results"] = tool_results

    async def _observe_phase(self, context: ExecutionContext) -> bool:
        results = context.metadata.get("last_tool_results", [])
        error_count = sum(1 for r in results if r.is_error)

        if error_count > 0:
            context.consecutive_errors += 1
            logger.warning(
                "Tool execution errors",
                error_count=error_count,
                total_count=len(results),
                consecutive_errors=context.consecutive_errors,
            )
        else:
            context.consecutive_errors = 0

        if context.consecutive_errors >= self.config.tool_error_threshold:
            if context.reflection_attempts < self.config.max_reflections:
                self._inject_reflection_message(context, results)
                context.reflection_attempts += 1
                context.consecutive_errors = 0
                return True

            logger.error(
                "Tool error threshold exceeded, stopping",
                consecutive_errors=context.consecutive_errors,
                reflection_attempts=context.reflection_attempts,
                turn=context.turn,
            )
            return False

        return True

    def _inject_reflection_message(
        self, 
        context: ExecutionContext, 
        results: List[ToolExecutionResult]
    ) -> None:
        error_details = "\n".join(
            f"- {r.tool_name}: {r.result}" for r in results if r.is_error
        )

        system_feedback = (
            "System Notification: Multiple tool execution errors detected.\n"
            "Error Details:\n"
            f"{error_details}\n\n"
            "Please analyze these errors, adjust parameters or tool choice, and retry."
        )

        reflection_msg = Message.assistant(content=system_feedback, tool_calls=None)

        meta = {
            "type": "system_reflection",
            "error_summary": error_details,
            "reflection_attempt": context.reflection_attempts + 1,
        }

        self._attach_metadata_safe(reflection_msg, meta)

        store = context.metadata.setdefault("_gecko_msg_metadata", {})
        store[id(reflection_msg)] = meta

        context.add_message(reflection_msg)

    def _detect_loop(self, context: ExecutionContext, msg: Message) -> bool:
        if not msg.safe_tool_calls:
            return False

        try:
            calls_data = [
                {
                    "name": tc.get("function", {}).get("name"),
                    "args": tc.get("function", {}).get("arguments"),
                }
                for tc in msg.safe_tool_calls
            ]
            calls_dump = json.dumps(calls_data, sort_keys=True, ensure_ascii=False)
            current_hash = hashlib.sha256(calls_dump.encode("utf-8")).hexdigest()

            repeat_run = 1
            for h in reversed(context.last_tool_hashes):
                if h == current_hash:
                    repeat_run += 1
                else:
                    break

            if repeat_run >= self.config.loop_repeat_threshold:
                logger.warning(
                    "Consecutive tool call loop detected",
                    repeat_run=repeat_run,
                    threshold=self.config.loop_repeat_threshold,
                    tool_hash=current_hash[:16],
                )
                return True

            if len(context.last_tool_hashes) >= 2:
                if (current_hash == context.last_tool_hashes[-2] and
                    current_hash != context.last_tool_hashes[-1]):
                    logger.warning(
                        "Oscillation pattern detected (A-B-A alternating calls)",
                        tool_hash=current_hash[:16],
                    )
                    return True

            context.last_tool_hashes.append(current_hash)
            context.last_tool_hash = current_hash

            return False

        except Exception as e:
            logger.warning("Loop detection failed (fail-open)", error=str(e), exc_info=True)
            return False

    def _truncate_observation(self, content: str, tool_name: str) -> str:
        if len(content) > self.max_observation_length:
            logger.info(
                "Truncating output for tool",
                tool_name=tool_name,
                original_length=len(content),
                max_length=self.max_observation_length,
            )
            return content[: self.max_observation_length] + f"\n...(truncated, total {len(content)} chars)"
        return content

    def _normalize_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        func_block = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        name = func_block.get("name", "") if isinstance(func_block, dict) else ""
        raw_args = func_block.get("arguments", "{}") if isinstance(func_block, dict) else "{}"

        parsed_args: Dict[str, Any] = {}

        try:
            if isinstance(raw_args, str):
                parsed_args = json.loads(raw_args)
            elif isinstance(raw_args, dict):
                parsed_args = raw_args
            else:
                parsed_args = {}
        except json.JSONDecodeError as e:
            parsed_args = {
                "__gecko_parse_error__": (
                    f"JSON format error: {str(e)}. "
                    f"Content: {raw_args[:200] if isinstance(raw_args, str) and len(raw_args) > 200 else raw_args}"
                )
            }
            logger.warning("Tool arguments JSON parse failed", tool_name=name, error=str(e))

        return {"id": tool_call.get("id", ""), "name": name, "arguments": parsed_args}

    def _get_model_name(self) -> str:
        model_name = getattr(self.model, "model_name", None) or getattr(self.model, "model", None) or "gpt-3.5-turbo"
        if not isinstance(model_name, str):
            logger.warning("Model name type unexpected", model_name_type=type(model_name).__name__)
            model_name = "gpt-3.5-turbo"
        return model_name

    def _build_llm_params(self, response_model: Any, strategy: str = "auto") -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        tools_schema = self.toolbox.to_openai_schema()

        if response_model and self._supports_functions:
            structure_tool = StructureEngine.to_openai_tool(response_model)
            combined_tools = tools_schema + [structure_tool]
            params["tools"] = combined_tools
            params["tool_choice"] = {"type": "function", "function": {"name": structure_tool["function"]["name"]}}

        elif tools_schema and self._supports_functions:
            params["tools"] = tools_schema
            params["tool_choice"] = "auto"

        return params

    def _build_messages_payload(self, context: ExecutionContext) -> List[Dict[str, Any]]:
        meta_map = context.metadata.get("_gecko_msg_metadata", {}) or {}
        messages_payload: List[Dict[str, Any]] = []

        for m in context.messages:
            payload = m.to_openai_format()
            md = self._extract_message_metadata(m, meta_map, context)

            if md:
                payload["metadata"] = md

            messages_payload.append(payload)

        return messages_payload

    def _extract_message_metadata(
        self, 
        msg: Message, 
        meta_map: Dict[int, Dict[str, Any]], 
        context: ExecutionContext
    ) -> Optional[Dict[str, Any]]:
        md = None

        try:
            if hasattr(msg, "metadata"):
                v = getattr(msg, "metadata", None)
                if isinstance(v, dict) and v:
                    md = v
        except Exception:
            pass

        if md is None:
            try:
                v2 = meta_map.get(id(msg))
                if isinstance(v2, dict) and v2:
                    md = v2
            except Exception:
                pass

        if md is None:
            try:
                msg_id = getattr(msg, "_gecko_msg_id", None)
                if msg_id and msg_id in context.message_metadata:
                    md = context.message_metadata[msg_id]
            except Exception:
                pass

        return md

    async def _build_execution_context(self, input_messages: List[Message]) -> ExecutionContext:
        history = await self._load_history()
        all_messages = history + input_messages

        has_system_msg = any(m.role == "system" for m in all_messages)
        if not has_system_msg:
            system_content = self._render_system_prompt()
            all_messages.insert(0, Message.system(system_content))
        else:
            logger.debug("Using user-specified system message")

        max_context_chars = self.config.max_context_chars
        return ExecutionContext(all_messages, max_chars=max_context_chars)

    def _render_system_prompt(self) -> str:
        template_vars = {
            "tools": self.toolbox.to_openai_schema(),
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            return self.prompt_template.format_safe(**template_vars)
        except Exception as e:
            logger.warning("System prompt formatting failed", error=str(e))
            return "You are a helpful AI assistant."

    async def _load_history(self) -> List[Message]:
        if not getattr(self.memory, "storage", None):
            return []
        try:
            data = await self.memory.storage.get(self.memory.session_id)
            if data and "messages" in data:
                return await self.memory.get_history(data["messages"])
        except Exception as e:
            logger.warning("Failed to load history", error=str(e))
        return []

    async def _save_context(self, context: ExecutionContext, force: bool = False, max_retries: int = 3) -> None:
        if not getattr(self.memory, "storage", None):
            return

        messages_data = [m.to_openai_format() for m in context.messages]

        retries = max_retries if force else 1
        for attempt in range(retries):
            try:
                await self.memory.storage.set(self.memory.session_id, {"messages": messages_data})
                return
            except Exception as e:
                if not force or attempt >= retries - 1:
                    logger.warning("Failed to save context", error=str(e), attempt=attempt + 1, force=force)
                    if force:
                        raise
                    return

                base_delay = min(0.1 * (2 ** attempt), MAX_RETRY_DELAY_SECONDS)
                jitter = random.uniform(0, base_delay * 0.5)
                await asyncio.sleep(base_delay + jitter)

    def _attach_metadata_safe(self, msg: Message, metadata: Dict[str, Any]) -> None:
        if not metadata:
            return

        try:
            if hasattr(msg, "metadata"):
                current = getattr(msg, "metadata", None)
                if isinstance(current, dict):
                    current.update(metadata)
                    return
                else:
                    try:
                        setattr(msg, "metadata", dict(metadata))
                        return
                    except Exception:
                        pass
        except Exception:
            pass

        try:
            object.__setattr__(msg, "metadata", dict(metadata))
        except Exception:
            logger.debug("Failed to attach metadata to message (ignored)")


__all__ = [
    "ReActEngine",
    "ExecutionContext",
    "ReActConfig",
    "DEFAULT_REACT_TEMPLATE",
    "STRUCTURE_TOOL_PREFIX",
    "MAX_RETRY_DELAY_SECONDS",
]