# gecko/core/engine/react.py
"""
ReAct 推理 引擎 (Production Grade / 生产级)

架构设计：
1. 无状态设计 (Stateless): 
   Engine 实例不持有任何单次请求的状态，所有状态封装在 ExecutionContext 中。
   这使得 Engine 单例可以在多线程/异步环境下安全地处理并发请求。

2. 生命周期分解 (Lifecycle Decomposition):
   将复杂的 ReAct while 循环拆解为 _phase_think (思考), _phase_act (行动), 
   _phase_observe (观察) 三个独立阶段。子类（如 ReflexionEngine）可以重写特定阶段
   而不必复制整个循环逻辑。

3. 事件驱动 (Event Driven):
   统一使用 AgentStreamEvent 协议，消除了 yield 返回类型不明确的问题。
   同步接口 step 仅仅是流式接口 step_stream 的消费者。

4. 鲁棒性 (Robustness):
   - 集成 StreamBuffer 处理流式碎片及修复不规范 JSON。
   - 内置死循环熔断机制。
   - 内置观察值截断机制，防止 Context Window 爆炸。
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
import time
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
    cast,
)

from pydantic import BaseModel

from gecko.core.engine.base import CognitiveEngine
from gecko.core.engine.buffer import StreamBuffer
from gecko.core.events.types import AgentStreamEvent
from gecko.core.protocols import StreamChunk 
from gecko.core.toolbox import ToolExecutionResult 
from gecko.core.exceptions import AgentError
from gecko.core.logging import get_logger
from gecko.core.memory import TokenMemory
from gecko.core.message import Message
from gecko.core.output import AgentOutput
from gecko.core.prompt import PromptTemplate
from gecko.core.structure import StructureEngine
from gecko.core.toolbox import ToolBox
from gecko.core.utils import ensure_awaitable
from gecko.config import get_settings

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

# 默认的 ReAct 提示词模板
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
    执行上下文 (Runtime Context)
    
    承载单次 Agent.run/stream 的所有运行时状态。
    随请求创建，随请求销毁。
    """
    def __init__(self, messages: List[Message], max_history: int = 50):
        # 浅拷贝消息列表，防止污染传入的原始列表，但在处理过程中会追加新消息
        self.messages = messages.copy()
        self.max_history = max_history  # [P2-2 Fix] 消息历史上限
        self.turn = 0
        self.metadata: Dict[str, Any] = {}
        
        # --- 状态追踪：用于死循环检测与错误熔断 ---
        self.consecutive_errors: int = 0  # 连续工具错误次数
        self.last_tool_hash: Optional[int] = None # 上一次工具调用的指纹
        self.last_tool_hashes: List[int] = []  # [P2-3 Fix] 最近 3 轮的工具调用哈希

    def add_message(self, message: Message) -> None:
        self.messages.append(message)
        
        # 1. 基础条数限制
        if len(self.messages) > self.max_history:
            self._trim_context()
            return

        # 2. [新增] 字符级滑动窗口保护 (防止 Context Window 溢出)
        # 估算：1 token ≈ 4 chars。
        # 假设最大安全窗口为 100k chars (约 25k tokens)，留给 prompt template 和 reasoning
        # 这个阈值可以根据实际模型调整，或者作为参数传入
        SAFE_CHAR_LIMIT = 100000 
        
        current_chars = sum(len(m.get_text_content()) for m in self.messages)
        
        if current_chars > SAFE_CHAR_LIMIT:
            logger.warning(
                f"Context size ({current_chars} chars) exceeded limit during execution loop. Trimming history."
            )
            self._trim_context(target_chars=SAFE_CHAR_LIMIT)

    def _trim_context(self, target_chars: Optional[int] = None) -> None:
        """
        [新增] 智能裁剪逻辑
        策略：保留 System Message (index 0)，从最早的对话历史开始移除。
        """
        # 必须保留 system 消息
        system_msgs = [m for m in self.messages if m.role == "system"]
        # 对话消息
        conversation_msgs = [m for m in self.messages if m.role != "system"]
        
        # 如果没有对话消息可删，直接返回（避免删掉 system）
        if not conversation_msgs:
            return

        # 模式 A: 按条数裁剪
        if target_chars is None:
            # 计算保留数量
            keep_count = max(1, self.max_history - len(system_msgs))
            conversation_msgs = conversation_msgs[-keep_count:]
        
        # 模式 B: 按字符数裁剪
        else:
            current_len = sum(len(m.get_text_content()) for m in (system_msgs + conversation_msgs))
            while current_len > target_chars and len(conversation_msgs) > 1:
                # 移除最老的一条对话
                removed = conversation_msgs.pop(0)
                current_len -= len(removed.get_text_content())

        # 重组
        self.messages = system_msgs + conversation_msgs

    @property
    def last_message(self) -> Message:
        """获取历史中最后一条消息，用于检查 LLM 的最新输出"""
        if not self.messages:
            raise ValueError("Context is empty")
        return self.messages[-1]


class ReActEngine(CognitiveEngine):
    """
    生产级 ReAct 引擎实现。
    """

    def __init__(
        self,
        model: Any,
        toolbox: ToolBox,
        memory: TokenMemory,
        max_turns: int = 10,
        max_observation_length: int = 2000,
        system_prompt: Union[str, PromptTemplate, None] = None,
        # 生命周期钩子 Hooks
        on_turn_start: Optional[Callable[[ExecutionContext], Any]] = None,
        on_turn_end: Optional[Callable[[ExecutionContext], Any]] = None,
        **kwargs: Any,
    ):
        super().__init__(model, toolbox, memory, **kwargs)
        self.max_turns = max_turns
        self.max_observation_length = max_observation_length
        self.on_turn_start = on_turn_start
        self.on_turn_end = on_turn_end

        # 初始化 System Prompt 模板
        if system_prompt is None:
            self.prompt_template = PromptTemplate(template=DEFAULT_REACT_TEMPLATE)
        elif isinstance(system_prompt, str):
            self.prompt_template = PromptTemplate(template=system_prompt)
        else:
            self.prompt_template = system_prompt

        # 检测模型能力：是否支持原生 Function Calling
        self._supports_functions = getattr(self.model, "_supports_function_calling", True)

    # ================= 核心入口 (Public API) =================

    async def step( # type: ignore
        self,
        input_messages: List[Message],
        response_model: Optional[Type[T]] = None,
        max_retries: int = 0,
        **kwargs: Any,
    ) -> Union[AgentOutput, T]:
        """
        同步执行入口。
        
        逻辑流程：
        1. 作为一个消费者，迭代 `step_stream` 产生的事件流。
        2. 忽略中间的 Token 事件，只捕获 `result` 或处理 `error`。
        3. 如果指定了 `response_model`，则对结果进行结构化解析。
        4. 如果解析失败且配置了重试，则将错误反馈给模型并重新运行推理。
        
        参数:
            input_messages: 用户输入消息列表
            response_model: (可选) Pydantic 模型，用于强制结构化输出
            max_retries: 结构化解析失败时的最大重试次数
            **kwargs: 传递给 LLM 的额外参数
        """
        
        # 将 response_model 放入 kwargs 传递给 step_stream，以便底层构建 Tool Schema
        kwargs['response_model'] = response_model

        # [新增] 验证 response_model
        if response_model is not None:
            from inspect import isclass
            if not (isclass(response_model) and issubclass(response_model, BaseModel)):
                raise TypeError(
                    f"response_model must be a Pydantic BaseModel subclass, "
                    f"got {type(response_model).__name__}"
                )

        # 内部闭包：执行一次完整的推理流程，直到产生结果或报错
        async def _run_once(msgs: List[Message]) -> AgentOutput:
            """
            执行一次完整的 ReAct 生命周期，并从事件流中提取最终 AgentOutput。

            设计要点：
            - 只要看到 `result` 事件，就更新 final_res。
            - 遇到 `error` 事件时，根据错误内容做分类：
              * 已经是 "Infinite loop detected" -> 原样抛出；
              * 包含 "StopIteration" -> 视为模型流式异常导致的“死循环”场景，升级为 Infinite loop；
              * 其它错误 -> 原样抛出 AgentError。
            - 如果整个生命周期跑完 **没有任何 result 事件**（final_res 仍为 None），
              则视为 ReAct 未能收敛（例如 max_turns 用尽、逻辑走偏等），
              统一抛出 "Infinite loop detected: no result generated"。
            """
            final_res: Optional[AgentOutput] = None

            try:
                async for event in self.step_stream(msgs, **kwargs):
                    if event.type == "result" and event.data:
                        # 从事件载荷中提取 AgentOutput 对象
                        final_res = cast(AgentOutput, event.data.get("output"))
                    elif event.type == "error":
                        # 通过事件通道传上来的错误，按 AgentError 处理
                        msg = str(event.content)
                        logger.error("Engine step error event: %s", msg)

                        # 1) 已经是死循环检测
                        if "Infinite loop detected" in msg:
                            raise AgentError(msg)

                        # 2) 模型流式异常（StopIteration），视为一种“无法收敛”的死循环
                        if "StopIteration" in msg:
                            raise AgentError(
                                "Infinite loop detected: model streaming stopped unexpectedly"
                            )

                        # 3) 其他错误，原样抛出
                        raise AgentError(msg)

            except AgentError:
                # 保持 AgentError 原样向上抛出，避免包裹改变 message
                raise
            except Exception as e:
                # 其他异常统一兜底为 Step execution failed
                logger.exception("Stream processing failed in _run_once", error=str(e))
                raise

            # 整个 step_stream 跑完，没有任何 result 事件 -> 视为死循环/未收敛
            if final_res is None:
                err_msg = "Infinite loop detected: no result generated"
                logger.error(err_msg)
                raise AgentError(err_msg)

            return final_res


        # 1. 首次运行
        # [P1-1 Fix] 深拷贝消息列表，防止外部修改污染上下文
        try:
            current_messages = [
                Message(**m.dict()) if hasattr(m, 'dict') else m 
                for m in input_messages
            ]
        except Exception:
            # Fallback: 如果深拷贝失败，使用浅拷贝
            current_messages = list(input_messages)
        
        try:
            final_output = await _run_once(current_messages)
            
            if not final_output:
                # [P1-1 Fix] 补充计费信息，即使无输出也应记录 token 使用
                return AgentOutput(
                    content="[System Error] No output generated.",
                    usage={"input_tokens": 0, "output_tokens": 0} # type: ignore
                )
        except AgentError:
            # AgentError 直接重新抛出
            raise
        except Exception as e:
            logger.exception("step() execution failed", error=str(e))
            raise AgentError(f"Step execution failed: {e}") from e
            
        # 2. 结构化解析 + 自动重试循环
        # 如果不需要结构化输出，直接返回 AgentOutput
        if response_model:
            attempts = 0
            while True:
                try:
                    # 优先策略 A: 尝试从 Tool Calls 中解析 (OpenAI 模式)
                    # 如果 LLM 正确调用了我们注入的结构化工具，数据会在 tool_calls 中
                    if final_output.tool_calls:
                        return await StructureEngine.parse(
                            content="", 
                            model_class=response_model, 
                            raw_tool_calls=final_output.tool_calls
                        )
                    # 优先策略 B: 回退到 Content 解析 (JSON Mode / Text 提取)
                    return await StructureEngine.parse(final_output.content, response_model)
                
                except Exception as e:
                    # 解析失败，检查是否还有重试机会
                    if attempts >= max_retries:
                        raise AgentError(f"Structured parsing failed: {e}") from e
                    
                    attempts += 1
                    logger.warning(f"Structure parse failed, retrying ({attempts}/{max_retries})")
                    
                    # 构造反馈消息并追加到本次会话历史中
                    # 让模型看到自己上次的输出和对应的错误信息
                    current_messages.append(Message.assistant(
                        content=final_output.content,
                        tool_calls=final_output.tool_calls
                    ))
                    current_messages.append(Message.user(
                        f"Error parsing response: {e}. Please try again using the correct format."
                    ))
                    
                    # 再次运行 Engine (Retry)
                    final_output = await _run_once(current_messages)
                    if not final_output:
                         raise AgentError("Retry returned no output")

        return final_output


    async def step_stream( # type: ignore
        self, 
        input_messages: List[Message], 
        timeout: Optional[float] = None,
        **kwargs: Any
    ) -> AsyncIterator[AgentStreamEvent]:
        """
        流式执行入口。
        
        生成 `AgentStreamEvent` 流，包含：
        - `token`: 实时生成的文本片段
        - `tool_input`: 工具调用意图和参数
        - `tool_output`: 工具执行结果
        - `result`: 最终生成的回复
        - `error`: 执行过程中的非致命错误或异常
        
        参数:
            input_messages: 输入消息列表
            timeout: 本次执行的超时时间(秒)。为 None 时使用全局配置 settings.default_model_timeout。
            **kwargs: 传递给引擎的其他参数
        """
        # 统一超时配置：优先使用调用方传入，其次使用全局默认值
        if timeout is None:
            timeout = get_settings().default_model_timeout
        
        # 1. 基础校验与 Hook
        self.validate_input(input_messages)
        await self.before_step(input_messages, **kwargs)

        # 2. 构建执行上下文 (Context) - 这是线程安全的局部变量
        context = await self._build_execution_context(input_messages)
        start_time = time.time()
        
        try:
            # 3. 生命周期主循环，带超时保护和优雅关闭
            async for event in self._execute_lifecycle_with_timeout(context, timeout, **kwargs):
                yield event
        except AgentError as e:
            logger.error(f"step_stream caught AgentError: {e}")
            yield AgentStreamEvent(type="error", content=str(e))
            raise
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(
                f"step_stream timeout",
                elapsed_seconds=elapsed,
                max_timeout=timeout,
                current_turn=context.turn,
                message_count=len(context.messages)
            )
            
            # 尝试保存上下文（graceful shutdown）
            try:
                await self._save_context(context, force=True)
                logger.debug(f"Context saved before timeout shutdown")
            except Exception as save_error:
                logger.warning(f"Failed to save context on timeout: {save_error}")
            
            # 向客户端报告
            yield AgentStreamEvent(
                type="error",
                content=f"Execution timeout after {timeout}s at turn {context.turn}"
            )
            raise
            
        except asyncio.CancelledError:
            logger.warning(
                f"step_stream was cancelled",
                current_turn=context.turn
            )
            # 也要保存上下文
            try:
                await self._save_context(context, force=True)
            except Exception:
                pass
            raise
            
        except Exception as e:
            logger.exception("Lifecycle execution failed", error=str(e))
            await self.on_error(e, input_messages, **kwargs)
            # 将未捕获的异常转换为 Error 事件抛出给前端
            yield AgentStreamEvent(type="error", content=str(e))
            raise


    async def _execute_lifecycle_with_timeout(
        self, 
        context: ExecutionContext, 
        timeout: float, 
        **kwargs: Any
    ) -> AsyncIterator[AgentStreamEvent]:
        """生命周期执行，包含超时控制。"""
        import asyncio
        try:
            # Python 3.11+ 使用原生 asyncio.timeout
            if hasattr(asyncio, 'timeout'):
                async with asyncio.timeout(timeout):
                    async for event in self._execute_lifecycle(context, **kwargs):
                        yield event
            else:
                # Python 3.10 及以下兼容方案
                async for event in asyncio.wait_for(
                    self._execute_lifecycle(context, **kwargs), # type: ignore
                    timeout=timeout
                ):
                    yield event
        except asyncio.TimeoutError:
            # 重新抛出，让 step_stream 处理
            raise

    # ================= 生命周期主循环 (Lifecycle) =================

    async def _execute_lifecycle(
        self, 
        context: ExecutionContext, 
        **kwargs: Any
    ) -> AsyncIterator[AgentStreamEvent]:
        """
        ReAct 核心循环：Think -> Act -> Observe
        """
        
        # 预处理：检查是否在进行结构化输出
        # 如果是，我们需要获取那个“虚拟工具”的名称，以便拦截它
        response_model = kwargs.get('response_model')
        structure_tool_name = None
        if response_model and self._supports_functions:
            schema = StructureEngine.to_openai_tool(response_model)
            structure_tool_name = schema["function"]["name"]

        while context.turn < self.max_turns:
            context.turn += 1
            
            # Hook: 轮次开始
            if self.on_turn_start:
                await ensure_awaitable(self.on_turn_start, context)

            # ---------------- Phase 1: Think (Reasoning) ----------------
            buffer = StreamBuffer() # 创建新的流式缓冲区
            
            # 调用底层 LLM 生成流 (yield StreamChunk)
            async for chunk in self._phase_think(context, **kwargs):
                # 实时计算文本增量并加入缓冲区
                text_delta = buffer.add_chunk(chunk)
                if text_delta:
                    # 实时 yield 文本 Token 给前端
                    yield AgentStreamEvent(type="token", content=text_delta)
            
            # 从缓冲区构建完整的 Assistant 消息 (包含清洗过的 JSON 参数)
            assistant_msg = buffer.build_message()
            context.add_message(assistant_msg)

            # --- Safety Check: 死循环熔断 ---
            if self._detect_loop(context, assistant_msg):
                err_msg = "Infinite loop detected"
                yield AgentStreamEvent(type="error", content=err_msg)
                raise AgentError(err_msg)

            # --- Decision: 检查是否调用了结构化输出工具 ---
            # 如果 LLM 调用了我们指定的结构化工具，说明它完成了任务
            # 我们拦截这个调用，不传给 ToolBox，直接作为结果返回
            tool_calls = assistant_msg.safe_tool_calls
            
            if structure_tool_name and tool_calls:
                # 查找目标工具
                target_call = next((tc for tc in tool_calls if tc["function"]["name"] == structure_tool_name), None)
                if target_call:
                    # 这是一个结构化输出结果
                    final_output = AgentOutput(
                        content="", # 内容在 tool_calls 里
                        tool_calls=[target_call], 
                        metadata={"is_structured": True}
                    )
                    yield AgentStreamEvent(type="result", data={"output": final_output})
                    break # 任务结束

            # --- Decision: 正常结束 ---
            # 如果没有工具调用，说明 LLM 输出了纯文本回复，任务结束
            if not tool_calls:
                final_output = AgentOutput(
                    content=str(assistant_msg.content),
                    tool_calls=[],
                )
                yield AgentStreamEvent(type="result", data={"output": final_output})
                break

            # ---------------- Phase 2: Act (Tool Execution) ----------------
            # 通知前端：即将执行工具
            yield AgentStreamEvent(
                type="tool_input", 
                data={"tools": tool_calls}
            )
            
            # 执行工具 (并发)
            tool_results = await self._phase_act(tool_calls)
            
            for res in tool_results:
                # 截断过长的输出，防止 Context Window 爆炸
                content_to_save = self._truncate_observation(res.result, res.tool_name)
                
                # 将结果写入上下文
                context.add_message(
                    Message.tool_result(res.call_id, content_to_save, res.tool_name)
                )
                
                # 通知前端：工具执行完成
                yield AgentStreamEvent(
                    type="tool_output", 
                    content=res.result,
                    data={"tool_name": res.tool_name, "is_error": res.is_error}
                )

            # ---------------- Phase 3: Observe (Reflection Hook) ----------------
            # 这是一个关键的扩展点，子类可以重写此方法实现 Reflexion 或 Human-in-the-loop
            should_continue = await self._phase_observe(context, tool_results)
            
            # Hook: 轮次结束
            if self.on_turn_end:
                await ensure_awaitable(self.on_turn_end, context)
            
            # Checkpoint: 持久化上下文 (防止进程崩溃导致进度丢失)
            await self._save_context(context)

            if not should_continue:
                stop_output = AgentOutput(content="Task stopped by system monitor.")
                yield AgentStreamEvent(type="result", data={"output": stop_output})
                break

    # ================= 阶段实现 (Protected Methods) =================
    # 这些方法设计为可被子类 (如 ReflexionEngine) 重写

    async def _phase_think(
        self, 
        context: ExecutionContext, 
        **kwargs: Any
    ) -> AsyncIterator[StreamChunk]:
        """
        Thinking 阶段：构造 Prompt 并调用 LLM。
        
        负责：
        1. 构造 LLM 调用参数
        2. 流式接收模型响应
        3. 追踪 token 使用和成本
        """
        messages_payload = [m.to_openai_format() for m in context.messages]
        
        # 1. 构造参数
        llm_params = self._build_llm_params(kwargs.get('response_model'), "auto")
        
        # 2. 合并用户传入的 kwargs
        safe_kwargs = {k: v for k, v in kwargs.items() if k not in ['response_model']}
        llm_params.update(safe_kwargs)
        llm_params["stream"] = True
        
        # 3. Token 追踪变量
        input_tokens = 0
        output_tokens = 0
        model_name = self._get_model_name()
        
        try:
            stream_gen = self.model.astream(messages=messages_payload, **llm_params) # type: ignore
            async for chunk in stream_gen:
                # 验证块格式
                if not isinstance(chunk, StreamChunk):
                    logger.warning(f"Invalid StreamChunk type: {type(chunk)}")
                    continue
                
                # 提取 usage 信息（通常在最后的 chunk 中）
                if hasattr(chunk, 'usage') and chunk.usage: # type: ignore
                    # 使用 max 避免 usage 被覆盖
                    input_tokens = max(input_tokens, getattr(chunk.usage, 'prompt_tokens', 0)) # type: ignore
                    output_tokens = max(output_tokens, getattr(chunk.usage, 'completion_tokens', 0)) # type: ignore
                
                yield chunk
            
            # 记录 token 使用到统计（关键！）
            if input_tokens > 0 or output_tokens > 0:
                # 直接更新 stats 中的 token 数
                if self.stats:
                    self.stats.input_tokens += input_tokens
                    self.stats.output_tokens += output_tokens
                
                # 同时记录成本
                self.record_cost(input_tokens, output_tokens, model_name)
                
                logger.debug(
                    f"Tokens tracked in phase_think",
                    input=input_tokens,
                    output=output_tokens,
                    model=model_name
                )
        
        except Exception as e:
            logger.error(f"Model streaming failed: {e}", error_type=type(e).__name__)
            # 通知上游流处理已失败
            yield AgentStreamEvent(
                type="error",
                content=f"Model API error: {e}"
            ) # type: ignore
            raise
    
    def _get_model_name(self) -> str:
        """安全地获取模型名称，包含多个备选方案。"""
        # 优先级顺序：model_name > model > 默认值
        model_name = (
            getattr(self.model, 'model_name', None) or
            getattr(self.model, 'model', None) or
            'gpt-3.5-turbo'
        )
        
        if not isinstance(model_name, str):
            logger.warning(f"Model name is not a string: {type(model_name)}")
            model_name = 'gpt-3.5-turbo'
        
        return model_name

    async def _phase_act(
        self, 
        tool_calls: List[Dict[str, Any]]
    ) -> List[ToolExecutionResult]:
        """
        Acting 阶段：标准化工具调用并并发执行。
        """
        # _normalize_tool_call 会尝试解析 JSON 字符串为 Dict
        flat_calls = [self._normalize_tool_call(tc) for tc in tool_calls]
        return await self.toolbox.execute_many(flat_calls)

    async def _phase_observe(
        self, 
        context: ExecutionContext, 
        results: List[ToolExecutionResult]
    ) -> bool:
        """
        Observing 阶段：分析执行结果，决定是否继续。
        
        [P2-4 Fix] 改进的策略：
        - 统计连续错误次数。
        - 最多允许 2 轮自动重试，防止无限循环。
        - 使用系统反馈而非用户消息，避免 OpenAI API 兼容性问题。
        """
        error_count = sum(1 for r in results if r.is_error)
        
        if error_count > 0:
            context.consecutive_errors += 1
        else:
            context.consecutive_errors = 0
        
        # [P2-4 Fix] 最多允许 2 轮自动重试，防止无限循环
        max_auto_retries = 2
        
        if context.consecutive_errors >= 3:
            if context.turn < max_auto_retries:
                logger.warning(
                    f"Tool errors detected: {error_count}/{len(results)}, auto-retrying",
                    consecutive_errors=context.consecutive_errors,
                    turn=context.turn
                )
                
                # 构造错误摘要
                error_details = "\n".join(
                    f"- {r.tool_name}: {r.result}" for r in results if r.is_error
                )
                
                # [优化] 将错误信息放入 content，而非仅在 metadata
                # 提示词明确要求模型分析错误
                system_feedback = (
                    "System Notification: Multiple tool execution errors detected.\n"
                    "Error Details:\n"
                    f"{error_details}\n\n"
                    "Please analyze these errors, adjust your parameters or tool choice, and try again."
                )

                context.add_message(Message.assistant(
                    content=system_feedback,  # ✅ 填充实际内容
                    tool_calls=None,          # 明确为 None
                    metadata={"type": "system_reflection", "error_summary": error_details} # type: ignore
                ))
                
                context.consecutive_errors = 0
                return True
            else:
                # [P2-4 Fix] 超过自动重试上限，停止执行
                logger.error(
                    "Tool error threshold exceeded, stopping auto-retry",
                    consecutive_errors=context.consecutive_errors,
                    turn=context.turn,
                    max_retries=max_auto_retries
                )
                return False
            
        return True

    # ================= 辅助方法 (Helpers) =================

    def _detect_loop(self, context: ExecutionContext, msg: Message) -> bool:
        """改进的死循环检测算法。
        
        策略：
        1. 检测连续重复（A->A->A）
        2. 检测振荡模式（A->B->A）
        3. 避免合法的重复工具调用（带不同参数）
        """
        import hashlib
        
        if not msg.safe_tool_calls:
            return False
        
        try:
            # 计算当前工具调用的指纹
            calls_dump = json.dumps(
                [
                    {
                        "name": tc.get("function", {}).get("name"),
                        "args": tc.get("function", {}).get("arguments"),
                    }
                    for tc in msg.safe_tool_calls
                ],
                sort_keys=True,
            )
            current_hash = hashlib.sha256(calls_dump.encode()).hexdigest()
            
            # 策略 1: 检测严格的连续循环（同一工具调用重复 N 次）
            consecutive_count = sum(
                1 for h in reversed(context.last_tool_hashes) 
                if h == current_hash
            )
            
            if consecutive_count >= 3:  # 允许最多 3 次相同调用
                logger.warning(
                    f"Consecutive tool call loop detected ({consecutive_count} times)",
                    tool_hash=current_hash[:16],
                    advice="Consider adding early termination logic or changing tool parameters"
                )
                return True
            
            # 策略 2: 检测振荡模式（A->B->A->B）
            if len(context.last_tool_hashes) >= 2:
                # 检查当前 hash 是否与倒数第 2 个相同（且与倒数第 1 个不同）
                if (current_hash == context.last_tool_hashes[-2] and
                    current_hash != context.last_tool_hashes[-1]):
                    logger.warning(
                        "Oscillation pattern detected (A-B-A pattern)",
                        tool_hash=current_hash[:16],
                        advice="Model is alternating between two tool calls. Consider rephrasing instructions."
                    )
                    return True
            
            # 添加当前哈希到历史（保留最近 5 轮）
            context.last_tool_hashes.append(current_hash) # type: ignore
            context.last_tool_hashes = context.last_tool_hashes[-5:]
            context.last_tool_hash = current_hash  # type: ignore # 保持向后兼容
            
            return False
            
        except Exception as e:
            logger.warning(f"Loop detection failed: {e}", exc_info=True)
            # 失败时不触发熔断，继续执行（fail-open）
            return False

    def _truncate_observation(self, content: str, tool_name: str) -> str:
        """截断过长的工具输出，保留头部信息"""
        if len(content) > self.max_observation_length:
            logger.info(
                f"Truncating output for tool {tool_name}", 
                original_len=len(content)
            )
            return (
                content[: self.max_observation_length]
                + f"\n...(truncated, total {len(content)} chars)"
            )
        return content

    def _normalize_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        将 OpenAI 格式工具调用转换为 ToolBox 所需的扁平格式。
        在此处尝试解析 arguments JSON 字符串。
        """
        func_block = tool_call.get("function", {})
        name = func_block.get("name", "")
        raw_args = func_block.get("arguments", "{}")
        
        parsed_args: Dict[str, Any] = {}
        try:
            if isinstance(raw_args, str):
                parsed_args = json.loads(raw_args)
            elif isinstance(raw_args, dict):
                parsed_args = raw_args
        except json.JSONDecodeError as e:
            # 解析失败时，不抛出异常，而是传递特殊标记给 ToolBox
            # ToolBox 会识别此标记并返回友好的错误提示给 LLM
            parsed_args = {
                "__gecko_parse_error__": f"JSON format error: {str(e)}. Content: {raw_args}"
            }

        return {
            "id": tool_call.get("id", ""),
            "name": name,
            "arguments": parsed_args,
        }

    async def _build_execution_context(self, input_messages: List[Message]) -> ExecutionContext:
        """加载历史并构建 ExecutionContext"""
        history = await self._load_history()
        all_messages = history + input_messages
        
        # [P1-5 Fix] 更严格的系统提示管理
        # 检查是否已有 system 消息（包括 input_messages 中的）
        system_msg = next((m for m in all_messages if m.role == "system"), None)
        
        if system_msg is None:
            # 需要注入系统提示
            template_vars = {
                "tools": self.toolbox.to_openai_schema(),
                "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            try:
                system_content = self.prompt_template.format_safe(**template_vars)
                # [P1-5 Fix] 确保 system 消消息在最前面（OpenAI API 要求）
                all_messages.insert(0, Message.system(system_content))
            except Exception as e:
                logger.warning(f"System prompt formatting failed: {e}")
                # [P1-5 Fix] Fallback: 使用最小系统提示而非让其缺失
                all_messages.insert(0, Message.system("You are a helpful AI assistant."))
        elif any(m.role == "system" for m in input_messages):
            # [P1-5 Fix] 用户在 input_messages 中显式指定了 system，记录日志
            logger.info("User-specified system message will be used")
            
        return ExecutionContext(all_messages)

    async def _load_history(self) -> List[Message]:
        """从 Memory 加载历史消息"""
        if not self.memory.storage:
            return []
        try:
            data = await self.memory.storage.get(self.memory.session_id)
            if data and "messages" in data:
                return await self.memory.get_history(data["messages"])
        except Exception:
            return []
        return []

    async def _save_context(self, context: ExecutionContext, force: bool = False) -> None:
        """保存当前上下文到 Memory"""
        if not self.memory.storage:
            return
        try:
            messages_data = [m.to_openai_format() for m in context.messages]
            await self.memory.storage.set(
                self.memory.session_id, {"messages": messages_data}
            )
        except Exception as e:
            logger.warning("Failed to save context", error=str(e))

    def _build_llm_params(self, response_model: Any, strategy: str) -> Dict[str, Any]:
        """
        构建 LLM 调用参数 (Tools, Tool Choice)
        """
        params: Dict[str, Any] = {}
        tools_schema = self.toolbox.to_openai_schema()

        # 1. 结构化输出模式 (Structure Mode)
        if response_model and self._supports_functions:
            # 将 Response Model 转换为 Tool Schema
            structure_tool = StructureEngine.to_openai_tool(response_model)
            
            # 合并现有工具 (不能修改原列表)
            combined_tools = tools_schema + [structure_tool]
            params["tools"] = combined_tools
            
            # 强制调用该工具 (OpenAI `tool_choice` syntax)
            params["tool_choice"] = {
                "type": "function",
                "function": {"name": structure_tool["function"]["name"]}
            }
            
        # 2. 标准 ReAct 模式 (Standard Mode)
        elif tools_schema and self._supports_functions:
            params["tools"] = tools_schema
            params["tool_choice"] = "auto"
            
        return params