from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, cast

from pydantic import BaseModel

from gecko.core.events import AgentRunEvent, EventBus
from gecko.core.events.types import AgentStreamEvent
from gecko.core.message import Message
from gecko.core.output import AgentOutput
from gecko.core.toolbox import ToolBox
from gecko.core.memory import TokenMemory
from gecko.core.engine.base import CognitiveEngine
from gecko.core.engine.react import ReActEngine
from gecko.core.logging import get_logger
from gecko.core.exceptions import AgentError

from gecko.config import get_settings
from gecko.core.limits import global_limiter
from gecko.core.telemetry import get_telemetry

logger = get_logger(__name__)


# 统一消息输入类型：
# - 允许纯字符串、单条 Message、dict、以及由 Message/dict 组成的 list/tuple
# - 这样 run/stream/stream_events 的签名更简洁，可读性更好
MessagesInput = (
    str
    | Message
    | list[Message]
    | tuple[Message, ...]
    | list[dict[str, Any]]
    | tuple[dict[str, Any], ...]
    | dict[str, Any]
)


class Agent:
    """
    Agent 负责协调「模型 ⟺ 工具箱 ⟺ 记忆」之间的一次完整推理流程。

    对外暴露三个核心接口：
    - run          : 单次推理，返回结构化结果（AgentOutput 或 Pydantic 模型）
    - stream       : 文本流式推理，只关心 token 文本
    - stream_events: 高级流接口，直接暴露 AgentStreamEvent 事件流

    设计要点：
    - 内部通过 CognitiveEngine 实现具体的 ReAct 等策略；
    - 通过 EventBus 对外抛出运行事件，方便日志、审计、指标采集；
    - 通过 Telemetry（如 OpenTelemetry）统一打点 Trace/Span；
    - 通过 global_limiter 实现 agent 级别的并发限流。
    """

    def __init__(
        self,
        model: Any,
        toolbox: ToolBox,
        memory: TokenMemory,
        engine_cls: type[CognitiveEngine] = ReActEngine,
        event_bus: EventBus | None = None,
        **engine_kwargs: Any,
    ):
        """
        初始化 Agent。

        参数说明：
        - model:      具体的模型驱动对象（需实现同步/异步推理接口）
        - toolbox:    工具箱对象，封装所有可被调用的 Tool
        - memory:     记忆模块（如 TokenMemory），负责上下文记忆与摘要
        - engine_cls: 推理引擎类，默认使用 ReActEngine，可替换为其它策略
        - event_bus:  事件总线，若不传则内部创建一个新的 EventBus
        - engine_kwargs: 透传给引擎的其他配置，如 name、温度、最大步数等
        """
        # 事件总线：用于对外广播 run/stream 等生命周期事件
        self.event_bus: EventBus = event_bus or EventBus()  # type: ignore
        self.toolbox = toolbox
        self.memory = memory
        self.model = model

        # 引擎实例：
        # - 引擎本身应尽量无状态（一次请求的状态放在 ExecutionContext 内部）
        # - 便于在同一 Agent 中多次调用 engine.step / engine.step_stream
        self.engine: CognitiveEngine = engine_cls(
            model=model,
            toolbox=toolbox,
            memory=memory,
            event_bus=self.event_bus,
            **engine_kwargs,
        )

        # Agent 名称：
        # - 用于日志标识、限流维度（global_limiter 的 key）
        # - 如果子类已经定义了 self.name，则优先使用子类的；否则使用 kwargs 中的 name；再否则为 "default"
        self.name = getattr(self, "name", engine_kwargs.get("name", "default"))

    # ----------------------------------------------------------------------
    # 核心调用接口
    # ----------------------------------------------------------------------
    async def run(
        self,
        messages: MessagesInput,
        response_model: type[BaseModel] | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> AgentOutput | BaseModel:
        """
        单次推理接口：将各种输入格式统一转成 Message 列表，然后交给引擎处理，
        最终返回 AgentOutput 或 Pydantic 模型实例（如果指定了 response_model）。

        典型调用示例：
            result = await agent.run("帮我总结一下这段文本")
            result = await agent.run(
                [{"role": "user", "content": "你好"}],
                response_model=MySchema,
            )
        """
        telemetry = get_telemetry()

        # 1. 规范化消息，并拼接 raw_text 方便日志/审计
        input_msgs, raw_text = self._prepare_input(messages)

        # 2. 发出 "run_started" 事件
        await self.event_bus.publish(
            AgentRunEvent(
                type="run_started",
                data=self._build_input_metadata(input_msgs, raw_text),
            )
        )

        # 3. 确定本次调用的超时时间：优先用参数；否则用全局配置
        effective_timeout = self._resolve_timeout(timeout)
        agent_label = self.name

        async def _do_run() -> AgentOutput | BaseModel:
            # 使用统一的 Telemetry Span 封装一次完整的 run 调用
            async with self._telemetry_span(
                telemetry,
                span_name="gecko.agent.run",
                agent_label=agent_label,
                input_msgs=input_msgs,
                raw_text=raw_text,
                timeout=effective_timeout,
            ) as span:
                try:
                    # 调用引擎的单次推理接口
                    output = await self.engine.step(
                        input_msgs,
                        response_model=response_model,
                        timeout=effective_timeout,
                        **kwargs,
                    )

                    payload = self._serialize_output(output)

                    # 标记本次是否有 content 字段（方便后续分析）
                    if span:
                        span.set_attribute("output.has_content", bool(payload.get("content")))

                    # 发出 "run_completed" 事件
                    await self.event_bus.publish(
                        AgentRunEvent(type="run_completed", data={"output": payload})
                    )
                    return output

                # 业务级错误：AgentError
                except AgentError as ge:
                    logger.warning("Agent run failed with AgentError: %s", ge)
                    await self.event_bus.publish(AgentRunEvent(type="run_error", error=str(ge)))
                    if span:
                        span.record_exception(ge)
                    raise

                # 其他未预期异常：统一视为系统错误
                except Exception as e:
                    logger.exception("Agent run failed")
                    await self.event_bus.publish(AgentRunEvent(type="run_error", error=str(e)))
                    if span:
                        span.record_exception(e)
                    raise

        # 4. 包一层 agent 级限流（如：每个 Agent 最多并发 N 个请求）
        async with self._agent_limit(agent_label):
            return await _do_run()

    async def stream(
        self,
        messages: MessagesInput,
        timeout: float | None = None,
    ) -> AsyncIterator[str]:
        """
        简化版流式接口：只返回模型增量生成的文本 token。

        适用场景：
        - 前端流式展示（如 Chat UI），只关心文本，不关心工具调用等中间细节。
        """
        telemetry = get_telemetry()
        input_msgs, raw_text = self._prepare_input(messages)

        # 发出 "stream_started" 事件
        await self.event_bus.publish(
            AgentRunEvent(
                type="stream_started",
                data=self._build_input_metadata(input_msgs, raw_text),
            )
        )

        effective_timeout = self._resolve_timeout(timeout)
        agent_label = self.name

        async def _gen() -> AsyncIterator[str]:
            # 和 run 一样，统一封装一个 Telemetry Span
            async with self._telemetry_span(
                telemetry,
                span_name="gecko.agent.stream",
                agent_label=agent_label,
                input_msgs=input_msgs,
                raw_text=raw_text,
                timeout=effective_timeout,
            ) as span:
                try:
                    # 调用引擎的流式接口：
                    # 注意：engine.step_stream 返回一个「可 await 的对象」，await 之后得到 AsyncIterator
                    async for event in await self.engine.step_stream(
                        input_msgs, timeout=effective_timeout
                    ):
                        # 只把 token 类型的事件透传给上层调用方
                        if event.type == "token" and event.content is not None:
                            yield str(event.content)

                        # 流内 error 事件：记录日志 + span 上记录异常
                        if event.type == "error":
                            logger.warning("Stream error event: %s", event.content)
                            if span:
                                span.record_exception(AgentError(str(event.content)))

                    # 正常结束：发出 "stream_completed" 事件
                    await self.event_bus.publish(AgentRunEvent(type="stream_completed"))

                except Exception as e:
                    # 整个流失败（而不仅仅是中间的 error 事件）
                    logger.exception("Agent stream failed")
                    await self.event_bus.publish(AgentRunEvent(type="stream_error", error=str(e)))
                    if span:
                        span.record_exception(e)
                    raise

        # 统一包上一层 agent 级限流
        async with self._agent_limit(agent_label):
            async for chunk in _gen():
                yield chunk

    async def stream_events(
        self,
        messages: MessagesInput,
        timeout: float | None = None,
    ) -> AsyncIterator[AgentStreamEvent]:
        """
        高级流式接口：直接返回 AgentStreamEvent 事件流。

        事件类型（示例）：
        - token       : 模型增量文本
        - tool_input  : 即将调用某个工具（包含工具名和参数）
        - tool_output : 工具执行结果
        - result      : 最终 AgentOutput
        - error       : 中间非致命错误（引擎内部可恢复）
        """
        telemetry = get_telemetry()
        input_msgs, raw_text = self._prepare_input(messages)
        effective_timeout = self._resolve_timeout(timeout)
        agent_label = self.name

        # 与 run/stream 对齐，增加 stream_events 的生命周期事件
        await self.event_bus.publish(
            AgentRunEvent(
                type="stream_events_started",
                data=self._build_input_metadata(input_msgs, raw_text),
            )
        )

        async def _gen() -> AsyncIterator[AgentStreamEvent]:
            # 为事件流也补齐 Telemetry Span
            async with self._telemetry_span(
                telemetry,
                span_name="gecko.agent.stream_events",
                agent_label=agent_label,
                input_msgs=input_msgs,
                raw_text=raw_text,
                timeout=effective_timeout,
            ) as span:
                try:
                    # 调用引擎流式接口，直接将事件 yield 给上层
                    async for event in await self.engine.step_stream(
                        input_msgs, timeout=effective_timeout
                    ):
                        yield event

                    # 正常结束：发出 "stream_events_completed" 事件
                    await self.event_bus.publish(AgentRunEvent(type="stream_events_completed"))
                except Exception as e:
                    # 整个流异常：记录日志 + span 记录异常 + 发布错误事件
                    logger.exception("Agent stream_events failed, propagating error event")
                    if span:
                        span.record_exception(e)

                    await self.event_bus.publish(
                        AgentRunEvent(type="stream_events_error", error=str(e))
                    )

                    # 仍然尝试向前端发送一个 error 事件，提示错误原因
                    yield AgentStreamEvent(type="error", content=str(e))
                    raise

        async with self._agent_limit(agent_label):
            async for ev in _gen():
                yield ev

    # ----------------------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------------------
    def _normalize_messages(self, messages: MessagesInput) -> list[Message]:
        """
        将多种输入形式统一转换为标准的 list[Message]。

        输入支持：
        1) str                 -> [Message.user(str)]
        2) Message             -> [Message]
        3) dict:
            - 含 "role" 字段   -> [Message(**dict)]
            - 否则视为 {"input": "..."} -> [Message.user(input)]
        4) list/tuple:
            - 若是 list 且首元素为 Message，则直接原样返回（不拷贝、不转换）
            - 否则依次遍历：
                - Message -> 直接加入
                - dict    -> Message(**dict)
                - 其他    -> 抛 AgentError
        """
        # 1) 单条 Message
        if isinstance(messages, Message):
            return [messages]

        # 2) 字符串 -> 转成单条 user 消息
        if isinstance(messages, str):
            return [Message.user(messages)]

        # 3) 单条 dict
        if isinstance(messages, dict):
            if "role" in messages:
                # 兼容 OpenAI ChatCompletion 风格：{"role": "...", "content": "..."}
                return [Message(**messages)]
            # 兼容 {"input": "..."} 这种简单结构
            text = messages.get("input") or str(messages)
            return [Message.user(text)]

        # 4) 序列（list 或 tuple）
        if isinstance(messages, (list, tuple)):
            if not messages:
                raise AgentError("消息列表为空")

            # 如果是 list 且首元素就是 Message，则直接返回原 list：
            # - 避免多余拷贝
            # - 保持对象 ID 不变（有测试会用 is 判断）
            if isinstance(messages, list) and isinstance(messages[0], Message):
                return cast(list[Message], messages)

            # 其余情况：统一逐个规范化为 Message
            normalized: list[Message] = []
            for item in messages:
                if isinstance(item, Message):
                    normalized.append(item)
                elif isinstance(item, dict):
                    normalized.append(Message(**item))
                else:
                    # 混入了不支持的元素类型
                    raise AgentError(f"无法识别的消息元素类型: {type(item)}")
            return normalized

        # 5) 其他类型一律视为错误
        raise AgentError(f"不支持的消息类型: {type(messages)}")

    def _prepare_input(self, messages: MessagesInput) -> tuple[list[Message], str]:
        """
        做两件事：
        1) 调用 _normalize_messages 得到标准化的 list[Message]
        2) 将所有消息的文本内容拼接为 raw_text（换行分隔），用于：
           - 日志打印
           - 审计记录
           - Telemetry Span 的属性
        """
        input_msgs = self._normalize_messages(messages)
        raw_text = "\n".join(m.get_text_content() for m in input_msgs)
        return input_msgs, raw_text

    @staticmethod
    def _build_input_metadata(input_msgs: list[Message], raw_text: str) -> dict[str, Any]:
        """
        为 AgentRunEvent 构造统一的输入元数据结构。
        """
        return {"input": raw_text, "input_count": len(input_msgs)}

    def _serialize_output(self, output: AgentOutput | BaseModel) -> dict[str, Any]:
        """
        将 AgentOutput 或 Pydantic 模型序列化为 dict，便于事件总线/日志处理。

        当前策略：
        - 若对象有 model_dump 方法（Pydantic v2），则直接调用 model_dump()
        - 否则兜底为 {"content": str(output)}
        """
        if hasattr(output, "model_dump"):
            return output.model_dump()
        return {"content": str(output)}

    def _resolve_timeout(self, timeout: float | None) -> float:
        """
        统一解析超时时间：
        - None    -> 使用全局配置中的 default_model_timeout
        - 0       -> 允许，表示“立即超时”/极短 TTL（通常用于测试/特定控制）
        - < 0     -> 直接视为非法参数，抛 ValueError
        - 非数字  -> 抛 ValueError
        """
        if timeout is None:
            return get_settings().default_model_timeout

        if not isinstance(timeout, (int, float)):
            raise ValueError(f"Timeout must be a number, got {type(timeout)}")

        if timeout < 0:
            raise ValueError(f"Timeout cannot be negative: {timeout}")

        return float(timeout)

    @asynccontextmanager
    async def _telemetry_span(
        self,
        telemetry: Any,
        span_name: str,
        agent_label: str,
        input_msgs: list[Message],
        raw_text: str,
        timeout: float,
    ) -> AsyncIterator[Any]:
        """
        内部统一封装 Telemetry span 创建逻辑，避免 run/stream/stream_events 重复代码。

        所有 Span 都会带上以下公共属性：
        - agent.name    : 当前 Agent 名称
        - input.length  : 文本总长度（字符数）
        - input.count   : 消息条数
        - agent.timeout : 本次调用的超时时长（秒）
        """
        attributes = {
            "agent.name": agent_label,
            "input.length": len(raw_text),
            "input.count": len(input_msgs),
            "agent.timeout": timeout,
        }
        async with telemetry.async_span(span_name, attributes=attributes) as span:
            # 将 span 透传给上层，方便记录异常/附加属性等
            yield span

    @asynccontextmanager
    async def _agent_limit(self, agent_label: str) -> AsyncIterator[None]:
        """
        Agent 级并发限流统一入口。

        - 从全局配置 settings.agent_max_concurrent 读取并发上限；
        - 当值 <= 0 时不做任何限流；
        - 当值 > 0 时，通过 global_limiter.limit("agent", agent_label, max_c)
          对当前 Agent 名称做维度限流。

        用法示例：
            async with self._agent_limit(self.name):
                ...  # 执行一次完整的 Agent 调用
        """
        max_c = get_settings().agent_max_concurrent
        if max_c > 0:
            async with global_limiter.limit("agent", agent_label, max_c):
                yield
        else:
            yield
