# gecko/core/agent.py
from __future__ import annotations

from typing import Any, AsyncIterator, Iterable, List, Optional, Type, Union

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


class Agent:
    """
    Agent 对象负责在模型、工具箱、记忆之间协调一次推理任务。

    设计目标：
    - 封装一次「从输入消息 → 模型调用 → 工具调用 → 结构化输出」的完整流程；
    - 对调用方暴露一个简单稳定的 API（run / stream / stream_events）；
    - 隐藏引擎实现细节（ReAct、Reflexion 等），便于未来更换引擎实现；
    - 提供事件总线（EventBus）作为观测与扩展点（日志、审计、指标等）。
    """

    def __init__(
        self,
        model: Any,
        toolbox: ToolBox,
        memory: TokenMemory,
        engine_cls: Type[CognitiveEngine] = ReActEngine,
        event_bus: Optional[EventBus] = None,
        **engine_kwargs: Any,
    ):
        """
        初始化 Agent。

        参数：
            model:   实际模型驱动（需实现 ModelProtocol/StreamableModelProtocol）
            toolbox: 工具箱对象，负责管理并执行工具调用
            memory:  记忆模块（如 TokenMemory）
            engine_cls: 推理引擎类，默认使用 ReActEngine
            event_bus: 事件总线实例（可选）
            **engine_kwargs: 传递给引擎的其他配置参数
        """
        self.event_bus = event_bus or EventBus()
        self.toolbox = toolbox
        self.memory = memory
        self.model = model

        # 引擎实例（无状态，单次请求状态封装在 ExecutionContext 中）
        self.engine: CognitiveEngine = engine_cls(
            model=model,
            toolbox=toolbox,
            memory=memory,
            event_bus=self.event_bus,
            **engine_kwargs,
        )

        # Agent 名称：用于日志/限流维度标识（如果外层已有 name，可覆盖）
        self.name = getattr(self, "name", engine_kwargs.get("name", "default"))

    # ----------------------------------------------------------------------
    # 核心调用接口
    # ----------------------------------------------------------------------
    async def run(
        self,
        messages: str | Message | List[Message] | List[dict] | dict,
        response_model: Optional[Type[BaseModel]] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> AgentOutput | BaseModel:
        """
        单次推理入口：对多种输入格式统一转换为 Message 列表。

        功能说明：
        1. 调用 _normalize_messages 将 str/dict/List[...] 正规化为 Message 列表；
        2. 触发 AgentRunEvent("run_started") 事件，便于记录输入；
        3. 在统一的 Telemetry Span 中执行引擎 step：
           - 默认超时使用 settings.default_model_timeout，可通过参数覆盖；
           - Agent 级并发限流使用 settings.agent_max_concurrent；
        4. 将执行结果封装为 AgentOutput 或 Pydantic 模型（response_model）；
        5. 触发 AgentRunEvent("run_completed"/"run_error") 事件，便于审计与监控。
        """
        telemetry = get_telemetry()
        input_msgs = self._normalize_messages(messages)

        # 拼接用户可读文本，供日志 / Guardrails / 审计使用
        raw_text = "\n".join(m.get_text_content() for m in input_msgs)

        await self.event_bus.publish(
            AgentRunEvent(
                type="run_started",
                data={"input": raw_text, "input_count": len(input_msgs)},
            )
        )

        # 统一超时：优先使用调用者传入，其次使用全局配置
        effective_timeout = timeout or get_settings().default_model_timeout
        agent_label = self.name

        async def _do_run() -> AgentOutput | BaseModel:
            # 为一次 Agent.run 包一个顶层 Trace Span
            async with telemetry.async_span(
                "gecko.agent.run",
                attributes={
                    "agent.name": agent_label,
                    "input.length": len(raw_text),
                    "input.count": len(input_msgs),
                },
            ) as span:
                try:
                    # 将 timeout 显式透传到 Engine
                    output = await self.engine.step(
                        input_msgs,
                        response_model=response_model,
                        timeout=effective_timeout,
                        **kwargs,
                    )

                    payload = self._serialize_output(output)

                    if span:
                        span.set_attribute(
                            "output.has_content", bool(payload.get("content"))
                        )

                    await self.event_bus.publish(
                        AgentRunEvent(type="run_completed", data={"output": payload})
                    )
                    return output
                except AgentError as ge:
                    logger.warning("Agent run failed with AgentError: %s", ge)
                    await self.event_bus.publish(
                        AgentRunEvent(type="run_error", error=str(ge))
                    )
                    if span:
                        span.record_exception(ge)
                    raise
                except Exception as e:
                    logger.exception("Agent run failed", exc_info=True)
                    await self.event_bus.publish(
                        AgentRunEvent(type="run_error", error=str(e))
                    )
                    if span:
                        span.record_exception(e)
                    raise

        # Agent 级并发限流：scope="agent", name=self.name
        max_c = get_settings().agent_max_concurrent
        if max_c > 0:
            async with global_limiter.limit("agent", agent_label, max_c):
                return await _do_run()
        else:
            return await _do_run()

    async def stream(
        self,
        messages: str | Message | List[Message] | List[dict] | dict,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[str]:
        """
        流式推理（简易模式）：仅返回文本 Token。

        说明：
        - 这是对 Engine.step_stream 的一个「简化包装」：
          只对 `token` 事件进行 yield，其它事件（tool_input/tool_output/result/error）
          由更底层的 stream_events 处理；
        - 适合「只关心模型最终文本输出」的调用场景（如 Chat UI）。
        """
        telemetry = get_telemetry()
        input_msgs = self._normalize_messages(messages)
        raw_text = "\n".join(m.get_text_content() for m in input_msgs)

        await self.event_bus.publish(
            AgentRunEvent(
                type="stream_started",
                data={"input": raw_text, "input_count": len(input_msgs)},
            )
        )

        effective_timeout = timeout or get_settings().default_model_timeout
        agent_label = self.name

        async def _gen() -> AsyncIterator[str]:
            async with telemetry.async_span(
                "gecko.agent.stream",
                attributes={
                    "agent.name": agent_label,
                    "input.length": len(raw_text),
                    "input.count": len(input_msgs),
                },
            ) as span:
                try:
                    # 迭代 Engine 产生的高级事件
                    async for event in self.engine.step_stream(
                        input_msgs, timeout=effective_timeout
                    ):  # type: ignore[arg-type]
                        # 1. 如果是 Token，直接 yield 给用户
                        if event.type == "token" and event.content is not None:
                            yield str(event.content)

                        # 2. 其它事件（工具调用、最终结果等）由上层决定是否使用
                        #    - 这里不做处理，以保持 stream 接口的简单性
                        #    - 如果需要更精细的事件流，请使用 stream_events

                        # 3. 如果是 error 事件，记录日志并打点
                        if event.type == "error":
                            logger.warning("Stream error event: %s", event.content)
                            if span:
                                span.record_exception(
                                    AgentError(str(event.content))
                                )

                    await self.event_bus.publish(
                        AgentRunEvent(type="stream_completed")
                    )
                except Exception as e:
                    logger.exception("Agent stream failed", exc_info=True)
                    await self.event_bus.publish(
                        AgentRunEvent(type="stream_error", error=str(e))
                    )
                    if span:
                        span.record_exception(e)
                    raise

        max_c = get_settings().agent_max_concurrent
        if max_c > 0:
            async with global_limiter.limit("agent", agent_label, max_c):
                async for chunk in _gen():
                    yield chunk
        else:
            async for chunk in _gen():
                yield chunk

    async def stream_events(
        self,
        messages: str | Message | List[Message] | List[dict] | dict,
        timeout: Optional[float] = None,
    ) -> AsyncIterator[AgentStreamEvent]:
        """
        高级流式接口：直接返回 AgentStreamEvent 事件流。

        事件类型包括：
        - token:      增量文本 Token
        - tool_input: 即将调用某个工具（包含工具名和参数）
        - tool_output: 工具执行结果
        - result:     最终 AgentOutput
        - error:      中间非致命错误

        适用场景：
        - WebSocket / SSE 服务端，将事件原样推送到前端；
        - 自定义调试工具，实时观察 ReAct 内部行为；
        - 高级编排框架，把 tool_input/tool_output 事件接入其它系统。
        """
        input_msgs = self._normalize_messages(messages)
        effective_timeout = timeout or get_settings().default_model_timeout

        agent_label = self.name

        async def _gen() -> AsyncIterator[AgentStreamEvent]:
            try:
                async for event in self.engine.step_stream(
                    input_msgs, timeout=effective_timeout
                ):  # type: ignore[arg-type]
                    yield event
            except Exception as e:
                # 即使发生异常，也尝试 yield 一个 error 事件给前端
                logger.exception(
                    "Agent stream_events failed, propagating error event", exc_info=True
                )
                yield AgentStreamEvent(type="error", content=str(e))
                raise

        max_c = get_settings().agent_max_concurrent
        if max_c > 0:
            async with global_limiter.limit("agent", agent_label, max_c):
                async for ev in _gen():
                    yield ev
        else:
            async for ev in _gen():
                yield ev

    # ----------------------------------------------------------------------
    # 辅助方法
    # ----------------------------------------------------------------------
    def _normalize_messages(  
        self,  
        messages: str | Message | List[Message] | List[dict] | dict  
    ) -> List[Message]:  
        """  
        将多种输入形式转换为标准的 Message 列表。  
  
        支持以下输入：  
        1. 字符串 (str)  
           - 视为单条 user 消息，等价于 [Message.user(text)]  
  
        2. 单条 Message 对象  
           - 直接包装为列表 [Message] 返回  
  
        3. List[Message]  
           - 视为已经标准化好的消息列表，**原样返回**（保持对象 ID 不变）  
             （测试用例依赖这一点：传入的列表对象与传入 engine.step 的参数应为同一个）  
  
        4. dict  
           - 若包含 "role" 字段，按 Message 的字段进行构造：Message(**dict)  
           - 否则视为 {"input": "..."} 形式，从 input 字段中取文本作为 user 消息内容  
  
        5. List[dict]  
           - 每个 dict 转换为 Message(**item) 后返回新的列表  
        """  
        # 1) 单条 Message  
        if isinstance(messages, Message):  
            return [messages]  
  
        # 2) 字符串 -> 单条 user 消息  
        if isinstance(messages, str):  
            return [Message.user(messages)]  
  
        # 3) 单条 dict  
        if isinstance(messages, dict):  
            if "role" in messages:  
                # 兼容 OpenAI ChatCompletion 风格的 {"role": "...", "content": "..."}  
                return [Message(**messages)]  
            # 兼容 {"input": "..."} 这类简单调用形式  
            text = messages.get("input") or str(messages)  
            return [Message.user(text)]  
  
        # 4) List 场景  
        if isinstance(messages, list):  
            if not messages:  
                raise AgentError("消息列表为空")  
  
            # List[Message] -> 原样返回，保持对象 ID 不变（测试用例有 is 判断）  
            if isinstance(messages[0], Message):  
                return messages  # type: ignore[return-value]  
  
            # List[dict] / 混合列表 -> 逐个规范化  
            normalized: List[Message] = []  
            for item in messages:  
                if isinstance(item, Message):  
                    normalized.append(item)  
                elif isinstance(item, dict):  
                    normalized.append(Message(**item))  
                else:  
                    raise AgentError(f"无法识别的消息元素类型: {type(item)}")  
            return normalized  
  
        # 5) 其他类型一律视为错误  
        raise AgentError(f"不支持的消息类型: {type(messages)}") 
    
    def _serialize_output(self, output: AgentOutput | BaseModel) -> dict:
        """
        将 AgentOutput 或 Pydantic 模型序列化为字典，用于事件总线/日志。

        - 对于 Pydantic 模型，优先使用 model_dump()
        - 对于 AgentOutput，直接返回其 model_dump()
        - 兜底为 {"content": str(output)}
        """
        if hasattr(output, "model_dump"):
            return output.model_dump()  # type: ignore[return-value]
        return {"content": str(output)}
