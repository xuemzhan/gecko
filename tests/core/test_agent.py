# tests/core/test_agent.py
import asyncio
from typing import Any

import pytest
from unittest.mock import AsyncMock, MagicMock

from gecko.core.agent import Agent
from gecko.core.events.bus import EventBus
from gecko.core.events.types import AgentStreamEvent
from gecko.core.exceptions import AgentError
from gecko.core.message import Message


@pytest.mark.asyncio
async def test_agent_run_input_formats(mock_llm, toolbox, memory, event_bus) -> None:
    agent = Agent(model=mock_llm, toolbox=toolbox, memory=memory, event_bus=event_bus)

    # Mock engine step to avoid actual ReAct logic here
    agent.engine.step = AsyncMock(return_value=MagicMock(content="ok", model_dump=lambda: {}))

    # 1. String input
    await agent.run("hello")
    call_args = agent.engine.step.call_args[0][0]
    assert isinstance(call_args, list)
    assert isinstance(call_args[0], Message)
    assert call_args[0].content == "hello"

    # 2. List[Message] input (优化路径)
    msgs = [Message.user("test")]
    await agent.run(msgs)
    call_args = agent.engine.step.call_args[0][0]
    # 对于 List[Message]，应保持原对象（不拷贝）
    assert call_args is msgs

    # 3. Dict input {"input": "..."}
    await agent.run({"input": "dict_test"})
    call_args = agent.engine.step.call_args[0][0]
    assert call_args[0].content == "dict_test"


@pytest.mark.asyncio
async def test_agent_events(mock_llm, toolbox, memory, event_bus) -> None:
    """测试 Agent 生命周期事件 run_started / run_completed"""
    received_events = []

    async def handler(e):
        received_events.append(e)

    # 订阅具体的事件类型
    event_bus.subscribe("run_started", handler)
    event_bus.subscribe("run_completed", handler)
    event_bus.subscribe("run_error", handler)

    agent = Agent(model=mock_llm, toolbox=toolbox, memory=memory, event_bus=event_bus)

    # Mock output
    mock_output = MagicMock()
    mock_output.content = "ok"
    mock_output.model_dump = lambda: {"content": "ok"}

    agent.engine.step = AsyncMock(return_value=mock_output)

    # 执行 Agent
    await agent.run("start")

    # 等待 EventBus 的后台任务执行完毕
    await asyncio.sleep(0.1)

    # 检查是否收到事件
    assert len(received_events) >= 2

    event_types = [e.type for e in received_events]
    assert "run_started" in event_types
    assert "run_completed" in event_types


@pytest.mark.asyncio
async def test_agent_event_bus_wiring(mock_llm, toolbox, memory) -> None:
    """
    验证 Agent 的 EventBus 是否正确连接到了 Engine，
    以及未显式传入时是否自动创建 EventBus 实例。
    """
    bus = EventBus()
    agent = Agent(
        model=mock_llm,
        toolbox=toolbox,
        memory=memory,
        event_bus=bus,
    )

    # 验证 Engine 拥有 event_bus 属性且是同一个实例
    assert hasattr(agent.engine, "event_bus")
    assert agent.engine.event_bus is bus

    # 验证如果不传，Agent 会自动创建
    agent_default = Agent(model=mock_llm, toolbox=toolbox, memory=memory)
    assert agent_default.engine.event_bus is not None
    assert isinstance(agent_default.engine.event_bus, EventBus)


@pytest.mark.asyncio
async def test_agent_run_uses_global_concurrency_limiter(
    monkeypatch,
    mock_llm,
    toolbox,
    memory,
    event_bus,
) -> None:
    """
    验证 run 在 agent_max_concurrent > 0 时会走 global_limiter.limit
    """
    from gecko.config import configure_settings
    from gecko.core.limits import global_limiter
    from gecko.core.output import AgentOutput

    # 1. 设置 agent_max_concurrent = 1
    configure_settings(agent_max_concurrent=1)

    agent = Agent(model=mock_llm, toolbox=toolbox, memory=memory, event_bus=event_bus)

    # 2. Mock engine.step，避免跑 ReAct 逻辑
    step_calls = {}

    async def fake_step(messages, **kwargs):
        step_calls["called"] = True
        return AgentOutput(content="ok")

    agent.engine.step = AsyncMock(side_effect=fake_step)

    # 3. Mock global_limiter.limit，记录调用参数 & 确认被用作 async context manager
    limit_calls: dict[str, Any] = {}

    class DummyCM:
        async def __aenter__(self):
            limit_calls["entered"] = True

        async def __aexit__(self, exc_type, exc, tb):
            pass

    def fake_limit(scope: str, name: str, limit: int):
        # 注意：这是普通函数，不是 async def
        limit_calls["args"] = (scope, name, limit)
        return DummyCM()

    monkeypatch.setattr(global_limiter, "limit", fake_limit)

    # 4. 执行 Agent.run
    await agent.run("hello")

    # 5. 断言：Limiter 被调用 & 按 agent 维度限流
    assert limit_calls["args"] == ("agent", agent.name, 1)
    assert limit_calls["entered"] is True
    assert step_calls["called"] is True


@pytest.mark.asyncio
async def test_agent_timeout_zero_does_not_fallback(mock_llm, toolbox, memory, event_bus) -> None:
    """
    验证：timeout=0 时不应使用默认超时，而应原样透传到 Engine。
    """
    from gecko.core.output import AgentOutput

    agent = Agent(model=mock_llm, toolbox=toolbox, memory=memory, event_bus=event_bus)

    seen: dict[str, Any] = {}

    async def fake_step(messages, **kwargs):
        # 捕获透传下来的 timeout
        seen["timeout"] = kwargs.get("timeout")
        return AgentOutput(content="ok")

    agent.engine.step = AsyncMock(side_effect=fake_step)

    await agent.run("hello", timeout=0)

    # _resolve_timeout 会转成 float(0)
    assert seen["timeout"] == 0.0


# ----------------------------------------------------------------------
# 新增：覆盖 _normalize_messages / _resolve_timeout / _serialize_output
#       以及 run/stream/stream_events 的各类异常与分支逻辑
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_normalize_messages_various_inputs(
    mock_llm, toolbox, memory, event_bus
) -> None:
    """覆盖 _normalize_messages 的所有正常输入分支"""
    agent = Agent(model=mock_llm, toolbox=toolbox, memory=memory, event_bus=event_bus)

    # 1) 单条 Message
    msg = Message.user("hello")
    res = agent._normalize_messages(msg)
    assert res == [msg]
    assert res[0] is msg  # 同一个对象

    # 2) dict 含 role 字段
    res = agent._normalize_messages({"role": "user", "content": "hi"})
    assert len(res) == 1
    assert isinstance(res[0], Message)
    assert res[0].role == "user"
    assert res[0].content == "hi"

    # 3) List[dict]
    res = agent._normalize_messages(
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
    )
    assert [m.role for m in res] == ["user", "assistant"]

    # 4) Tuple[dict, ...]
    res = agent._normalize_messages(
        (
            {"role": "user", "content": "tuple_a"},
            {"role": "assistant", "content": "tuple_b"},
        )
    )
    assert len(res) == 2
    assert isinstance(res[0], Message)
    assert res[0].content == "tuple_a"

    # 5) Tuple[Message, ...]
    m1 = Message.user("m1")
    m2 = Message.user("m2")
    res = agent._normalize_messages((m1, m2))
    assert res == [m1, m2]  # 转成 list，但元素对象相同

    # 6) {"input": "..."} 简单 dict 形式
    res = agent._normalize_messages({"input": "simple_input"})
    assert len(res) == 1
    assert res[0].content == "simple_input"


@pytest.mark.asyncio
async def test_agent_normalize_messages_errors(mock_llm, toolbox, memory, event_bus) -> None:
    """覆盖 _normalize_messages 的异常分支：空序列 / 不支持类型 / 序列中非法元素"""
    agent = Agent(model=mock_llm, toolbox=toolbox, memory=memory, event_bus=event_bus)

    # 空 list
    with pytest.raises(AgentError):
        agent._normalize_messages([])

    # 空 tuple
    with pytest.raises(AgentError):
        agent._normalize_messages(())

    # 不支持的顶层类型（int）
    with pytest.raises(AgentError):
        agent._normalize_messages(123)  # type: ignore[arg-type]

    # 序列中包含不支持的元素类型（首元素不是 Message）
    with pytest.raises(AgentError):
        agent._normalize_messages([123])  # type: ignore[list-item]


@pytest.mark.asyncio
async def test_agent_resolve_timeout_invalid_and_negative(
    mock_llm, toolbox, memory, event_bus
) -> None:
    """覆盖 _resolve_timeout 的非法类型和负数分支"""
    agent = Agent(model=mock_llm, toolbox=toolbox, memory=memory, event_bus=event_bus)

    # 非数字类型
    with pytest.raises(ValueError):
        agent._resolve_timeout("not-a-number")  # type: ignore[arg-type]

    # 负数
    with pytest.raises(ValueError):
        agent._resolve_timeout(-1.0)


class _NoDumpObject:
    """用于测试 _serialize_output 的兜底分支"""

    def __str__(self) -> str:
        return "dummy-output"


@pytest.mark.asyncio
async def test_agent_serialize_output_fallback(mock_llm, toolbox, memory, event_bus) -> None:
    """覆盖 _serialize_output 中没有 model_dump 的兜底分支"""
    agent = Agent(model=mock_llm, toolbox=toolbox, memory=memory, event_bus=event_bus)

    obj = _NoDumpObject()
    payload = agent._serialize_output(obj)
    assert payload == {"content": "dummy-output"}


@pytest.mark.asyncio
async def test_agent_run_agenterror_emits_run_error_event(
    mock_llm, toolbox, memory, event_bus
) -> None:
    """engine.step 抛 AgentError 时，run 应重抛且发出 run_error 事件"""
    received = []

    async def handler(e):
        received.append(e)

    event_bus.subscribe("run_error", handler)

    agent = Agent(model=mock_llm, toolbox=toolbox, memory=memory, event_bus=event_bus)

    async def fake_step(messages, **kwargs):
        raise AgentError("bad-input")

    agent.engine.step = AsyncMock(side_effect=fake_step)

    with pytest.raises(AgentError):
        await agent.run("x")

    # 等待事件派发完成
    await asyncio.sleep(0.1)
    assert any(e.type == "run_error" for e in received)
    assert "bad-input" in (received[0].error or "")


@pytest.mark.asyncio
async def test_agent_run_exception_emits_run_error_event(
    mock_llm, toolbox, memory, event_bus
) -> None:
    """engine.step 抛普通 Exception 时，run 应重抛且发出 run_error 事件"""
    received = []

    async def handler(e):
        received.append(e)

    event_bus.subscribe("run_error", handler)

    agent = Agent(model=mock_llm, toolbox=toolbox, memory=memory, event_bus=event_bus)

    async def fake_step(messages, **kwargs):
        raise RuntimeError("boom")

    agent.engine.step = AsyncMock(side_effect=fake_step)

    with pytest.raises(RuntimeError):
        await agent.run("x")

    await asyncio.sleep(0.1)
    assert any(e.type == "run_error" for e in received)
    assert "boom" in (received[0].error or "")


@pytest.mark.asyncio
async def test_agent_stream_yields_tokens_and_events(mock_llm, toolbox, memory, event_bus) -> None:
    """覆盖 stream 的正常路径：token 事件 + stream_started/stream_completed 事件"""
    lifecycle = []

    async def handler(e):
        lifecycle.append(e)

    event_bus.subscribe("stream_started", handler)
    event_bus.subscribe("stream_completed", handler)

    agent = Agent(model=mock_llm, toolbox=toolbox, memory=memory, event_bus=event_bus)

    async def fake_step_stream(messages, **kwargs):
        async def gen():
            yield AgentStreamEvent(type="token", content="A")
            yield AgentStreamEvent(type="token", content="B")

        return gen()

    agent.engine.step_stream = AsyncMock(side_effect=fake_step_stream)

    chunks: list[str] = []
    async for ch in agent.stream("hello"):
        chunks.append(ch)

    assert chunks == ["A", "B"]

    await asyncio.sleep(0.1)
    types = [e.type for e in lifecycle]
    assert "stream_started" in types
    assert "stream_completed" in types


@pytest.mark.asyncio
async def test_agent_stream_handles_internal_error_event(
    mock_llm, toolbox, memory, event_bus
) -> None:
    """
    覆盖 stream 内部 event.type == 'error' 分支：
    - 不应向上抛异常
    - 只是记录日志与 span 异常
    """
    agent = Agent(model=mock_llm, toolbox=toolbox, memory=memory, event_bus=event_bus)

    async def fake_step_stream(messages, **kwargs):
        async def gen():
            # 只有 error 事件，没有 token 事件
            yield AgentStreamEvent(type="error", content="inner-error")

        return gen()

    agent.engine.step_stream = AsyncMock(side_effect=fake_step_stream)

    chunks: list[str] = []
    async for ch in agent.stream("hello"):
        chunks.append(ch)

    # 没有 token 输出，但也不会抛异常
    assert chunks == []


@pytest.mark.asyncio
async def test_agent_stream_outer_exception_emits_stream_error_event(
    mock_llm, toolbox, memory, event_bus
) -> None:
    """engine.step_stream 抛异常时，stream 应重抛并发出 stream_error 事件"""
    received = []

    async def handler(e):
        received.append(e)

    event_bus.subscribe("stream_error", handler)

    agent = Agent(model=mock_llm, toolbox=toolbox, memory=memory, event_bus=event_bus)

    async def fake_step_stream(messages, **kwargs):
        raise RuntimeError("stream-boom")

    agent.engine.step_stream = AsyncMock(side_effect=fake_step_stream)

    with pytest.raises(RuntimeError):
        async for _ in agent.stream("hello"):
            pass

    await asyncio.sleep(0.1)
    assert any(e.type == "stream_error" for e in received)
    assert "stream-boom" in (received[0].error or "")


@pytest.mark.asyncio
async def test_agent_stream_events_success_and_lifecycle(
    mock_llm, toolbox, memory, event_bus
) -> None:
    """覆盖 stream_events 正常路径：token/result 事件 + stream_events_* 生命周期事件"""
    lifecycle = []

    async def handler(e):
        lifecycle.append(e)

    event_bus.subscribe("stream_events_started", handler)
    event_bus.subscribe("stream_events_completed", handler)

    agent = Agent(model=mock_llm, toolbox=toolbox, memory=memory, event_bus=event_bus)

    async def fake_step_stream(messages, **kwargs):
        async def gen():
            yield AgentStreamEvent(type="token", content="A")
            yield AgentStreamEvent(type="result", content="done")

        return gen()

    agent.engine.step_stream = AsyncMock(side_effect=fake_step_stream)

    events: list[AgentStreamEvent] = []
    async for ev in agent.stream_events("hello"):
        events.append(ev)

    assert [e.type for e in events] == ["token", "result"]

    await asyncio.sleep(0.1)
    types = [e.type for e in lifecycle]
    assert "stream_events_started" in types
    assert "stream_events_completed" in types


@pytest.mark.asyncio
async def test_agent_stream_events_outer_exception_yields_error_event(
    mock_llm, toolbox, memory, event_bus
) -> None:
    """
    engine.step_stream 抛异常时，stream_events 应：
    1) 先 yield 一个 error 事件给调用方
    2) 随后在下一次迭代时抛出异常
    3) 同时发出 stream_events_error 生命周期事件
    """
    lifecycle = []

    async def handler(e):
        lifecycle.append(e)

    event_bus.subscribe("stream_events_error", handler)

    agent = Agent(model=mock_llm, toolbox=toolbox, memory=memory, event_bus=event_bus)

    async def fake_step_stream(messages, **kwargs):
        raise RuntimeError("events-boom")

    agent.engine.step_stream = AsyncMock(side_effect=fake_step_stream)

    events: list[AgentStreamEvent] = []
    agen = agent.stream_events("hello")

    try:
        async for ev in agen:
            events.append(ev)
    except RuntimeError:
        pass

    # 至少收到一个 error 事件
    assert len(events) == 1
    assert events[0].type == "error"
    assert "events-boom" in (events[0].content or "")

    await asyncio.sleep(0.1)
    assert any(e.type == "stream_events_error" for e in lifecycle)
