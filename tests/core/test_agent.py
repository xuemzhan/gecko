# tests/core/test_agent.py
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from gecko.core.agent import Agent, AgentRunEvent
from gecko.core.events.bus import EventBus
from gecko.core.message import Message
from gecko.core.engine.react import ReActEngine

@pytest.mark.asyncio
async def test_agent_run_input_formats(mock_llm, toolbox, memory, event_bus):
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
    assert call_args is msgs # Should be same object ID
    
    # 3. Dict input
    await agent.run({"input": "dict_test"})
    call_args = agent.engine.step.call_args[0][0]
    assert call_args[0].content == "dict_test"

@pytest.mark.asyncio
async def test_agent_events(mock_llm, toolbox, memory, event_bus):
    """测试 Agent 生命周期事件"""
    received_events = []
    
    # 定义回调函数
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
    
    # 【关键修复】增加 sleep，等待 EventBus 的后台任务执行完毕
    await asyncio.sleep(0.1)
    
    # 检查是否收到事件
    assert len(received_events) >= 2
    
    # 可选：验证事件类型
    event_types = [e.type for e in received_events]
    assert "run_started" in event_types
    assert "run_completed" in event_types

@pytest.mark.asyncio
async def test_agent_event_bus_wiring(mock_llm, toolbox, memory):
    """
    [New] 验证 Agent 的 EventBus 是否正确连接到了 Engine
    这是修复 ReActEngine 缺少 event_bus 属性的关键验证
    """
    bus = EventBus()
    agent = Agent(
        model=mock_llm, 
        toolbox=toolbox, 
        memory=memory, 
        event_bus=bus
    )
    
    # 验证 Engine 拥有 event_bus 属性且是同一个实例
    assert hasattr(agent.engine, "event_bus")
    assert agent.engine.event_bus is bus
    
    # 验证如果不传，Agent 会自动创建
    agent_default = Agent(model=mock_llm, toolbox=toolbox, memory=memory)
    assert agent_default.engine.event_bus is not None
    assert isinstance(agent_default.engine.event_bus, EventBus)