# tests/core/test_events.py
import pytest
import asyncio
from gecko.core.events import EventBus, BaseEvent

# [修复] 重命名 TestEvent -> MockEvent
# 避免 Pytest 误将其识别为测试类 (PytestCollectionWarning)
class MockEvent(BaseEvent):
    type: str = "test_event"

@pytest.mark.asyncio
async def test_event_bus_subscribe_publish(event_bus):
    received = []
    
    async def handler(event: BaseEvent):
        received.append(event)
        
    event_bus.subscribe("test_event", handler)
    # 使用 MockEvent
    await event_bus.publish(MockEvent(), wait=True)
    
    assert len(received) == 1
    assert received[0].type == "test_event"

@pytest.mark.asyncio
async def test_event_bus_sync_and_async_handlers(event_bus):
    """测试同时支持同步和异步处理器 (验证 _safe_execute 修复)"""
    results = []
    
    async def async_handler(event):
        await asyncio.sleep(0.01)
        results.append("async")
        
    def sync_handler(event):
        results.append("sync")
        
    event_bus.subscribe("test_event", async_handler)
    event_bus.subscribe("test_event", sync_handler)
    
    # 使用 MockEvent
    await event_bus.publish(MockEvent(), wait=True)
    
    assert "async" in results
    assert "sync" in results
    assert len(results) == 2

@pytest.mark.asyncio
async def test_event_bus_error_isolation(event_bus):
    """测试错误隔离：一个处理器崩溃不应影响其他处理器"""
    results = []
    
    async def bad_handler(event):
        raise ValueError("Boom!")
        
    async def good_handler(event):
        results.append("ok")
        
    event_bus.subscribe("test_event", bad_handler)
    event_bus.subscribe("test_event", good_handler)
    
    # 不应抛出异常
    # 使用 MockEvent
    await event_bus.publish(MockEvent(), wait=True)
    
    assert results == ["ok"]

@pytest.mark.asyncio
async def test_middleware(event_bus):
    """测试中间件拦截与修改"""
    
    async def middleware(event):
        if event.data.get("block"):
            return None # 拦截
        event.data["processed"] = True
        return event
        
    event_bus.add_middleware(middleware)
    
    received = []
    event_bus.subscribe("test_event", lambda e: received.append(e))
    
    # Case 1: Pass through
    # 使用 MockEvent
    await event_bus.publish(MockEvent(data={"block": False}), wait=True)
    assert len(received) == 1
    assert received[0].data["processed"] is True
    
    # Case 2: Blocked
    # 使用 MockEvent
    await event_bus.publish(MockEvent(data={"block": True}), wait=True)
    assert len(received) == 1 # 数量不变