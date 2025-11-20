# tests/core/test_v03_events.py
import pytest
import asyncio
from gecko.core.events import EventBus, BaseEvent

# [修复] 重命名类，防止 pytest 将其视为测试套件
class MockEvent(BaseEvent):
    type: str = "test_event"

@pytest.mark.asyncio
async def test_event_bus_robustness():
    bus = EventBus()
    results = []

    async def middleware(event: BaseEvent):
        event.data["middleware_touched"] = True
        return event
    
    bus.add_middleware(middleware)

    async def async_handler(event: BaseEvent):
        results.append(f"async_{event.data.get('middleware_touched')}")

    def sync_handler(event: BaseEvent):
        results.append("sync_executed")

    async def error_handler(event: BaseEvent):
        raise ValueError("Boom!")

    bus.subscribe("test_event", async_handler)
    bus.subscribe("test_event", sync_handler)
    bus.subscribe("test_event", error_handler)

    # [修复] 使用 MockEvent
    await bus.publish(MockEvent(data={"raw": 1}), wait=True)

    assert "async_True" in results
    assert "sync_executed" in results
    assert len(results) == 2