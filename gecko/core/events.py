# gecko/core/events.py
from __future__ import annotations
import asyncio
from pydantic import BaseModel
from typing import Any, Callable, Dict, List, Optional

class BaseEvent(BaseModel):
    type: str
    data: Dict[str, Any] = {}
    error: Optional[str] = None

class RunEvent(BaseEvent):
    """运行时事件"""
    pass

class AppEvent(BaseEvent):
    """应用层事件（兼容旧代码）"""
    pass

class EventBus:
    def __init__(self):
        self.subscribers: List[Callable[[BaseEvent], None]] = []

    async def publish(self, event: BaseEvent):
        # 简单的广播逻辑
        for subscriber in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(event)
                else:
                    subscriber(event)
            except Exception as e:
                print(f"Event handler error: {e}")

    def subscribe(self, callback: Callable[[BaseEvent], None]):
        self.subscribers.append(callback)