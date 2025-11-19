# gecko/core/events.py
from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict, Optional
from pydantic import BaseModel

class AppEvent(BaseModel):
    type: str
    data: Dict[str, Any] = {}
    error: Optional[str] = None

EventHandler = Callable[[AppEvent], Awaitable[None] | None]

class EventBus:
    def __init__(self):
        self.handlers: list[EventHandler] = []

    async def publish(self, event: AppEvent | dict):
        if isinstance(event, dict):
            event = AppEvent(**event)
        for handler in self.handlers:
            result = handler(event)
            if result is not None:
                await result

    def subscribe(self, handler: EventHandler):
        self.handlers.append(handler)