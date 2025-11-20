# gecko/core/events.py
from __future__ import annotations
import asyncio
import logging
import time
import inspect
from typing import Any, Callable, Dict, List, Optional, Awaitable, Type
from pydantic import BaseModel, Field

logger = logging.getLogger("gecko.events")

class BaseEvent(BaseModel):
    type: str
    timestamp: float = Field(default_factory=time.time)
    data: Dict[str, Any] = {}
    error: Optional[str] = None

# 定义处理器类型
EventHandler = Callable[[BaseEvent], Awaitable[None]]

class EventBus:
    """
    企业级事件总线
    特性：
    1. 异步并发处理 (fire-and-forget 或 wait)
    2. 错误隔离 (Handler 崩溃不影响主流程)
    3. 中间件支持 (用于全链路追踪/日志)
    """
    def __init__(self):
        self._subscribers: Dict[str, List[EventHandler]] = {}
        self._middlewares: List[Callable[[BaseEvent], Awaitable[BaseEvent]]] = []

    def subscribe(self, event_type: str, handler: EventHandler):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        return self

    def add_middleware(self, mw: Callable):
        self._middlewares.append(mw)

    async def publish(self, event: BaseEvent, wait: bool = False):
        """
        发布事件
        :param wait: 是否等待所有处理器执行完毕 (True用于关键流程，False用于日志/监控)
        """
        # 1. 执行中间件
        try:
            for mw in self._middlewares:
                event = await mw(event)
                if not event: return # 中间件拦截
        except Exception as e:
            logger.error(f"Middleware error: {e}")

        # 2. 查找订阅者
        handlers = self._subscribers.get(event.type, [])
        if not handlers:
            # 支持通配符 "*" 订阅
            handlers = self._subscribers.get("*", [])

        if not handlers:
            return

        # 3. 调度执行
        tasks = [self._safe_execute(h, event) for h in handlers]
        
        if wait:
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Fire and forget: 放入后台任务，防止阻塞主流程
            for t in tasks:
                asyncio.create_task(t)

    async def _safe_execute(self, handler: EventHandler, event: BaseEvent):
        """安全执行包裹器，防止 Handler 异常搞崩系统"""
        try:
            await handler(event)
        except Exception as e:
            logger.exception(f"Event handler failed for {event.type}: {e}")
            # 可选：发布系统级错误事件
            # await self.publish(SystemErrorEvent(...)) 
            
    async def _safe_execute(self, handler: EventHandler, event: BaseEvent):
        """安全执行包裹器"""
        try:
            if inspect.iscoroutinefunction(handler):
                await handler(event)
            else:
                # 兼容同步函数：在线程池中运行，或者直接调用（如果耗时短）
                # 考虑到 EventBus 应该是高吞吐的，直接调用可能阻塞 Loop
                # 但为了兼容性，先直接调用
                handler(event)
        except Exception as e:
            logger.exception(f"Event handler failed for {event.type}: {e}")