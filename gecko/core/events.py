# gecko/core/events.py
"""
事件总线

提供异步事件发布/订阅机制，支持中间件和后台任务管理。

核心功能：
1. 强类型事件（基于 Pydantic）
2. 异步/同步订阅者支持
3. 中间件拦截与处理
4. 健壮的后台任务管理（任务追踪与优雅关闭）

优化日志：
1. 增加后台任务追踪集合，防止任务被 GC
2. 增加 shutdown 方法等待后台任务完成
3. 增加上下文管理器支持
4. 优化中间件错误处理
5. 修复对异步可调用对象（非函数）的支持
"""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Type, Union

from pydantic import BaseModel, Field

from gecko.core.logging import get_logger

logger = get_logger(__name__)


# ===== 事件模型 =====

class BaseEvent(BaseModel):
    """事件基类"""
    type: str
    timestamp: float = Field(default_factory=time.time)
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    
    model_config = {"arbitrary_types_allowed": True}


# EventHandler 可以是返回 None 的同步函数，或者返回 Awaitable 的函数
EventHandler = Callable[[BaseEvent], Union[Awaitable[None], None, Any]]
Middleware = Callable[[BaseEvent], Awaitable[Optional[BaseEvent]]]


class EventBus:
    """
    异步事件总线
    """
    
    def __init__(self):
        self._subscribers: Dict[str, List[EventHandler]] = {}
        self._middlewares: List[Middleware] = []
        self._background_tasks: Set[asyncio.Task] = set()
        self._running = True

    # --- 订阅管理 ---
    
    def subscribe(self, event_type: str, handler: EventHandler) -> "EventBus":
        """
        订阅事件
        支持通配符 "*" 订阅所有事件
        """
        if not callable(handler):
            raise TypeError(f"Event handler must be callable, got {type(handler)}")
        
        self._subscribers.setdefault(event_type, []).append(handler)
        logger.debug("Handler subscribed", event_type=event_type, handler=handler)
        return self

    def unsubscribe(self, event_type: str, handler: EventHandler) -> "EventBus":
        """取消订阅"""
        handlers = self._subscribers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)
            logger.debug("Handler unsubscribed", event_type=event_type, handler=handler)
        return self

    def add_middleware(self, middleware: Middleware) -> "EventBus":
        """
        添加中间件
        """
        self._middlewares.append(middleware)
        return self

    # --- 发布事件 ---
    async def publish(self, event: BaseEvent, wait: bool = False):
        """
        发布事件
        
        参数:
            event: 事件对象
            wait: 是否等待所有处理器执行完毕
        """
        if not self._running:
            logger.warning("EventBus is shutting down, event ignored", event_type=event.type)
            return

        # 保存原始事件类型用于日志（因为中间件可能返回 None）
        original_type = event.type

        # 1. 执行中间件
        try:
            for mw in self._middlewares:
                event = await mw(event)
                if event is None:
                    logger.debug("Event blocked by middleware", event_type=original_type)
                    return
        except Exception as e:
            logger.error("Middleware error", error=str(e), event_type=original_type)
            return

        # 2. 获取订阅者
        handlers = self._subscribers.get(event.type, []) + self._subscribers.get("*", [])
        if not handlers:
            return

        # 3. 执行处理（去重）
        unique_handlers = list(dict.fromkeys(handlers))
        
        # 创建执行协程
        tasks = [self._execute_handler(h, event) for h in unique_handlers]

        if wait:
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            for coro in tasks:
                task = asyncio.create_task(coro)
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
    
    async def _execute_handler(self, handler: EventHandler, event: BaseEvent):
        """
        执行单个处理器（包含错误捕获）
        
        采用统一的调用方式：先调用，再判断返回值是否为 Awaitable。
        这兼容了 async def, def, 以及 async __call__ 对象。
        """
        try:
            result = handler(event)
            if inspect.isawaitable(result):
                await result
        except Exception as e:
            logger.exception(
                "Event handler failed", 
                event_type=event.type, 
                handler=getattr(handler, "__name__", str(handler)),
                error=str(e)
            )

    # --- 生命周期 ---

    async def shutdown(self, wait: bool = True):
        """
        关闭事件总线
        """
        self._running = False
        
        if wait and self._background_tasks:
            count = len(self._background_tasks)
            if count > 0:
                logger.info("Waiting for background tasks to finish", count=count)
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
        logger.info("EventBus shutdown completed")

    async def __aenter__(self):
        self._running = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()


# ==== 常用事件类型 ====

class AgentRunEvent(BaseEvent):
    """Agent 运行过程事件"""
    pass


class WorkflowEvent(BaseEvent):
    """Workflow 运行过程事件"""
    pass