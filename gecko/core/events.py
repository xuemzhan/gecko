# gecko/core/events.py  
"""  
事件总线（升级版）  
  
特性：  
1. BaseEvent 使用 Pydantic，所有字段默认安全可序列化  
2. Middleware 可修改事件；返回 None 表示拦截  
3. 订阅者支持异步/同步函数；错误被捕获并记录  
4. wait=False 时后台任务异常依然可追踪  
5. 支持 unsubscribe，便于测试或动态注销  
"""  
  
from __future__ import annotations  
  
import asyncio  
import inspect  
import time  
from typing import Any, Awaitable, Callable, Dict, List, Optional  
  
from pydantic import BaseModel, Field  
  
from gecko.core.logging import get_logger  
  
logger = get_logger(__name__)  
  
# ===== 事件模型 =====  
class BaseEvent(BaseModel):  
    type: str  
    timestamp: float = Field(default_factory=time.time)  
    data: Dict[str, Any] = Field(default_factory=dict)  
    error: Optional[str] = None  
  
  
EventHandler = Callable[[BaseEvent], Awaitable[None]]  
Middleware = Callable[[BaseEvent], Awaitable[BaseEvent | None]]  
  
  
class EventBus:  
    def __init__(self):  
        self._subscribers: Dict[str, List[EventHandler]] = {}  
        self._middlewares: List[Middleware] = []  
  
    # --- 订阅管理 ---  
    def subscribe(self, event_type: str, handler: EventHandler):  
        if not callable(handler):  
            raise TypeError("Event handler 必须是可调用的")  
        self._subscribers.setdefault(event_type, []).append(handler)  
        return self  
  
    def unsubscribe(self, event_type: str, handler: EventHandler):  
        handlers = self._subscribers.get(event_type, [])  
        if handler in handlers:  
            handlers.remove(handler)  
        return self  
  
    def add_middleware(self, middleware: Middleware):  
        self._middlewares.append(middleware)  
        return self  
  
    # --- 发布事件 ---  
    async def publish(self, event: BaseEvent, wait: bool = False):  
        # 1. 依次执行中间件，可修改事件或拦截  
        try:  
            for mw in self._middlewares:  
                new_event = await mw(event)  
                if new_event is None:  
                    logger.debug("Event blocked by middleware", event_type=event.type)  
                    return  
                event = new_event  
        except Exception as e:  
            logger.error("Middleware error", error=str(e))  
            return  # 中间件异常直接终止，避免传播不一致事件  
  
        # 2. 查找订阅者（支持通配符 "*")  
        handlers = self._subscribers.get(event.type) or self._subscribers.get("*", [])  
        if not handlers:  
            return  
  
        tasks = [self._safe_execute(handler, event) for handler in handlers]  
  
        if wait:  
            await asyncio.gather(*tasks, return_exceptions=True)  
        else:  
            for coro in tasks:  
                asyncio.create_task(self._log_task_exception(coro))  
  
    async def _safe_execute(self, handler: EventHandler, event: BaseEvent):  
        try:  
            if inspect.iscoroutinefunction(handler):  
                await handler(event)  
            else:  
                handler(event)  
        except Exception as e:  
            logger.exception("Event handler failed", event_type=event.type, error=str(e))  
  
    async def _log_task_exception(self, coro):  
        try:  
            await coro  
        except Exception as e:  
            logger.exception("Background event handler failed", error=str(e))  
  
  
# ==== 示例：常见事件类型 ====  
  
  
class AgentRunEvent(BaseEvent):  
    """Agent 运行过程事件"""  
    pass  
  
  
class WorkflowEvent(BaseEvent):  
    """Workflow 运行过程事件"""  
    pass  
