# gecko/core/events.py
from __future__ import annotations
import asyncio
from pydantic import BaseModel
from typing import Any, Callable, Dict, List, Optional

class BaseEvent(BaseModel):
    """
    Gecko 统一事件基类
    - 所有事件继承自 BaseModel，确保结构化、可序列化
    - 支持 OpenTelemetry / 日志 / UI 监控
    """
    type: str
    data: Dict[str, Any] = {}
    error: Optional[str] = None

class RunEvent(BaseEvent):
    """
    运行时事件标准模型（新增）
    - 用于 Workflow / Agent / Node 全链路追踪
    - type 示例: "workflow_started", "node_completed", "tool_called"
    """
    pass

class EventBus:
    """简易异步事件总线（Day1 缺失部分补全）"""
    def __init__(self):
        self.subscribers: List[Callable[[RunEvent], None]] = []

    async def publish(self, event: RunEvent):
        for subscriber in self.subscribers:
            # 支持 async subscriber
            if asyncio.iscoroutinefunction(subscriber):
                await subscriber(event)
            else:
                subscriber(event)

    def subscribe(self, callback: Callable[[RunEvent], None]):
        self.subscribers.append(callback)

# 兼容旧代码（Agent 可能还用 AppEvent）
AppEvent = BaseEvent  # 别名，未来废弃

# 示例事件（可选，方便使用）
WorkflowStarted = RunEvent(type="workflow_started")
WorkflowCompleted = RunEvent(type="workflow_completed")
NodeStarted = RunEvent(type="node_started")
NodeCompleted = RunEvent(type="node_completed")
NodeError = RunEvent(type="node_error")