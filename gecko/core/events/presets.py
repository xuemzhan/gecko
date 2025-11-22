"""预置系统事件"""
from gecko.core.events.types import BaseEvent

class AgentRunEvent(BaseEvent):
    """Agent 运行事件"""
    pass

class WorkflowEvent(BaseEvent):
    """Workflow 运行事件"""
    pass

class SessionEvent(BaseEvent):
    """会话变更事件"""
    pass