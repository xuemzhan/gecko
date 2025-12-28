# gecko/compose/__init__.py
"""
Gecko Compose 模块

提供多智能体编排能力：
- Workflow: DAG 工作流引擎
- Team: 并行多智能体执行
- step: 节点装饰器
- ensure_awaitable: 同步/异步统一调用工具
- Next: 控制流指令
"""
from gecko.compose.workflow.engine import Workflow
from gecko.compose.workflow.models import (
    WorkflowContext,
    CheckpointStrategy,
    NodeStatus,
    NodeExecution
)
# 导出新的 State 类，方便高级用户使用
from gecko.compose.workflow.state import COWDict 
from gecko.core.exceptions import WorkflowError, WorkflowCycleError

__all__ = [
    "Workflow",
    "WorkflowContext",
    "CheckpointStrategy",
    "NodeStatus",
    "NodeExecution",
    "WorkflowError",
    "WorkflowCycleError",
    "COWDict"
]