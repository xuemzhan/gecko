# gecko/compose/workflow/__init__.py
"""
Workflow 包入口

通过在此处重新导出子模块的类，保持对旧版单文件 workflow.py 的 API 兼容性。
外部调用者无需感知内部拆分。
"""

# 1. 核心引擎
from gecko.compose.workflow.engine import Workflow

# 2. 数据模型 (Context, Enums, Trace)
from gecko.compose.workflow.models import (
    WorkflowContext,
    CheckpointStrategy,
    NodeStatus,
    NodeExecution
)

# 3. 异常 (如果之前是在 workflow.py 定义的，现在建议引用 core.exceptions)
# 但如果为了兼容性保留了别名，也可以在这里处理
from gecko.core.exceptions import WorkflowError, WorkflowCycleError

__all__ = [
    "Workflow",
    "WorkflowContext",
    "CheckpointStrategy",
    "NodeStatus",
    "NodeExecution",
    "WorkflowError",
    "WorkflowCycleError"
]