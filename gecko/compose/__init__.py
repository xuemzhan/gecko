# gecko/compose/__init__.py
# 导出核心 API，确保用户一键导入
from gecko.compose.workflow import Workflow
from gecko.compose.team import Team
from gecko.compose.nodes import step, Condition, Loop, Parallel

__all__ = ["Workflow", "Team", "step", "Condition", "Loop", "Parallel"]