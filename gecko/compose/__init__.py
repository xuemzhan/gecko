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
from gecko.compose.workflow import Workflow
from gecko.compose.team import Team
from gecko.compose.nodes import step, ensure_awaitable, Next

__all__ = ["Workflow", "Team", "step", "ensure_awaitable", "Next"]