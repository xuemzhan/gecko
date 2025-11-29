# gecko/__init__.py
from __future__ import annotations

"""
Gecko 顶层包入口

职责：
1. 暴露 v1.0 规划中的核心稳定 API（L1）
2. 提供统一的版本号 (__version__)
3. 保持 import gecko 即可访问常用类，而无需记复杂子模块路径
"""

from gecko.version import __version__  # ✅ 从单一版本源导入版本号

# ======= 核心对象导出（L1 稳定 API） =======

from gecko.core.agent import Agent
from gecko.core.builder import AgentBuilder
from gecko.core.message import Message, Role
from gecko.core.output import AgentOutput, TokenUsage
from gecko.core.memory import TokenMemory, SummaryTokenMemory
from gecko.core.structure import StructureEngine

from gecko.compose.workflow import Workflow
from gecko.compose.nodes import step, Next
from gecko.compose.team import Team

# 导入侧效：在 gecko.utils.cleanup 中注册 LiteLLM 的 atexit 清理
import gecko.utils.cleanup  # noqa: F401

__all__ = [
    # 版本
    "__version__",
    # Agent & Builder
    "Agent",
    "AgentBuilder",
    # 消息与角色
    "Message",
    "Role",
    # 输出与 Token 统计
    "AgentOutput",
    "TokenUsage",
    # 记忆模块
    "TokenMemory",
    "SummaryTokenMemory",
    # 结构化输出
    "StructureEngine",
    # 工作流与多智能体
    "Workflow",
    "step",
    "Next",
    "Team",
]
