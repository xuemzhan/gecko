# gecko/__init__.py
from __future__ import annotations

from gecko.core.agent import Agent
from gecko.core.builder import AgentBuilder
from gecko.core.message import Message

# 导入侧效：在 gecko.utils.cleanup 中注册 LiteLLM 的 atexit 清理
import gecko.utils.cleanup

__version__ = "0.3.0"

__all__ = ["Agent", "AgentBuilder", "Message"]