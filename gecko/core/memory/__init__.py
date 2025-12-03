# gecko/core/memory/__init__.py
"""
memory 包对外统一入口

拆分前：
    gecko/core/memory.py    （单文件）

拆分后：
    gecko/core/memory/      （包）
    ├─ __init__.py
    ├─ _executor.py
    ├─ base.py
    └─ summary.py

为保持向后兼容，对外仍然暴露同样的 API：

    from gecko.core.memory import TokenMemory, SummaryTokenMemory, shutdown_token_executor

因此，调用方无需修改任何 import 代码。
"""

from __future__ import annotations

from gecko.core.memory.base import TokenMemory
from gecko.core.memory.summary import SummaryTokenMemory
from gecko.core.memory.hybrid import HybridMemory
from gecko.core.memory._executor import shutdown_token_executor

__all__ = [
    "TokenMemory",
    "SummaryTokenMemory",
    "HybridMemory", 
    "shutdown_token_executor",
]
