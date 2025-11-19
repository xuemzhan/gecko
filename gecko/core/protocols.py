# gecko/core/protocols.py
from __future__ import annotations

from typing import Protocol, runtime_checkable, Any,List, Dict

@runtime_checkable
class ModelProtocol(Protocol):
    """所有模型插件必须实现的异步协议"""
    async def acompletion(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        ...