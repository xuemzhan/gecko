# gecko/core/protocols/storage.py
"""存储相关协议"""
from __future__ import annotations
from typing import Any, Dict, Optional, Protocol, runtime_checkable
from .base import get_missing_methods

@runtime_checkable
class StorageProtocol(Protocol):
    """存储后端协议"""
    async def get(self, key: str) -> Optional[Dict[str, Any]]: ...
    async def set(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> None: ...
    async def delete(self, key: str) -> bool: ...

def validate_storage(storage: Any) -> None:
    """
    验证存储是否满足 StorageProtocol 所需的方法/属性（鸭子类型检查）
    """
    missing = get_missing_methods(storage, StorageProtocol)
    if missing:
        raise TypeError(
            "Storage does not implement StorageProtocol. "
            f"Missing methods: {', '.join(missing)}"
        )