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
    验证存储
    
    修复点：
    调整错误消息格式，包含 "does not implement StorageProtocol" 以匹配测试正则。
    """
    if not isinstance(storage, StorageProtocol):
        missing = get_missing_methods(storage, StorageProtocol)
        raise TypeError(
            f"Storage does not implement StorageProtocol. "
            f"Missing methods: {', '.join(missing) if missing else 'unknown'}"
        )