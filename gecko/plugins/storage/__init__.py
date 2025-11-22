# gecko/plugins/storage/__init__.py
"""
Gecko Storage 插件系统

提供统一的接口用于访问 Session (KV) 和 Vector (RAG) 存储。
"""
from gecko.plugins.storage.abc import AbstractStorage
from gecko.plugins.storage.factory import create_storage
from gecko.plugins.storage.interfaces import SessionInterface, VectorInterface
from gecko.plugins.storage.registry import register_storage

__all__ = [
    "AbstractStorage",
    "create_storage",
    "SessionInterface",
    "VectorInterface",
    "register_storage",
]