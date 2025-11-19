# gecko/plugins/storage/__init__.py
# 快速开发插件本地注册 + 暴露工厂
from .factory import get_storage_by_url

# 手动注册核心快速开发插件（0 依赖启动）
from gecko.plugins.storage.sqlite import SQLiteSessionStorage
from gecko.plugins.storage.lancedb import LanceDBVectorStorage
from gecko.plugins.storage.factory import _register_local

_register_local("sqlite", SQLiteSessionStorage)
_register_local("lancedb", LanceDBVectorStorage)

__all__ = ["get_storage_by_url"]