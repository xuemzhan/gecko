# gecko/plugins/storage/__init__.py
from gecko.plugins.storage.factory import get_storage_by_url
from gecko.plugins.storage.interfaces import SessionInterface, VectorInterface

# 显式导入以触发 @register_storage 装饰器
import gecko.plugins.storage.sqlite
import gecko.plugins.storage.redis
try:
    import gecko.plugins.storage.lancedb
except ImportError:
    pass # 允许用户不安装 lancedb

# Postgres 依赖较重，通常作为 extra 安装，这里尝试可选导入
try:
    import gecko.plugins.storage.postgres_pgvector
except ImportError:
    pass

__all__ = ["get_storage_by_url", "SessionInterface", "VectorInterface"]