# gecko/plugins/storage/backends/redis.py
"""
Redis 存储后端

基于 redis-py (asyncio) 实现的高性能会话存储。
由于 Redis 客户端原生支持 asyncio，因此不需要使用 ThreadOffloadMixin。

核心特性：
1. **原生异步**：直接利用 asyncio Event Loop，性能极高。
2. **TTL 管理**：支持会话自动过期。

更新日志：
- [Robustness] 初始化失败时自动清理资源。
- [Robustness] 所有操作统一抛出 StorageError，屏蔽底层 redis 异常。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import redis.asyncio as redis
except ImportError:
    redis = None  # type: ignore

from gecko.core.exceptions import StorageError
from gecko.core.logging import get_logger
from gecko.plugins.storage.abc import AbstractStorage
from gecko.plugins.storage.interfaces import SessionInterface
from gecko.plugins.storage.mixins import JSONSerializerMixin
from gecko.plugins.storage.registry import register_storage
from gecko.plugins.storage.utils import parse_storage_url
from gecko.config import get_settings

logger = get_logger(__name__)


@register_storage("redis")
class RedisStorage(
    AbstractStorage,
    SessionInterface,
    JSONSerializerMixin
):
    def __init__(self, url: str, **kwargs):
        super().__init__(url, **kwargs)
        if redis is None:
            raise ImportError(
                "Redis client not installed. Please install: pip install redis"
            )
        
        scheme, path, params = parse_storage_url(url)
        
        try:
            self.ttl = int(params.get("ttl", 3600 * 24 * 7))
        except ValueError:
            self.ttl = 3600 * 24 * 7
            
        self.prefix = kwargs.get("prefix", "gecko:session:")
        self.client: Optional[redis.Redis] = None # type: ignore

    async def initialize(self) -> None:
        if self.is_initialized:
            return

        logger.info("Connecting to Redis", url=self.url)

        settings = get_settings()
        
        try:
            # [优化] 配置 Redis 连接池
            # redis-py 的 from_url 支持 max_connections 参数
            max_connections = self.config.get("max_connections", settings.storage_pool_size * 2)
            
            self.client = redis.from_url( # type: ignore
                self.url,
                decode_responses=True,
                encoding="utf-8",
                max_connections=max_connections, # [新增]
                socket_timeout=5.0, # [新增] 防止网络卡死
                socket_connect_timeout=5.0
            )
            if self.client:
                await self.client.ping()
            self._is_initialized = True
            logger.debug("Redis connected successfully")
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            # 初始化失败时尝试清理
            await self.shutdown()
            raise StorageError(f"Failed to connect to Redis: {e}") from e

    async def shutdown(self) -> None:
        if self.client:
            try:
                await self.client.aclose()
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")
            finally:
                self.client = None
        self._is_initialized = False

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not self.client:
            raise StorageError("RedisStorage not initialized")
            
        key = f"{self.prefix}{session_id}"
        try:
            data = await self.client.get(key)
            return self._deserialize(data)
        except Exception as e:
            logger.error("Redis get failed", session_id=session_id, error=str(e))
            raise StorageError(f"Redis get failed: {e}") from e

    async def set(self, session_id: str, state: Dict[str, Any]) -> None:
        if not self.client:
            raise StorageError("RedisStorage not initialized")
            
        key = f"{self.prefix}{session_id}"
        json_str = self._serialize(state)
        
        try:
            await self.client.setex(key, self.ttl, json_str)
        except Exception as e:
            logger.error("Redis set failed", session_id=session_id, error=str(e))
            raise StorageError(f"Redis set failed: {e}") from e

    async def delete(self, session_id: str) -> None:
        if not self.client:
            raise StorageError("RedisStorage not initialized")
            
        key = f"{self.prefix}{session_id}"
        try:
            await self.client.delete(key)
        except Exception as e:
            logger.error("Redis delete failed", session_id=session_id, error=str(e))
            raise StorageError(f"Redis delete failed: {e}") from e