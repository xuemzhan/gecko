# gecko/plugins/storage/backends/redis.py
"""
Redis 存储后端

基于 redis-py (asyncio) 实现的高性能会话存储。
由于 Redis 客户端原生支持 asyncio，因此不需要使用 ThreadOffloadMixin。

核心特性：
1. **原生异步**：直接利用 asyncio Event Loop，性能极高。
2. **TTL 管理**：支持会话自动过期。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import redis.asyncio as redis
except ImportError:
    redis = None  # type: ignore

from gecko.core.logging import get_logger
from gecko.plugins.storage.abc import AbstractStorage
from gecko.plugins.storage.interfaces import SessionInterface
from gecko.plugins.storage.mixins import JSONSerializerMixin
from gecko.plugins.storage.registry import register_storage
from gecko.plugins.storage.utils import parse_storage_url

logger = get_logger(__name__)


@register_storage("redis")
class RedisStorage(
    AbstractStorage,
    SessionInterface,
    JSONSerializerMixin
):
    """
    Redis 会话存储
    
    URL 示例: 
        redis://localhost:6379/0?ttl=3600
    """
    
    def __init__(self, url: str, **kwargs):
        super().__init__(url, **kwargs)
        if redis is None:
            raise ImportError(
                "Redis client not installed. Please install: pip install redis"
            )
        
        scheme, path, params = parse_storage_url(url)
        
        # 解析 TTL 参数 (默认 7 天)
        try:
            self.ttl = int(params.get("ttl", 3600 * 24 * 7))
        except ValueError:
            self.ttl = 3600 * 24 * 7
            
        self.prefix = kwargs.get("prefix", "gecko:session:")
        
        # 客户端引用 (延迟初始化)
        self.client: Optional[redis.Redis] = None # type: ignore

    async def initialize(self) -> None:
        """异步初始化：建立连接并测试连通性"""
        if self.is_initialized:
            return

        logger.info("Connecting to Redis", url=self.url)
        
        # redis-py 的 from_url 会复用连接池
        # decode_responses=True 让 Redis 直接返回 str 而不是 bytes
        self.client = redis.from_url( # type: ignore
            self.url,
            decode_responses=True,
            encoding="utf-8"
        )
        
        try:
            # Ping 测试确保连接可用
            if self.client:
                await self.client.ping()
            self._is_initialized = True
            logger.debug("Redis connected successfully")
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            # 连接失败时确保清理资源
            await self.shutdown()
            raise

    async def shutdown(self) -> None:
        """关闭连接"""
        if self.client:
            await self.client.aclose()
            self.client = None
        self._is_initialized = False

    # ==================== SessionInterface 实现 ====================

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not self.client:
            raise RuntimeError("RedisStorage not initialized")
            
        key = f"{self.prefix}{session_id}"
        try:
            data = await self.client.get(key)
            return self._deserialize(data)
        except Exception as e:
            logger.error("Redis get failed", session_id=session_id, error=str(e))
            raise

    async def set(self, session_id: str, state: Dict[str, Any]) -> None:
        if not self.client:
            raise RuntimeError("RedisStorage not initialized")
            
        key = f"{self.prefix}{session_id}"
        json_str = self._serialize(state)
        
        try:
            # setex: 设置值并指定过期时间 (原子操作)
            await self.client.setex(key, self.ttl, json_str)
        except Exception as e:
            logger.error("Redis set failed", session_id=session_id, error=str(e))
            raise

    async def delete(self, session_id: str) -> None:
        if not self.client:
            raise RuntimeError("RedisStorage not initialized")
            
        key = f"{self.prefix}{session_id}"
        try:
            await self.client.delete(key)
        except Exception as e:
            logger.error("Redis delete failed", session_id=session_id, error=str(e))
            raise