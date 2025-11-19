# gecko/plugins/storage/redis.py
from __future__ import annotations
import json
from typing import Dict, Any

try:
    import redis.asyncio as redis
except ImportError:
    raise ImportError("请安装 redis 客户端: pip install redis")

from gecko.plugins.storage.registry import register_storage
from gecko.plugins.storage.interfaces import SessionInterface

@register_storage("redis")
class RedisStorage(SessionInterface):
    """
    基于 Redis 的高性能 Session 存储
    URL 示例: redis://localhost:6379/0
    """
    def __init__(self, storage_url: str, ttl: int = 3600 * 24 * 7, **kwargs):
        """
        :param storage_url: Redis 连接 URL
        :param ttl: 数据过期时间（秒），默认 7 天
        """
        # redis-py 可以直接解析 redis:// URL
        self.client = redis.from_url(storage_url, decode_responses=True)
        self.ttl = ttl
        self.prefix = "gecko:session:"

    async def get(self, session_id: str) -> Dict[str, Any] | None:
        key = f"{self.prefix}{session_id}"
        data = await self.client.get(key)
        if data:
            return json.loads(data)
        return None

    async def set(self, session_id: str, state: Dict[str, Any]) -> None:
        key = f"{self.prefix}{session_id}"
        json_str = json.dumps(state, ensure_ascii=False)
        # 设置值并重置过期时间
        await self.client.set(key, json_str, ex=self.ttl)

    async def delete(self, session_id: str) -> None:
        key = f"{self.prefix}{session_id}"
        await self.client.delete(key)
        
    async def close(self):
        await self.client.aclose()