# gecko/plugins/storage/redis.py
from __future__ import annotations
import redis.asyncio as redis
import json
from typing import Dict, Any
from gecko.plugins.storage.registry import register_storage
from gecko.plugins.storage.interfaces import SessionInterface

@register_storage("redis")
class RedisStorage(SessionInterface):
    """
    超低延迟 Session 存储：Redis
    - URL 示例：redis://localhost:6379/0
    - 特点：内存级速度、自动过期
    """
    def __init__(self, storage_url: str, **kwargs):
        url = storage_url.removeprefix("redis://")
        self.client = redis.from_url(f"redis://{url}")

    async def get(self, session_id: str) -> Dict[str, Any] | None:
        data = await self.client.get(f"gecko:session:{session_id}")
        return json.loads(data) if data else None

    async def set(self, session_id: str, state: Dict[str, Any]):
        await self.client.set(f"gecko:session:{session_id}", json.dumps(state), ex=86400*30)  # 30天过期

    async def delete(self, session_id: str):
        await self.client.delete(f"gecko:session:{session_id}")