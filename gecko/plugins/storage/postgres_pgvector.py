# gecko/plugins/storage/postgres_pgvector.py
from __future__ import annotations
import asyncpg
from typing import List, Dict, Any
from gecko.plugins.storage.registry import register_storage
from gecko.plugins.storage.interfaces import SessionInterface, VectorInterface

@register_storage("postgres+pgvector")
class PostgresPgVectorStorage(SessionInterface, VectorInterface):
    """
    企业级首选存储：PostgreSQL + pgvector
    - URL 示例：postgres+pgvector://user:pass@localhost:5432/db
    - 特点：ACID、一库两用、成熟生态
    - 需执行 SQL 创建表与扩展
    """
    def __init__(self, storage_url: str, collection_name: str = "gecko_default", **kwargs):
        self.dsn = storage_url.removeprefix("postgres+pgvector://")
        self.collection_name = collection_name

    async def _connect(self):
        return await asyncpg.connect(self.dsn)

    # Session 接口
    async def get(self, session_id: str) -> Dict[str, Any] | None:
        conn = await self._connect()
        try:
            row = await conn.fetchrow("SELECT state FROM gecko_sessions WHERE id = $1", session_id)
            return row["state"] if row else None
        finally:
            await conn.close()

    async def set(self, session_id: str, state: Dict[str, Any]):
        conn = await self._connect()
        try:
            await conn.execute(
                "INSERT INTO gecko_sessions (id, state) VALUES ($1, $2) ON CONFLICT (id) DO UPDATE SET state = $2",
                session_id, state
            )
        finally:
            await conn.close()

    async def delete(self, session_id: str):
        conn = await self._connect()
        try:
            await conn.execute("DELETE FROM gecko_sessions WHERE id = $1", session_id)
        finally:
            await conn.close()

    # Vector 接口
    async def upsert(self, documents: List[Dict[str, Any]]):
        conn = await self._connect()
        try:
            await conn.executemany(
                """INSERT INTO gecko_vectors (id, collection, embedding, text, metadata)
                   VALUES ($1, $2, $3, $4, $5)
                   ON CONFLICT (id) DO UPDATE SET embedding = $3, text = $4, metadata = $5""",
                [(d["id"], self.collection_name, d["embedding"], d["text"], d.get("metadata", {})) for d in documents]
            )
        finally:
            await conn.close()

    async def search(self, query_embedding: List[float], top_k: int = 5):
        conn = await self._connect()
        try:
            rows = await conn.fetch(
                """SELECT text, metadata, 1 - (embedding <=> $1) AS score
                   FROM gecko_vectors
                   WHERE collection = $2
                   ORDER BY embedding <=> $1 LIMIT $3""",
                query_embedding, self.collection_name, top_k
            )
            return [{"text": r["text"], "metadata": r["metadata"], "score": r["score"]} for r in rows]
        finally:
            await conn.close()