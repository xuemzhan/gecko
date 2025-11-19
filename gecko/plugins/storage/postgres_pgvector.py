# gecko/plugins/storage/postgres_pgvector.py
from __future__ import annotations
import json
from typing import List, Dict, Any

try:
    import asyncpg
except ImportError:
    raise ImportError("请安装 asyncpg: pip install asyncpg")

from gecko.plugins.storage.registry import register_storage
from gecko.plugins.storage.interfaces import SessionInterface, VectorInterface

@register_storage("postgres+pgvector")
class PostgresPgVectorStorage(SessionInterface, VectorInterface):
    """
    基于 PostgreSQL 的企业级存储
    同时支持 Session (JSONB) 和 Vector (pgvector)
    URL 示例: postgres+pgvector://user:pass@localhost:5432/dbname
    """
    def __init__(self, storage_url: str, collection_name: str = "gecko_default", **kwargs):
        # 移除自定义协议头，还原为标准 postgres URL
        self.dsn = storage_url.replace("postgres+pgvector://", "postgresql://")
        self.collection_name = collection_name
        self._pool = None
        
    async def _get_pool(self):
        if not self._pool:
            self._pool = await asyncpg.create_pool(self.dsn)
            await self._init_schema()
        return self._pool

    async def _init_schema(self):
        """初始化数据库 Schema"""
        async with self._pool.acquire() as conn:
            # 1. 启用 pgvector 扩展
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # 2. 创建 Session 表
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS gecko_sessions (
                    session_id TEXT PRIMARY KEY,
                    state JSONB,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 3. 创建 Vector 表
            # 注意：这里假设 embedding 维度为 1536 (OpenAI)，生产环境应动态处理
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS gecko_vectors (
                    id TEXT PRIMARY KEY,
                    collection TEXT,
                    embedding vector(1536),
                    text TEXT,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # 创建向量索引 (IVFFlat)
            # await conn.execute("CREATE INDEX ON gecko_vectors USING ivfflat (embedding vector_cosine_ops)")

    # --- Session Interface ---
    
    async def get(self, session_id: str) -> Dict[str, Any] | None:
        pool = await self._get_pool()
        row = await pool.fetchrow(
            "SELECT state FROM gecko_sessions WHERE session_id = $1", 
            session_id
        )
        return json.loads(row['state']) if row else None

    async def set(self, session_id: str, state: Dict[str, Any]) -> None:
        pool = await self._get_pool()
        state_json = json.dumps(state)
        await pool.execute("""
            INSERT INTO gecko_sessions (session_id, state) 
            VALUES ($1, $2)
            ON CONFLICT (session_id) 
            DO UPDATE SET state = $2, updated_at = CURRENT_TIMESTAMP
        """, session_id, state_json)

    async def delete(self, session_id: str) -> None:
        pool = await self._get_pool()
        await pool.execute("DELETE FROM gecko_sessions WHERE session_id = $1", session_id)

    # --- Vector Interface ---

    async def upsert(self, documents: List[Dict[str, Any]]) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                for doc in documents:
                    await conn.execute("""
                        INSERT INTO gecko_vectors (id, collection, embedding, text, metadata)
                        VALUES ($1, $2, $3, $4, $5)
                        ON CONFLICT (id) DO UPDATE 
                        SET embedding = $3, text = $4, metadata = $5
                    """, 
                    doc["id"], 
                    self.collection_name, 
                    doc["embedding"], 
                    doc["text"], 
                    json.dumps(doc.get("metadata", {}))
                    )

    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        pool = await self._get_pool()
        # 使用 <=> 操作符计算余弦距离 (Cosine Distance)
        # 相似度 = 1 - 距离
        rows = await pool.fetch("""
            SELECT text, metadata, 1 - (embedding <=> $1) as score
            FROM gecko_vectors
            WHERE collection = $2
            ORDER BY embedding <=> $1
            LIMIT $3
        """, query_embedding, self.collection_name, top_k)
        
        return [
            {
                "text": r["text"],
                "metadata": json.loads(r["metadata"]),
                "score": r["score"]
            } for r in rows
        ]