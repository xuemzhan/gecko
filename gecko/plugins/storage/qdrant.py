# gecko/plugins/storage/qdrant.py
from __future__ import annotations
from typing import List, Dict
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
except ImportError:
    raise ImportError("请安装 qdrant-client: pip install qdrant-client")

from gecko.plugins.storage.registry import register_storage
from gecko.plugins.storage.interfaces import VectorInterface

@register_storage("qdrant")
class QdrantStorage(VectorInterface):
    """
    Qdrant 向量存储实现
    URL 示例: qdrant://localhost:6333
    """
    def __init__(self, storage_url: str, collection_name: str = "gecko_default", embedding_dim: int = 1536, **kwargs):
        url = storage_url.removeprefix("qdrant://")
        # 支持内存模式
        if url == ":memory:":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(url=f"http://{url}")
            
        self.collection_name = collection_name
        
        # 检查并创建集合
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
            )

    async def upsert(self, documents: List[Dict]):
        if not documents:
            return
        points = [
            PointStruct(
                id=d["id"],
                vector=d["embedding"],
                payload={"text": d["text"], "metadata": d.get("metadata", {})}
            )
            for d in documents
        ]
        # Qdrant 客户端方法通常是同步的，但在 gecko 协议中我们包装为异步
        self.client.upsert(collection_name=self.collection_name, points=points)

    async def search(self, query_embedding: List[float], top_k: int = 5):
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        return [
            {
                "text": r.payload["text"], # type: ignore
                "metadata": r.payload["metadata"], # type: ignore
                "score": r.score
            }
            for r in results
        ]