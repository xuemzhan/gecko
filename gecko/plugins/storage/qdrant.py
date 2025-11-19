# gecko/plugins/storage/qdrant.py
from __future__ import annotations
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from typing import List, Dict
from gecko.plugins.storage.registry import register_storage
from gecko.plugins.storage.interfaces import VectorInterface

@register_storage("qdrant")
class QdrantStorage(VectorInterface):
    """
    高并发生产级 Vector 存储：Qdrant
    - URL 示例：qdrant://localhost:6333
    - 特点：分布式、高性能、支持 payload 过滤
    """
    def __init__(self, storage_url: str, collection_name: str = "gecko_default", embedding_dim: int = 1536, **kwargs):
        url = storage_url.removeprefix("qdrant://")
        self.client = QdrantClient(url=f"http://{url}")
        self.collection_name = collection_name
        
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
            )

    async def upsert(self, documents: List[Dict]):
        points = [
            PointStruct(
                id=d["id"],
                vector=d["embedding"],
                payload={"text": d["text"], "metadata": d.get("metadata", {})}
            )
            for d in documents
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    async def search(self, query_embedding: List[float], top_k: int = 5):
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        return [
            {"text": r.payload["text"], "metadata": r.payload["metadata"], "score": r.score}
            for r in results
        ]