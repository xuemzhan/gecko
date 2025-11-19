# gecko/plugins/storage/chroma.py
from __future__ import annotations
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from typing import List, Dict, Any
from gecko.plugins.storage.registry import register_storage
from gecko.plugins.storage.interfaces import SessionInterface, VectorInterface

@register_storage("chroma")
class ChromaStorage(SessionInterface, VectorInterface):
    """
    生产级本地存储插件：Chroma
    - URL 示例：chroma://./chroma_db
    - 特点：持久化目录、自动嵌入（SentenceTransformer）、支持 Session + Vector
    - 适用：本地开发到中小型生产（<100万向量）
    """
    def __init__(self, storage_url: str, collection_name: str = "gecko_default", **kwargs):
        db_path = storage_url.removeprefix("chroma://")
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        self.vector_collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )
        self.session_collection = self.client.get_or_create_collection(name="gecko_sessions")

    # Session 接口
    async def get(self, session_id: str) -> Dict[str, Any] | None:
        result = self.session_collection.get(ids=[session_id], include=["metadatas"])
        return result["metadatas"][0] if result["metadatas"] else None

    async def set(self, session_id: str, state: Dict[str, Any]):
        self.session_collection.upsert(ids=[session_id], metadatas=[state])

    async def delete(self, session_id: str):
        self.session_collection.delete(ids=[session_id])

    # Vector 接口
    async def upsert(self, documents: List[Dict]):
        self.vector_collection.upsert(
            ids=[d["id"] for d in documents],
            documents=[d["text"] for d in documents],
            metadatas=[d.get("metadata", {}) for d in documents],
            embeddings=[d.get("embedding") for d in documents]  # 可选外部嵌入
        )

    async def search(self, query_embedding: List[float], top_k: int = 5):
        results = self.vector_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        return [
            {
                "text": doc,
                "metadata": meta,
                "score": 1 - dist  # 转为相似度
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]