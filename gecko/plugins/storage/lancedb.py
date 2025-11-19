# gecko/plugins/storage/lancedb.py
from __future__ import annotations
import lancedb
import pyarrow as pa
from typing import List, Dict
from gecko.plugins.storage.registry import register_storage
from gecko.plugins.storage.interfaces import VectorInterface

@register_storage("lancedb")
class LanceDBVectorStorage(VectorInterface):
    """
    快速开发专用 Vector 存储插件
    - URL 示例：lancedb://./dev_vector_db
    - 特点：纯 Python、本地目录存储、毫秒启动、支持 ANN 搜索
    - 适用：本地 RAG 验证
    - 注意：首次 upsert 时自动创建表
    """
    def __init__(self, storage_url: str, collection_name: str = "gecko_default", embedding_dim: int = 1536, **kwargs):
        db_path = storage_url.removeprefix("lancedb://")
        self.db = lancedb.connect(db_path)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.table = self.db.open_table(collection_name) if collection_name in self.db.table_names() else None

    async def upsert(self, documents: List[Dict]):
        """插入/更新文档，首次自动创建表"""
        if not documents:
            return

        data = pa.table({
            "id": [d["id"] for d in documents],
            "vector": [d["embedding"] for d in documents],
            "text": [d["text"] for d in documents],
            "metadata": [d.get("metadata", {}) for d in documents]
        })

        if self.table is None:
            self.table = self.db.create_table(self.collection_name, data=data)
        else:
            self.table.add(data)

    async def search(self, query_embedding: List[float], top_k: int = 5):
        """搜索相似文档"""
        if self.table is None:
            return []

        results = self.table.search(query_embedding).limit(top_k).to_pylist()
        return [
            {"text": r["text"], "metadata": r["metadata"], "score": 1 - r["_distance"]}
            for r in results
        ]
