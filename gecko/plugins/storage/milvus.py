# gecko/plugins/storage/milvus.py
from __future__ import annotations
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from typing import List, Dict
from gecko.plugins.storage.registry import register_storage
from gecko.plugins.storage.interfaces import VectorInterface

@register_storage("milvus")
class MilvusStorage(VectorInterface):
    """
    超大规模生产级 Vector 存储：Milvus
    - URL 示例：milvus://localhost:19530
    - 特点：支持十亿级向量、分布式
    """
    def __init__(self, storage_url: str, collection_name: str = "gecko_default", embedding_dim: int = 1536, **kwargs):
        uri = storage_url.removeprefix("milvus://")
        connections.connect(uri=uri)
        self.collection_name = collection_name
        
        if not Collection.has_collection(collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=500),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            schema = CollectionSchema(fields)
            self.collection = Collection(collection_name, schema)
            self.collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}})
        else:
            self.collection = Collection(collection_name)
        self.collection.load()

    async def upsert(self, documents: List[Dict]):
        self.collection.insert([
            [d["id"] for d in documents],
            [d["embedding"] for d in documents],
            [d["text"] for d in documents],
            [d.get("metadata", {}) for d in documents]
        ])

    async def search(self, query_embedding: List[float], top_k: int = 5):
        params = {"metric_type": "IP", "params": {"nprobe": 16}}
        results = self.collection.search([query_embedding], "embedding", params, top_k, output_fields=["text", "metadata"])
        return [
            {"text": r.entity.get("text"), "metadata": r.entity.get("metadata"), "score": r.distance}
            for r in results[0]
        ]