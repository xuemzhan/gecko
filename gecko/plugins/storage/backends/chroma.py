# gecko/plugins/storage/backends/chroma.py
"""
ChromaDB 存储后端

更新日志：
- [Robustness] 增加 metadata 为 None 的防御性处理。
- [Robustness] 所有操作统一抛出 StorageError，屏蔽底层异常。
- [Feature] 实现 search 方法的 filters 参数支持 (Mapping dict to Chroma `where` clause)。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

# 仅用于类型检查的导入
if TYPE_CHECKING:
    from chromadb import ClientAPI, Collection # type: ignore

from gecko.core.exceptions import StorageError
from gecko.core.logging import get_logger
from gecko.plugins.storage.abc import AbstractStorage
from gecko.plugins.storage.interfaces import SessionInterface, VectorInterface
from gecko.plugins.storage.mixins import (
    JSONSerializerMixin,
    ThreadOffloadMixin,
)
from gecko.plugins.storage.registry import register_storage
from gecko.plugins.storage.utils import parse_storage_url

logger = get_logger(__name__)


class IdentityEmbeddingFunction:
    """
    空实现的 Embedding Function。
    用于禁用 Chroma 内置的模型加载，提升性能。
    """
    def __call__(self, input: Any) -> Any:
        return [[0.0] for _ in input]

    def name(self) -> str:
        return "gecko_identity"


@register_storage("chroma")
class ChromaStorage(
    AbstractStorage,
    VectorInterface,
    SessionInterface,
    ThreadOffloadMixin,
    JSONSerializerMixin
):
    def __init__(self, url: str, **kwargs):
        super().__init__(url, **kwargs)
        scheme, path, params = parse_storage_url(url)
        
        self.persist_path = path
        self.collection_name = params.get("collection", "gecko_default")
        
        self.client: Optional[ClientAPI] = None
        self.vector_col: Optional[Collection] = None
        self.session_col: Optional[Collection] = None

    async def initialize(self) -> None:
        if self.is_initialized:
            return

        def _init_sync():
            try:
                import chromadb
                from chromadb.config import Settings

                logger.info("Initializing ChromaDB", path=self.persist_path)
                
                client = chromadb.PersistentClient(
                    path=self.persist_path,
                    settings=Settings(anonymized_telemetry=False)
                )
                
                ef : Any = IdentityEmbeddingFunction()
                
                v_col = client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=ef
                )
                
                s_col = client.get_or_create_collection(
                    name=f"{self.collection_name}_sessions",
                    embedding_function=ef
                )
                
                return client, v_col, s_col
            except Exception as e:
                raise StorageError(f"Failed to initialize ChromaDB: {e}") from e

        self.client, self.vector_col, self.session_col = await self._run_sync(_init_sync)
        self._is_initialized = True

    async def shutdown(self) -> None:
        self.client = None
        self.vector_col = None
        self.session_col = None
        self._is_initialized = False

    # ==================== VectorInterface 实现 ====================

    async def upsert(self, documents: List[Dict[str, Any]]) -> None:
        if not documents or not self.vector_col:
            return

        def _sync_upsert():
            try:
                if self.vector_col: 
                    # [修复] Chroma 不接受空字典作为 metadata，必须为 None 或 非空字典
                    metadatas = []
                    for d in documents:
                        m = d.get("metadata")
                        # 如果 m 是 None 或 {}，都转为 None
                        metadatas.append(m if m else None)
                    
                    self.vector_col.upsert(
                        ids=[d["id"] for d in documents],
                        embeddings=[d["embedding"] for d in documents],
                        metadatas=metadatas, # type: ignore
                        documents=[d.get("text", "") for d in documents]
                    )
            except Exception as e:
                raise StorageError(f"Chroma upsert failed: {e}") from e

        await self._run_sync(_sync_upsert)

    async def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if not self.vector_col:
            return []

        def _sync_search():
            try:
                if not self.vector_col:
                    return []
                
                # [Feature] 构造 Chroma filter
                # Chroma where 语法: {"field": "value"} 或 {"$and": [...]}
                chroma_filter = None
                if filters:
                    if len(filters) == 1:
                        chroma_filter = filters
                    else:
                        # 多个条件默认为 AND
                        chroma_filter = {"$and": [{k: v} for k, v in filters.items()]}

                results = self.vector_col.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=chroma_filter, # type: ignore
                    include=["metadatas", "documents", "distances"]
                )
                
                parsed_results = []
                if not results["ids"]:
                    return []

                count = len(results["ids"][0])
                for i in range(count):
                    dist = results["distances"][0][i] # type: ignore
                    score = max(0.0, 1.0 - dist)

                    # [Fix] 处理 metadata 为 None 的情况，统一返回 {}
                    raw_meta = results["metadatas"][0][i] # type: ignore
                    meta = raw_meta if raw_meta is not None else {}
                    
                    parsed_results.append({
                        "id": results["ids"][0][i],
                        "text": results["documents"][0][i], # type: ignore
                        "metadata": meta,
                        "score": score
                    })
                
                return parsed_results
            except Exception as e:
                raise StorageError(f"Chroma search failed: {e}") from e

        return await self._run_sync(_sync_search)

    # ==================== SessionInterface 实现 ====================

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        if not self.session_col:
            return None

        def _sync_get():
            try:
                if not self.session_col:
                    return None
                result = self.session_col.get(
                    ids=[session_id],
                    include=["documents"] # type: ignore
                )
                if result["ids"] and result["documents"]:
                    return result["documents"][0]
                return None
            except Exception as e:
                raise StorageError(f"Chroma session get failed: {e}") from e

        json_str = await self._run_sync(_sync_get)
        return self._deserialize(json_str)

    async def set(self, session_id: str, state: Dict[str, Any]) -> None:
        if not self.session_col:
            return

        json_str = self._serialize(state)

        def _sync_set():
            try:
                if not self.session_col:
                    return
                self.session_col.upsert(
                    ids=[session_id],
                    documents=[json_str],
                    metadatas=[{"type": "session_state"}]
                )
            except Exception as e:
                raise StorageError(f"Chroma session set failed: {e}") from e

        await self._run_sync(_sync_set)

    async def delete(self, session_id: str) -> None:
        if not self.session_col:
            return

        def _sync_delete():
            try:
                if not self.session_col:
                    return
                self.session_col.delete(ids=[session_id])
            except Exception as e:
                raise StorageError(f"Chroma session delete failed: {e}") from e
        
        await self._run_sync(_sync_delete)