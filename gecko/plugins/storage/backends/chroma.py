# gecko/plugins/storage/backends/chroma.py
"""
ChromaDB 存储后端 (非阻塞优化版)

ChromaDB 是一个流行的开源向量数据库。
本实现通过 ThreadOffloadMixin 将 IO 密集型操作卸载到线程池，
并优化了 Session 存储策略，使用 JSON 序列化存入 document 字段。

支持接口:
1. VectorInterface: 向量检索 (Collection: {name})
2. SessionInterface: 会话存储 (Collection: {name}_sessions)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

# 仅用于类型检查的导入
if TYPE_CHECKING:
    from chromadb import ClientAPI, Collection # type: ignore

from gecko.core.logging import get_logger
from gecko.plugins.storage.abc import AbstractStorage
from gecko.plugins.storage.interfaces import SessionInterface, VectorInterface
from gecko.plugins.storage.mixins import (
    JSONSerializerMixin,
    ThreadOffloadMixin,
)
from gecko.plugins.storage.utils import parse_storage_url

logger = get_logger(__name__)



class IdentityEmbeddingFunction:
    """
    空实现的 Embedding Function。
    
    作用：
    1. 告诉 Chroma 不要加载默认的 SentenceTransformer 模型。
    2. 提供一个最小的合法向量 (dim=1) 以满足 Chroma 的非空检查。
    """
    
    def __call__(self, input: Any) -> Any:
        # [修复] 返回维度为 1 的哑向量，防止 Chroma 报错
        # 这里的向量值不重要，因为 Session 存储不依赖相似度搜索
        return [[0.0] for _ in input]

    def name(self) -> str:
        return "gecko_identity"


class ChromaStorage(
    AbstractStorage,
    VectorInterface,
    SessionInterface,
    ThreadOffloadMixin,
    JSONSerializerMixin
):
    """
    基于 ChromaDB 的统一存储后端
    
    特性：
    - 异步非阻塞：基于 ThreadOffloadMixin
    - 懒加载：仅在 initialize 时加载 chromadb 库
    - 双模式：同时支持向量存储和简单的 KV 会话存储
    """

    def __init__(self, url: str, **kwargs):
        super().__init__(url, **kwargs)
        scheme, path, params = parse_storage_url(url)
        
        self.persist_path = path
        self.collection_name = params.get("collection", "gecko_default")
        
        # 运行时对象 (初始化后可用)
        self.client: Optional[ClientAPI] = None
        self.vector_col: Optional[Collection] = None
        self.session_col: Optional[Collection] = None

    async def initialize(self) -> None:
        """
        异步初始化
        
        建立连接并确保集合存在。显式指定 embedding_function 以禁用 Chroma 内置模型。
        """
        if self.is_initialized:
            return

        def _init_sync():
            # 懒加载以加快启动速度
            import chromadb
            from chromadb.config import Settings

            logger.info("Initializing ChromaDB", path=self.persist_path)
            
            client = chromadb.PersistentClient(
                path=self.persist_path,
                settings=Settings(anonymized_telemetry=False)
            )
            
            ef = IdentityEmbeddingFunction()
            
            # 1. 向量集合
            v_col = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # 强制使用余弦相似度
                embedding_function=ef # type: ignore
            )
            
            # 2. 会话集合 (KV 模式)
            s_col = client.get_or_create_collection(
                name=f"{self.collection_name}_sessions",
                embedding_function=ef # type: ignore
            )
            
            return client, v_col, s_col

        # 卸载到线程池执行
        self.client, self.vector_col, self.session_col = await self._run_sync(_init_sync)
        self._is_initialized = True

    async def shutdown(self) -> None:
        """关闭连接（清理引用）"""
        self.client = None
        self.vector_col = None
        self.session_col = None
        self._is_initialized = False

    # ==================== VectorInterface 实现 ====================

    async def upsert(self, documents: List[Dict[str, Any]]) -> None:
        """
        插入或更新向量
        
        documents结构:
        [
            {"id": "...", "embedding": [...], "text": "...", "metadata": {...}}
        ]
        """
        if not documents:
            return

        def _sync_upsert():
            self.vector_col.upsert( # type: ignore
                ids=[d["id"] for d in documents],
                embeddings=[d["embedding"] for d in documents],
                metadatas=[d.get("metadata", {}) for d in documents],
                documents=[d.get("text", "") for d in documents]
            )

        await self._run_sync(_sync_upsert)

    async def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        向量搜索
        
        返回结果包含 score (相似度，范围 0-1)
        """
        def _sync_search():
            results = self.vector_col.query( # type: ignore
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["metadatas", "documents", "distances"]
            )
            
            parsed_results = []
            if not results["ids"]:
                return []

            # Chroma 返回的是 list of lists
            count = len(results["ids"][0])
            for i in range(count):
                dist = results["distances"][0][i] # type: ignore
                # Cosine Distance 范围 [0, 2], 0 表示完全相同
                # 转换为相似度: 1.0 - distance
                score = max(0.0, 1.0 - dist)
                
                parsed_results.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i], # type: ignore
                    "metadata": results["metadatas"][0][i], # type: ignore
                    "score": score
                })
            
            return parsed_results

        return await self._run_sync(_sync_search)

    # ==================== SessionInterface 实现 ====================

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话状态
        
        原理：读取 document 字段中的 JSON 字符串并反序列化
        """
        def _sync_get():
            result = self.session_col.get( # type: ignore
                ids=[session_id],
                include=["documents"]
            )
            if result["ids"] and result["documents"][0]: # type: ignore
                return result["documents"][0] # type: ignore
            return None

        json_str = await self._run_sync(_sync_get)
        return self._deserialize(json_str)

    async def set(self, session_id: str, state: Dict[str, Any]) -> None:
        """
        保存会话状态
        
        原理：将 state 序列化为 JSON 存入 document 字段
        """
        json_str = self._serialize(state)

        def _sync_set():
            # 使用 upsert 覆盖旧值
            self.session_col.upsert( # type: ignore
                ids=[session_id],
                documents=[json_str],
                metadatas=[{"type": "session_state"}] # 必须有 metadata 或 embedding
            )

        await self._run_sync(_sync_set)

    async def delete(self, session_id: str) -> None:
        """删除会话"""
        def _sync_delete():
            self.session_col.delete(ids=[session_id]) # type: ignore
        
        await self._run_sync(_sync_delete)