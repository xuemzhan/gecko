# gecko/plugins/storage/backends/chroma.py
"""
ChromaDB 存储后端

ChromaDB 是一个开源的嵌入式向量数据库。
本实现通过 ThreadOffloadMixin 将其同步 I/O 操作卸载到线程池，以避免阻塞 Event Loop。

核心特性：
1. **非阻塞架构**：所有数据库操作均在 Worker 线程执行。
2. **性能优化**：默认禁用 SentenceTransformer 模型加载，大幅降低内存占用和启动时间。
3. **双重接口**：同时支持 VectorInterface (RAG) 和 SessionInterface (KV)。

注意：
Session 数据将被序列化为 JSON 字符串存储在 Document 字段中，以绕过 Chroma Metadata 的类型限制。
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
from gecko.plugins.storage.registry import register_storage
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


@register_storage("chroma")
class ChromaStorage(
    AbstractStorage,
    VectorInterface,
    SessionInterface,
    ThreadOffloadMixin,
    JSONSerializerMixin
):
    """
    基于 ChromaDB 的统一存储后端
    
    URL 示例:
        chroma://./chroma_db?collection=my_app
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
            
            ef : Any = IdentityEmbeddingFunction()
            
            # 1. 向量集合
            v_col = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # 强制使用余弦相似度
                embedding_function=ef
            )
            
            # 2. 会话集合 (KV 模式)
            s_col = client.get_or_create_collection(
                name=f"{self.collection_name}_sessions",
                embedding_function=ef
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
        if not documents or not self.vector_col:
            return

        def _sync_upsert():
            # 这里的 self.vector_col 在初始化后一定存在，但静态检查不知道
            # 实际运行中有 is_initialized 保护
            if self.vector_col: 
                self.vector_col.upsert(
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
        if not self.vector_col:
            return []

        def _sync_search():
            if not self.vector_col:
                return []
                
            results = self.vector_col.query(
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
        if not self.session_col:
            return None

        def _sync_get():
            if not self.session_col:
                return None
            result = self.session_col.get(
                ids=[session_id],
                include=["documents"] # type: ignore
            )
            if result["ids"] and result["documents"]:
                return result["documents"][0]
            return None

        json_str = await self._run_sync(_sync_get)
        return self._deserialize(json_str)

    async def set(self, session_id: str, state: Dict[str, Any]) -> None:
        """
        保存会话状态
        
        原理：将 state 序列化为 JSON 存入 document 字段
        """
        if not self.session_col:
            return

        json_str = self._serialize(state)

        def _sync_set():
            if not self.session_col:
                return
            # 使用 upsert 覆盖旧值
            self.session_col.upsert(
                ids=[session_id],
                documents=[json_str],
                metadatas=[{"type": "session_state"}] # 必须有 metadata 或 embedding
            )

        await self._run_sync(_sync_set)

    async def delete(self, session_id: str) -> None:
        """删除会话"""
        if not self.session_col:
            return

        def _sync_delete():
            if not self.session_col:
                return
            self.session_col.delete(ids=[session_id])
        
        await self._run_sync(_sync_delete)