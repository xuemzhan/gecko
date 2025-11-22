# gecko/plugins/storage/backends/lancedb.py
"""
LanceDB 存储后端

LanceDB 是一个无服务器、基于 Arrow 的高性能向量数据库。
本实现通过 ThreadOffloadMixin 处理文件 I/O。

核心特性：
1. **自动建表**：首次 Upsert 时根据数据结构自动创建表。
2. **线程安全**：I/O 操作隔离。

更新日志：
- [Robustness] 增加 metadata 为 None 的防御性处理。
- [Robustness] 所有操作统一抛出 StorageError。
- [Feature] 实现 search 方法的 filters 参数支持 (SQL-like string construction)。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from gecko.core.exceptions import StorageError
from gecko.core.logging import get_logger
from gecko.plugins.storage.abc import AbstractStorage
from gecko.plugins.storage.interfaces import VectorInterface
from gecko.plugins.storage.mixins import ThreadOffloadMixin
from gecko.plugins.storage.registry import register_storage
from gecko.plugins.storage.utils import parse_storage_url

logger = get_logger(__name__)


@register_storage("lancedb")
class LanceDBStorage(
    AbstractStorage,
    VectorInterface,
    ThreadOffloadMixin
):
    def __init__(self, url: str, **kwargs):
        super().__init__(url, **kwargs)
        scheme, path, params = parse_storage_url(url)
        
        self.db_path = path
        self.table_name = params.get("table", "gecko_vectors")
        self.embedding_dim = int(params.get("dim", 1536))
        
        self.db: Any = None
        self.table: Any = None

    async def initialize(self) -> None:
        if self.is_initialized:
            return

        def _init_sync():
            try:
                import lancedb
                logger.info("Connecting to LanceDB", path=self.db_path)
                
                self.db = lancedb.connect(self.db_path)
                
                if self.table_name in self.db.table_names():
                    self.table = self.db.open_table(self.table_name)
                    logger.debug(f"Opened existing table: {self.table_name}")
                else:
                    self.table = None
                    logger.debug(f"Table {self.table_name} will be created on first upsert")
            except Exception as e:
                raise StorageError(f"Failed to initialize LanceDB: {e}") from e

        await self._run_sync(_init_sync)
        self._is_initialized = True

    async def shutdown(self) -> None:
        self.db = None
        self.table = None
        self._is_initialized = False

    async def upsert(self, documents: List[Dict[str, Any]]) -> None:
        if not documents:
            return

        def _sync_upsert():
            try:
                data = []
                for doc in documents:
                    # [防御性处理] 确保 metadata 是字典
                    meta = doc.get("metadata")
                    if meta is None:
                        meta = {}
                    
                    item = {
                        "id": doc["id"],
                        "vector": doc["embedding"],
                        "text": doc.get("text", ""),
                        "metadata": meta
                    }
                    data.append(item)

                if self.table is None:
                    try:
                        self.table = self.db.create_table(self.table_name, data=data)
                        logger.info(f"Created LanceDB table: {self.table_name}")
                    except Exception as e:
                        # 处理并发创建冲突
                        if self.table_name in self.db.table_names():
                            self.table = self.db.open_table(self.table_name)
                            self.table.add(data)
                        else:
                            raise e
                else:
                    self.table.add(data)
            except Exception as e:
                raise StorageError(f"LanceDB upsert failed: {e}") from e

        await self._run_sync(_sync_upsert)

    async def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if self.table is None:
            return []

        def _sync_search():
            try:
                if self.table is None:
                    return []
                
                # 构建查询
                query = self.table.search(query_embedding).limit(top_k)
                
                # [Feature] 添加 where 子句
                # LanceDB metadata 存储为 struct，查询语法: "metadata.key = 'value'"
                if filters:
                    clauses = []
                    for k, v in filters.items():
                        if isinstance(v, str):
                            clauses.append(f"metadata.{k} = '{v}'")
                        else:
                            clauses.append(f"metadata.{k} = {v}")
                    
                    if clauses:
                        where_str = " AND ".join(clauses)
                        query = query.where(where_str)

                results = query.to_list()
                
                parsed_results = []
                for r in results:
                    distance = r.get("_distance", 0.0)
                    
                    parsed_results.append({
                        "id": r["id"],
                        "text": r["text"],
                        "metadata": r["metadata"],
                        "score": 1.0 / (1.0 + distance) 
                    })
                return parsed_results
            except Exception as e:
                raise StorageError(f"LanceDB search failed: {e}") from e

        return await self._run_sync(_sync_search)