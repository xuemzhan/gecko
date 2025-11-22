# gecko/plugins/storage/backends/lancedb.py
"""
LanceDB 存储后端 (非阻塞优化版)

LanceDB 是基于 Arrow 格式的高性能、无服务器向量数据库。
文件 IO 操作通过 ThreadOffloadMixin 卸载。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from gecko.core.logging import get_logger
from gecko.plugins.storage.abc import AbstractStorage
from gecko.plugins.storage.interfaces import VectorInterface
from gecko.plugins.storage.mixins import ThreadOffloadMixin
from gecko.plugins.storage.utils import parse_storage_url

logger = get_logger(__name__)


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
        
        self.db = None
        self.table = None

    async def initialize(self) -> None:
        """异步初始化"""
        if self.is_initialized:
            return

        def _init_sync():
            import lancedb
            logger.info("Connecting to LanceDB", path=self.db_path)
            
            self.db = lancedb.connect(self.db_path)
            
            # 检查表是否存在
            if self.table_name in self.db.table_names():
                self.table = self.db.open_table(self.table_name)
                logger.debug(f"Opened existing table: {self.table_name}")
            else:
                self.table = None # 首次写入时创建
                logger.debug(f"Table {self.table_name} will be created on first upsert")

        await self._run_sync(_init_sync)
        self._is_initialized = True

    async def shutdown(self) -> None:
        self.db = None
        self.table = None
        self._is_initialized = False

    async def upsert(self, documents: List[Dict[str, Any]]) -> None:
        """插入数据 (首次自动建表)"""
        if not documents:
            return

        def _sync_upsert():
            import pyarrow as pa
            
            # 构造数据
            # LanceDB 期望数据格式：[{"vector": ..., "id": ..., ...}]
            data = []
            for doc in documents:
                item = {
                    "id": doc["id"],
                    "vector": doc["embedding"],
                    "text": doc.get("text", ""),
                    "metadata": doc.get("metadata", {})
                }
                data.append(item)

            if self.table is None:
                # 首次创建表
                try:
                    self.table = self.db.create_table(self.table_name, data=data) # type: ignore
                    logger.info(f"Created LanceDB table: {self.table_name}")
                except Exception as e:
                    # 处理并发创建冲突
                    if self.table_name in self.db.table_names(): # type: ignore
                        self.table = self.db.open_table(self.table_name) # type: ignore
                        self.table.add(data)
                    else:
                        raise e
            else:
                # 追加数据
                self.table.add(data)

        await self._run_sync(_sync_upsert)

    async def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """向量搜索"""
        if self.table is None:
            return []

        def _sync_search():
            # LanceDB 的 search API
            results = self.table.search(query_embedding).limit(top_k).to_list() # type: ignore
            
            parsed_results = []
            for r in results:
                # LanceDB 返回 distance (L2), 若用 cosine 需转换
                # 这里假设默认距离，返回原始 score (通常是 distance)
                # Gecko 约定 score 越高越好，这里做一个简单的 1 / (1 + distance) 或者直接返回
                # 很多场景下 LanceDB 默认是用 L2 distance
                distance = r.get("_distance", 0.0)
                
                parsed_results.append({
                    "id": r["id"],
                    "text": r["text"],
                    "metadata": r["metadata"],
                    "score": 1.0 - distance # 粗略转换，具体视 metric 而定
                })
            return parsed_results

        return await self._run_sync(_sync_search)