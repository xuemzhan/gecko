# gecko/plugins/storage/interfaces.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List

class SessionInterface(ABC):
    """Session 存储接口：KV 风格，必须实现的抽象方法"""
    @abstractmethod
    async def get(self, session_id: str) -> Dict[str, Any] | None:
        """获取会话状态，若不存在返回 None"""

    @abstractmethod
    async def set(self, session_id: str, state: Dict[str, Any]) -> None:
        """设置会话状态，自动创建或更新"""

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """删除会话"""

class VectorInterface(ABC):
    """Vector 存储接口：RAG 风格，必须实现的抽象方法"""
    @abstractmethod
    async def upsert(self, documents: List[Dict[str, Any]]) -> None:
        """插入或更新文档列表，每个文档必须含 id, embedding, text, metadata"""

    @abstractmethod
    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """搜索相似文档，返回 list[dict] 含 text, metadata, score"""