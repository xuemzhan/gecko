# gecko/plugins/storage/interfaces.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class SessionInterface(ABC):
    """
    Session 存储接口协议
    负责 Agent 的短期记忆（Conversation History）和状态（State）的持久化
    """
    @abstractmethod
    async def get(self, session_id: str) -> Dict[str, Any] | None:
        """
        获取会话状态
        :param session_id: 会话唯一标识
        :return: 状态字典，如果不存在返回 None
        """
        pass

    @abstractmethod
    async def set(self, session_id: str, state: Dict[str, Any]) -> None:
        """
        设置/更新会话状态
        :param session_id: 会话唯一标识
        :param state: 要保存的状态字典（需可 JSON 序列化）
        """
        pass

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """
        删除会话
        :param session_id: 会话唯一标识
        """
        pass

class VectorInterface(ABC):
    """
    Vector 存储接口协议 (RAG 用)
    负责文档的向量存储与检索
    """
    @abstractmethod
    async def upsert(self, documents: List[Dict[str, Any]]) -> None:
        """
        插入或更新向量文档
        :param documents: 文档列表，每项需包含 id, embedding, text, metadata
        """
        pass

    @abstractmethod
    async def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """
        向量相似度搜索
        :param query_embedding: 查询向量
        :param top_k: 返回结果数量
        :return: 包含 text, metadata, score 的结果列表
        """
        pass