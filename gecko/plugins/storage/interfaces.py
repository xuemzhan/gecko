# gecko/plugins/storage/interfaces.py
"""
业务接口定义

定义 Session（会话存储）和 Vector（向量存储）的标准行为。

更新日志：
- [Feat] VectorInterface.search 增加 filters 参数，支持元数据过滤。
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class SessionInterface(Protocol):
    """
    Session 存储接口协议
    
    负责 Agent 的短期记忆（Conversation History）和状态（State）的持久化。
    """
    
    @abstractmethod
    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话状态
        
        参数:
            session_id: 会话唯一标识
            
        返回:
            状态字典，如果不存在返回 None
        """
        ...

    @abstractmethod
    async def set(self, session_id: str, state: Dict[str, Any]) -> None:
        """
        设置/更新会话状态
        
        参数:
            session_id: 会话唯一标识
            state: 要保存的状态字典
        """
        ...

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """
        删除会话
        
        参数:
            session_id: 会话唯一标识
        """
        ...


@runtime_checkable
class VectorInterface(Protocol):
    """
    Vector 存储接口协议 (RAG 用)
    
    负责文档的向量存储与检索。
    """
    
    @abstractmethod
    async def upsert(self, documents: List[Dict[str, Any]]) -> None:
        """
        插入或更新向量文档
        
        参数:
            documents: 文档列表，每项需包含 id, embedding, text, metadata
        """
        ...

    @abstractmethod
    async def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        向量相似度搜索
        
        参数:
            query_embedding: 查询向量
            top_k: 返回结果数量
            filters: 元数据过滤条件 (Key-Value 精确匹配)
            
        返回:
            包含 text, metadata, score 的结果列表
        """
        ...