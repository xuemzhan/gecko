# gecko/plugins/models/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List

from gecko.core.protocols import (
    CompletionResponse,
    EmbedderProtocol,
    StreamChunk,
    StreamableModelProtocol,
)
from gecko.plugins.models.config import ModelConfig


class AbstractModel(ABC):
    """所有模型的根基类"""

    def __init__(self, config: ModelConfig):
        self.config = config


class BaseChatModel(AbstractModel, StreamableModelProtocol):
    """
    Chat 模型驱动基类
    
    所有具体的驱动器 (Driver) 都必须继承此类并实现 `acompletion` 和 `astream`。
    """

    @abstractmethod
    async def acompletion(self, messages: List[Dict[str, Any]], **kwargs: Any) -> CompletionResponse:
        """单次生成"""
        ...

    @abstractmethod
    async def astream(self, messages: List[Dict[str, Any]], **kwargs: Any) -> AsyncIterator[StreamChunk]: # type: ignore
        """流式生成"""
        ...


class BaseEmbedder(AbstractModel, EmbedderProtocol):
    """
    Embedding 模型基类
    """
    
    def __init__(self, config: ModelConfig, dimension: int):
        super().__init__(config)
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """返回向量维度"""
        return self._dimension

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入"""
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """单条查询嵌入"""
        ...