# gecko/plugins/knowledge/interfaces.py
from __future__ import annotations
from typing import List, Protocol, runtime_checkable

@runtime_checkable
class EmbedderProtocol(Protocol):
    """
    嵌入模型协议
    负责将文本转换为向量
    """
    @property
    def dimension(self) -> int:
        """返回向量维度 (例如 1536)"""
        ...

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档列表"""
        ...

    async def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询语句"""
        ...

@runtime_checkable
class ReaderProtocol(Protocol):
    """
    文件读取协议
    """
    def load(self, file_path: str) -> str:
        """读取文件内容为字符串"""
        ...