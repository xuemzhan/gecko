# gecko/plugins/knowledge/document.py
from __future__ import annotations
from uuid import uuid4
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class Document(BaseModel):
    """
    Gecko 标准文档对象
    在 Pipeline 中流转的核心数据结构
    """
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为存储层所需的字典格式"""
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "embedding": self.embedding
        }