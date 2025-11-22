"""嵌入模型协议"""
from __future__ import annotations
from typing import List, Protocol, runtime_checkable

@runtime_checkable
class EmbedderProtocol(Protocol):
    async def embed(self, texts: List[str]) -> List[List[float]]: ...
    def get_dimension(self) -> int: ...