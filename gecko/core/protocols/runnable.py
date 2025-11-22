"""可运行对象协议"""
from __future__ import annotations
from typing import Any, AsyncIterator, Protocol, runtime_checkable

@runtime_checkable
class RunnableProtocol(Protocol):
    async def run(self, input: Any) -> Any: ...

@runtime_checkable
class StreamableRunnableProtocol(RunnableProtocol, Protocol):
    async def stream(self, input: Any) -> AsyncIterator[str]:
        ...
        yield # type: ignore