# gecko/__init__.py
from __future__ import annotations

from gecko.core.agent import Agent
from gecko.core.builder import AgentBuilder
from gecko.core.message import Message

# 自动清理 LiteLLM 异步客户端，彻底消除 RuntimeWarning
import atexit
import asyncio
import litellm # type: ignore

def _cleanup_litellm():
    async def _close():
        try:
            if hasattr(litellm, "async_http_handler") and litellm.async_http_handler:
                await litellm.async_http_handler.client.close()
        except:
            pass
    try:
        asyncio.run(_close())
    except:
        pass

atexit.register(_cleanup_litellm)

__version__ = "0.1.0"
__all__ = ["Agent", "AgentBuilder", "Message"]