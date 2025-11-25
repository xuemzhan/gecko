# gecko/__init__.py
from __future__ import annotations

from gecko.core.agent import Agent
from gecko.core.builder import AgentBuilder
from gecko.core.message import Message

import atexit
import asyncio

# 移除原来的直接导入，改为安全的延迟清理注册

def _register_cleanup():
    """
    注册资源清理钩子
    
    优化点：
    1. 延迟导入 litellm，避免硬依赖导致未安装时框架不可用
    2. 只有在 litellm 确实被导入过的情况下才执行清理
    """
    import atexit
    import sys

    def _cleanup_wrapper():
        # 检查 sys.modules 中是否有 litellm
        # 如果用户从未导入过 litellm，这里就不需要做任何事
        if "litellm" not in sys.modules:
            return

        try:
            import asyncio
            import litellm # type: ignore
            
            async def _close():
                try:
                    # 尝试关闭异步客户端连接
                    if hasattr(litellm, "async_http_handler"
                               ) and litellm.async_http_handler: # type: ignore
                        await litellm.async_http_handler.client.close() # type: ignore
                except Exception:
                    pass

            # 依据 Event Loop 状态选择执行方式
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(_close())
                else:
                    asyncio.run(_close())
            except RuntimeError:
                asyncio.run(_close())
                
        except Exception:
            # 清理过程中的错误不应阻塞进程退出
            pass

    atexit.register(_cleanup_wrapper)

_register_cleanup()

__version__ = "0.2.0"

__all__ = ["Agent", "AgentBuilder", "Message"]