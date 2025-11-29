# gecko/utils/cleanup.py
import atexit
import asyncio

def register_litellm_cleanup():
    """在进程退出时优雅关闭 LiteLLM 异步客户端，避免 RuntimeWarning"""
    def _cleanup():
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果循环还在运行，调度清理任务
                loop.create_task(_close_clients())
            else:
                # 循环已关闭，直接新开一个临时循环执行清理
                asyncio.run(_close_clients())
        except Exception:
            pass  # 防止清理本身抛错

    async def _close_clients():
        try:
            import litellm
            # LiteLLM 官方提供的异步关闭方法（v1.40+ 支持）
            if hasattr(litellm, "async_http_handler"):
                if litellm.async_http_handler: # type: ignore
                    await litellm.async_http_handler.client.close() # type: ignore
            # 兼容旧版本
            if hasattr(litellm, "http_client"):
                if litellm.http_client: # type: ignore
                    await litellm.http_client.close() # type: ignore
        except Exception:
            pass

    atexit.register(_cleanup)

# 自动注册（模块导入即生效）
register_litellm_cleanup()