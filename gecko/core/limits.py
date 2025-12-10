# gecko/core/limits.py
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Tuple


class ConcurrencyLimiter:
    """
    简单进程内并发控制器

    - scope: "agent" / "model" / 其他自定义维度
    - name: 具体实例名，例如 Agent 名称、模型名
    """

    def __init__(self) -> None:
        # key: (scope, name) -> Semaphore
        self._semaphores: Dict[Tuple[str, str], asyncio.Semaphore] = {}

    def _get_semaphore(self, scope: str, name: str, limit: int) -> asyncio.Semaphore:
        key = (scope, name)
        sem = self._semaphores.get(key)
        if sem is None:
            sem = asyncio.Semaphore(limit)
            self._semaphores[key] = sem
        return sem

    @asynccontextmanager
    async def limit(self, scope: str, name: str, limit: int):
        """
        以 (scope, name) 维度控制并发。

        limit <= 0 表示不限制。
        """
        if limit <= 0:
            # 不限流则直接透传
            yield
            return

        sem = self._get_semaphore(scope, name, limit)
        await sem.acquire()
        try:
            yield
        finally:
            sem.release()


# 全局共享实例
global_limiter = ConcurrencyLimiter()
