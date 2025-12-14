# gecko/core/limits.py
from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Dict, Tuple, Optional


@dataclass
class _SemEntry:
    """
    缓存条目：保存 semaphore + 最近访问时间

    last_access：
      - 用于 TTL 清理
      - 同时也用于 LRU 的“最近访问”语义（OrderedDict 的 move_to_end）
    """
    sem: asyncio.Semaphore
    last_access: float


class ConcurrencyLimiter:
    """
    进程内并发控制器（支持 LRU + TTL 的 semaphore 缓存）

    为什么需要缓存？
    - 同一个 (scope, name, limit) 会被频繁请求
    - 复用 semaphore 可以实现对该维度的全局并发约束
    - 但缓存不能无界增长：需要 max_entries（LRU）和 ttl_seconds（过期清理）

    参数：
    - max_entries：
        缓存最大条目数（LRU），默认 1024
        <= 0 表示不限制（不建议）
    - ttl_seconds：
        条目过期时间（单位秒），默认 3600（1小时）
        <= 0 表示不启用 TTL
    """

    def __init__(self, max_entries: int = 1024, ttl_seconds: float = 3600) -> None:
        self._max_entries = int(max_entries)
        self._ttl_seconds = float(ttl_seconds)

        # OrderedDict 用于 LRU：
        # - key: (scope, name, limit)
        # - value: _SemEntry
        # 约定：最近访问的放在右侧（end）
        self._cache: "OrderedDict[Tuple[str, str, int], _SemEntry]" = OrderedDict()

        # 注意：_get_semaphore 是同步方法，且可能在多线程/多协程混用环境下被调用，
        # 用线程锁保护 OrderedDict 的一致性（不使用 asyncio.Lock，避免同步函数无法 await）
        self._lock = RLock()

    # ---------------------------------------------------------------------
    # 供测试使用的内部接口
    # ---------------------------------------------------------------------

    def _cache_size(self) -> int:
        """返回缓存条目数量（测试用）。"""
        with self._lock:
            return len(self._cache)

    # ---------------------------------------------------------------------
    # 核心：获取/创建 semaphore（LRU + TTL）
    # ---------------------------------------------------------------------

    def _get_semaphore(self, scope: str, name: str, limit: int) -> asyncio.Semaphore:
        """
        获取指定维度的 semaphore（必要时创建），并执行：
        1) TTL 清理
        2) LRU 维护（命中则 move_to_end）
        3) 插入后若超出 max_entries，则执行 LRU 淘汰

        说明：
        - key 包含 limit：避免同一 (scope,name) 在不同 limit 下复用导致并发语义错误
        - 如果你希望 limit 变化时复用旧 semaphore，需要更复杂的“重建”策略；这里采用更安全的区分 key 策略
        """
        if limit <= 0:
            # limit<=0 的语义一般是“不限流”，但 semaphore 不能表示无限；
            # 这里返回一个“永远可 acquire”的大 semaphore 也不合适。
            # 因为上层 limit() 已经对 limit<=0 做了透传处理，所以理论上不会走到这里。
            limit = 1

        now = time.time()
        key = (scope, name, int(limit))

        with self._lock:
            # 1) 先做 TTL 清理（避免缓存长期积累）
            self._cleanup_expired_locked(now)

            # 2) 命中：更新 last_access，并移动到 LRU 尾部
            entry = self._cache.get(key)
            if entry is not None:
                entry.last_access = now
                self._cache.move_to_end(key, last=True)
                return entry.sem

            # 3) 未命中：创建新 semaphore 并写入缓存
            sem = asyncio.Semaphore(int(limit))
            self._cache[key] = _SemEntry(sem=sem, last_access=now)
            self._cache.move_to_end(key, last=True)

            # 4) 写入后执行 LRU 淘汰（这是你当前失败的关键缺失点）
            self._evict_lru_locked()

            return sem

    def _cleanup_expired_locked(self, now: float) -> None:
        """
        TTL 清理（必须在持锁状态下调用）

        策略：
        - ttl_seconds <= 0：不启用 TTL
        - 从最老的开始清理更高效（OrderedDict 左侧更可能过期）
        """
        if self._ttl_seconds <= 0:
            return

        ttl = self._ttl_seconds

        # OrderedDict 按 LRU 排序：左边更旧，先检查左边
        keys_to_delete = []
        for k, entry in self._cache.items():
            if (now - entry.last_access) >= ttl:
                keys_to_delete.append(k)
            else:
                # 一旦遇到未过期的，后面的更“新”，可以停止
                break

        for k in keys_to_delete:
            self._cache.pop(k, None)

    def _evict_lru_locked(self) -> None:
        """
        LRU 淘汰（必须在持锁状态下调用）

        策略：
        - max_entries <= 0：不限制（不建议）
        - 超出上限时，持续 pop 最旧条目（OrderedDict 左侧）
        """
        if self._max_entries <= 0:
            return

        while len(self._cache) > self._max_entries:
            # popitem(last=False) -> 弹出最旧的 LRU 条目
            self._cache.popitem(last=False)

    # ---------------------------------------------------------------------
    # 对外 async context manager：limit()
    # ---------------------------------------------------------------------

    @asynccontextmanager
    async def limit(self, scope: str, name: str, limit: int):
        """
        异步并发限制上下文

        用法：
          async with limiter.limit("agent","my_agent", 3):
              ...

        语义：
        - limit <= 0：不限制，直接透传
        - limit > 0：acquire 对应 semaphore
        """
        if limit <= 0:
            yield
            return

        sem = self._get_semaphore(scope, name, limit)
        await sem.acquire()
        try:
            yield
        finally:
            sem.release()


# 全局共享实例（默认上限与 TTL 给一个合理值，避免缓存无界增长）
global_limiter = ConcurrencyLimiter(max_entries=1024, ttl_seconds=3600)
