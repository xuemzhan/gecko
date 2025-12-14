# tests/core/test_limits.py

import pytest
import time

from gecko.core.limits import ConcurrencyLimiter

@pytest.mark.asyncio
async def test_limiter_evict_by_max_entries():
    """
    验证：超过 max_entries 后会发生 LRU 淘汰，缓存不会无界增长。
    """
    limiter = ConcurrencyLimiter(max_entries=2, ttl_seconds=3600)

    # 访问三个不同 key，会超过 max_entries
    sem1 = limiter._get_semaphore("agent", "a", 1)
    sem2 = limiter._get_semaphore("agent", "b", 1)
    sem3 = limiter._get_semaphore("agent", "c", 1)

    # 缓存最多保留 2 个
    assert limiter._cache_size() <= 2

    # 已经拿到的 sem1/sem2/sem3 对象仍可用（淘汰只影响缓存，不影响引用）
    await sem1.acquire()
    sem1.release()

@pytest.mark.asyncio
async def test_limiter_evict_by_ttl(monkeypatch):
    """
    验证：TTL 到期后会淘汰旧 key。
    """
    limiter = ConcurrencyLimiter(max_entries=100, ttl_seconds=1)

    base = time.time()
    monkeypatch.setattr(time, "time", lambda: base)

    limiter._get_semaphore("agent", "a", 1)
    assert limiter._cache_size() == 1

    # 时间前进 2 秒，超过 TTL
    monkeypatch.setattr(time, "time", lambda: base + 2)

    limiter._get_semaphore("agent", "b", 1)  # 触发访问 -> 淘汰执行
    assert limiter._cache_size() <= 1  # a 被淘汰，仅保留 b
