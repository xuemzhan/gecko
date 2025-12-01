import pytest
import time
import os
from gecko.plugins.storage.backends.sqlite import SQLiteStorage

DB_FILE = "test_ttl.db"
DB_URL = f"sqlite:///./{DB_FILE}"

@pytest.fixture
async def ttl_store():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    store = SQLiteStorage(DB_URL)
    await store.initialize()
    yield store
    await store.shutdown()
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

@pytest.mark.asyncio
async def test_sqlite_native_ttl_cleanup(ttl_store):
    """验证 SQLite 的 expire_at 字段设置和过期清理逻辑"""
    
    # 1. 插入一个不过期的 Session (TTL = None)
    await ttl_store.set("session_forever", {"data": 1, "metadata": {"ttl": None}})
    
    # 2. 插入一个即将过期的 Session (TTL = 0.1s)
    # 我们模拟 updated_at 是当前时间
    await ttl_store.set("session_expiring", {
        "data": 2, 
        "metadata": {
            "ttl": 0.1, 
            "updated_at": time.time()
        }
    })
    
    # 3. 验证 expire_at 字段被正确写入
    # 我们通过 get 拿不到 expire_at (因为它不在 state json 里)，所以需要 hack 一下直接查库
    # 或者我们信任 cleanup_expired 的结果
    
    # 立即清理，应该删除 0 行 (还未过期)
    deleted = await ttl_store.cleanup_expired()
    assert deleted == 0
    
    # 等待过期
    time.sleep(0.2)
    
    # 4. 再次清理，应该删除 1 行
    deleted = await ttl_store.cleanup_expired()
    assert deleted == 1
    
    # 5. 验证结果
    assert await ttl_store.get("session_forever") is not None
    assert await ttl_store.get("session_expiring") is None