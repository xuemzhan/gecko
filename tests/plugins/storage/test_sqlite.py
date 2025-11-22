# tests/plugins/storage/test_sqlite.py
import asyncio
import os
import pytest
from gecko.plugins.storage.backends.sqlite import SQLiteStorage

DB_FILE = "./test_gecko.db"
DB_URL = f"sqlite:///{DB_FILE}"

@pytest.fixture
async def storage():
    # 清理旧文件
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
        
    store = SQLiteStorage(DB_URL)
    await store.initialize()
    yield store
    await store.shutdown()
    
    # 清理
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

@pytest.mark.asyncio
async def test_lifecycle(storage):
    """测试初始化和关闭"""
    assert storage.is_initialized
    # 检查文件是否创建
    assert os.path.exists(DB_FILE)

@pytest.mark.asyncio
async def test_crud_operations(storage):
    """测试基本的增删改查"""
    session_id = "sess_001"
    data = {"user": "test", "count": 1}

    # Create/Set
    await storage.set(session_id, data)
    
    # Read/Get
    loaded = await storage.get(session_id)
    assert loaded == data
    
    # Update
    data["count"] = 2
    await storage.set(session_id, data)
    loaded = await storage.get(session_id)
    assert loaded["count"] == 2
    
    # Delete
    await storage.delete(session_id)
    loaded = await storage.get(session_id)
    assert loaded is None

@pytest.mark.asyncio
async def test_concurrency_stress(storage):
    """
    并发压力测试
    
    验证 WAL 模式 + AtomicWriteMixin 是否能处理高并发写入
    而不会抛出 'database is locked'
    """
    concurrency = 20  # 增加并发数
    session_id = "sess_concurrent"
    
    # 初始化
    await storage.set(session_id, {"counter": 0})

    async def increment():
        # 模拟读-改-写业务逻辑
        # 注意：这里的逻辑本身在应用层不是原子的（除非加分布式锁）
        # 但我们要测的是 DB 层不会报错崩溃
        for _ in range(5):
            # 随机小延迟模拟真实 IO 间隔
            await asyncio.sleep(0.001)
            # 我们只测试 set 不会崩溃，不验证最终计数器的原子性(那属于业务层锁的问题)
            # 为了简单，直接覆盖写入
            await storage.set(session_id, {"counter": 999})
            # 同时也读
            await storage.get(session_id)

    # 启动大量并发任务
    tasks = [increment() for _ in range(concurrency)]
    
    # 应该无异常完成
    await asyncio.gather(*tasks)
    
    # 验证文件未损坏
    final = await storage.get(session_id)
    assert final is not None
    assert final["counter"] == 999