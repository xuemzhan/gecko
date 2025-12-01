import threading
import time
import pytest
import os
from unittest.mock import MagicMock
from gecko.plugins.storage.backends.sqlite import SQLiteStorage

DB_FILE = "test_opt.db"
DB_URL = f"sqlite:///./{DB_FILE}"

@pytest.fixture
async def optimized_store():
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    store = SQLiteStorage(DB_URL)
    await store.initialize()
    yield store
    await store.shutdown()
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)

@pytest.mark.asyncio
async def test_serialization_offloading(optimized_store):
    """
    验证 JSON 序列化确实是在子线程中执行的，而不是主线程。
    """
    main_thread_id = threading.get_ident()
    worker_thread_id = None

    # Hook _serialize 方法来捕获执行时的线程 ID
    original_serialize = optimized_store._serialize
    
    def spy_serialize(data):
        nonlocal worker_thread_id
        worker_thread_id = threading.get_ident()
        # 模拟大对象序列化耗时
        time.sleep(0.01) 
        return original_serialize(data)
    
    # 替换方法
    optimized_store._serialize = spy_serialize
    
    # 执行 set 操作
    await optimized_store.set("test_offload", {"key": "value"})
    
    # 验证
    assert worker_thread_id is not None, "_serialize 应该被调用"
    assert worker_thread_id != main_thread_id, "_serialize 应该在 Worker 线程中执行，而非主线程"

@pytest.mark.asyncio
async def test_deserialization_offloading(optimized_store):
    """
    验证 JSON 反序列化是在子线程中执行的。
    """
    # 先写入数据
    await optimized_store.set("test_offload_get", {"key": "value"})
    
    main_thread_id = threading.get_ident()
    worker_thread_id = None
    
    original_deserialize = optimized_store._deserialize
    
    def spy_deserialize(data):
        nonlocal worker_thread_id
        worker_thread_id = threading.get_ident()
        return original_deserialize(data)
        
    optimized_store._deserialize = spy_deserialize
    
    # 执行 get 操作
    await optimized_store.get("test_offload_get")
    
    assert worker_thread_id is not None
    assert worker_thread_id != main_thread_id, "_deserialize 应该在 Worker 线程中执行"