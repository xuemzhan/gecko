# tests/plugins/storage/test_storage_core.py
import asyncio
import threading
import time
import pytest
from gecko.plugins.storage.abc import AbstractStorage
from gecko.plugins.storage.mixins import ThreadOffloadMixin, AtomicWriteMixin, JSONSerializerMixin

# === Mocks & Helpers ===

class MockStorage(AbstractStorage, ThreadOffloadMixin, AtomicWriteMixin, JSONSerializerMixin):
    """用于测试 Mixin 的模拟存储类"""
    def __init__(self):
        super().__init__("mock://")
        
    async def initialize(self):
        self._is_initialized = True

    async def shutdown(self):
        self._is_initialized = False

    def sync_slow_operation(self, seconds: float) -> int:
        """模拟同步阻塞操作"""
        time.sleep(seconds)
        return threading.get_ident()

# === Tests ===

@pytest.mark.asyncio
async def test_thread_offload_mixin():
    """测试线程卸载：确保操作不在主线程执行"""
    storage = MockStorage()
    main_thread_id = threading.get_ident()
    
    # 执行耗时操作 (0.1s)
    start_time = time.time()
    worker_thread_id = await storage._run_sync(storage.sync_slow_operation, 0.1)
    duration = time.time() - start_time
    
    # 验证
    assert worker_thread_id != main_thread_id, "操作应该在不同的线程中执行"
    assert duration >= 0.1, "操作应该确实执行了耗时逻辑"

@pytest.mark.asyncio
async def test_atomic_write_mixin():
    """测试原子写锁：确保并发写入串行化"""
    storage = MockStorage()
    counter = 0
    
    async def unsafe_increment():
        nonlocal counter
        # 模拟读改写竞态
        temp = counter
        await asyncio.sleep(0.01) 
        counter = temp + 1

    async def safe_increment():
        nonlocal counter
        async with storage.write_guard():
            # 即使内部有 await，锁也能保证临界区互斥
            temp = counter
            await asyncio.sleep(0.01)
            counter = temp + 1

    # 1. 无锁测试 (预期失败或产生竞态，但在 asyncio 单线程模型下 yield 才会切换)
    # 为了演示锁的作用，我们主要验证锁是否能够被获取和释放
    async with storage.write_guard():
        assert storage._write_lock.locked() # type: ignore
    assert not storage._write_lock.locked() # type: ignore
    
    # 2. 并发测试
    tasks = [safe_increment() for _ in range(5)]
    await asyncio.gather(*tasks)
    assert counter == 5, "所有增量操作应该都被正确执行"

def test_json_serializer_mixin():
    """测试 JSON 序列化"""
    storage = MockStorage()
    data = {"key": "value", "中文": "测试"}
    
    # 序列化
    json_str = storage._serialize(data)
    assert '"中文": "测试"' in json_str, "应该保留非 ASCII 字符"
    
    # 反序列化
    restored = storage._deserialize(json_str)
    assert restored == data
    
    # 边缘情况
    assert storage._deserialize(None) is None
    assert storage._deserialize("") is None

@pytest.mark.asyncio
async def test_abstract_lifecycle():
    """测试生命周期上下文管理器"""
    storage = MockStorage()
    assert not storage.is_initialized
    
    async with storage as s:
        assert s.is_initialized
        assert s is storage
    
    assert not storage.is_initialized