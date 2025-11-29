# tests/plugins/storage/test_storage_core.py
import asyncio
import threading
import time
import pytest
from unittest.mock import MagicMock, patch

from gecko.plugins.storage.abc import AbstractStorage
from gecko.plugins.storage.mixins import (
    ThreadOffloadMixin, 
    AtomicWriteMixin, 
    JSONSerializerMixin
)

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
    
    # 执行耗时操作
    start_time = time.time()
    worker_thread_id = await storage._run_sync(storage.sync_slow_operation, 0.05)
    duration = time.time() - start_time
    
    # 验证
    assert worker_thread_id != main_thread_id, "操作应该在不同的线程中执行"
    assert duration >= 0.05, "操作应该确实执行了耗时逻辑"

@pytest.mark.asyncio
async def test_atomic_write_mixin_basic():
    """测试基础原子写锁（进程内 asyncio.Lock）"""
    storage = MockStorage()
    
    # 验证锁的懒加载
    assert storage._write_lock is None
    lock = storage.write_lock
    assert isinstance(lock, asyncio.Lock)
    assert storage._write_lock is not None
    
    # 验证上下文管理器
    async with storage.write_guard():
        assert storage.write_lock.locked()
    assert not storage.write_lock.locked()

@pytest.mark.asyncio
async def test_atomic_write_mixin_filelock_logic():
    """
    测试 FileLock 逻辑 (新架构)
    验证 write_guard 只处理协程锁，file_lock_guard 处理文件锁
    """
    with patch("gecko.plugins.storage.mixins.FILELOCK_AVAILABLE", True):
        with patch("gecko.plugins.storage.mixins.FileLock") as MockFileLock:
            mock_file_lock_instance = MagicMock()
            # 模拟 Context Manager (__enter__ / __exit__)
            mock_file_lock_instance.__enter__ = MagicMock()
            mock_file_lock_instance.__exit__ = MagicMock()
            MockFileLock.return_value = mock_file_lock_instance
            
            storage = MockStorage()
            storage.setup_multiprocess_lock("/tmp/test.db")
            
            # 验证 FileLock 初始化
            MockFileLock.assert_called_with("/tmp/test.db.lock")
            assert storage._file_lock == mock_file_lock_instance
            
            # 1. 验证 write_guard (Async) 
            # 预期：只获取 asyncio lock，不操作 FileLock
            async with storage.write_guard():
                assert storage.write_lock.locked()
                mock_file_lock_instance.acquire.assert_not_called()
                mock_file_lock_instance.__enter__.assert_not_called()
            
            # 2. 验证 file_lock_guard (Sync)
            # 预期：调用 FileLock 的上下文管理器
            with storage.file_lock_guard():
                mock_file_lock_instance.__enter__.assert_called_once()
            
            mock_file_lock_instance.__exit__.assert_called_once()

@pytest.mark.asyncio
async def test_atomic_write_mixin_filelock_missing():
    """测试 FileLock 未安装的情况"""
    with patch("gecko.plugins.storage.mixins.FILELOCK_AVAILABLE", False):
        with patch("gecko.plugins.storage.mixins.logger") as mock_logger:
            storage = MockStorage()
            storage.setup_multiprocess_lock("/tmp/test.db")
            
            # 验证发出了警告且未初始化锁
            mock_logger.warning.assert_called()
            assert "filelock module not installed" in mock_logger.warning.call_args[0][0]
            assert storage._file_lock is None
            
            # 验证 file_lock_guard 仍然可用（空操作）
            with storage.file_lock_guard():
                pass

@pytest.mark.asyncio
async def test_atomic_write_mixin_filelock_init_error():
    """测试 FileLock 初始化异常"""
    with patch("gecko.plugins.storage.mixins.FILELOCK_AVAILABLE", True):
        with patch("gecko.plugins.storage.mixins.FileLock", side_effect=Exception("Perm Error")):
            with patch("gecko.plugins.storage.mixins.logger") as mock_logger:
                storage = MockStorage()
                storage.setup_multiprocess_lock("/root/test.db")
                
                mock_logger.error.assert_called()
                assert "Failed to initialize FileLock" in mock_logger.error.call_args[0][0]

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
    
    # 错误处理
    with pytest.raises(Exception): 
        storage._serialize({"set": {1, 2}}) # Set 无法 JSON 序列化
    
    # 反序列化错误测试 (日志记录但不抛出，返回 None)
    assert storage._deserialize("{invalid_json") is None

@pytest.mark.asyncio
async def test_abstract_lifecycle():
    """测试抽象生命周期"""
    storage = MockStorage()
    assert not storage.is_initialized
    
    async with storage as s:
        assert s.is_initialized
        assert s is storage
    
    assert not storage.is_initialized

@pytest.mark.asyncio
async def test_atomic_mixin_filelock_behavior():
    """[New] 验证 AtomicWriteMixin 在 setup_multiprocess_lock 后的行为"""
    
    # 模拟 filelock 库已安装
    with patch("gecko.plugins.storage.mixins.FILELOCK_AVAILABLE", True):
        # Mock FileLock 类
        with patch("gecko.plugins.storage.mixins.FileLock") as MockFileLock:
            mock_lock_instance = MagicMock()
            MockFileLock.return_value = mock_lock_instance
            
            storage = MockStorage()
            # 1. 设置锁路径
            storage.setup_multiprocess_lock("./test.db")
            
            # 验证 FileLock 被实例化
            MockFileLock.assert_called_with("./test.db.lock")
            
            # 2. 测试 file_lock_guard (同步上下文)
            # 必须调用 acquire/release (或者 __enter__/__exit__)
            with storage.file_lock_guard():
                pass
            
            mock_lock_instance.__enter__.assert_called()
            mock_lock_instance.__exit__.assert_called()

@pytest.mark.asyncio
async def test_atomic_mixin_fallback_without_filelock():
    """[New] 验证 filelock 未安装时的降级行为"""
    
    with patch("gecko.plugins.storage.mixins.FILELOCK_AVAILABLE", False):
        with patch("gecko.plugins.storage.mixins.logger") as mock_logger:
            storage = MockStorage()
            storage.setup_multiprocess_lock("./test.db")
            
            # 1. 验证日志警告
            mock_logger.warning.assert_called()
            assert storage._file_lock is None
            
            # 2. 验证 file_lock_guard 不报错且无操作
            try:
                with storage.file_lock_guard():
                    pass # Should execute safely
            except Exception as e:
                pytest.fail(f"file_lock_guard raised exception in fallback mode: {e}")
