# tests/plugins/storage/test_redis_and_factory.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import sys
from importlib.metadata import EntryPoint

# === Mock Redis Module ===
mock_redis_module = MagicMock()
mock_redis_client = AsyncMock()
mock_redis_module.from_url.return_value = mock_redis_client

# 注入 Mock 到 sys.modules
with patch.dict(sys.modules, {"redis.asyncio": mock_redis_module}):
    from gecko.plugins.storage.backends.redis import RedisStorage

from gecko.plugins.storage.factory import create_storage, _load_from_entry_point
from gecko.core.exceptions import ConfigurationError, StorageError
from gecko.plugins.storage.abc import AbstractStorage

# ================= RedisStorage Tests =================

@pytest.mark.asyncio
async def test_redis_lifecycle_success():
    storage = RedisStorage("redis://localhost:6379/0")
    mock_redis_client.ping.return_value = True
    
    await storage.initialize()
    assert storage.is_initialized
    
    await storage.shutdown()
    assert not storage.is_initialized
    # 验证 client 被置空
    assert storage.client is None # type: ignore

@pytest.mark.asyncio
async def test_redis_connection_failure():
    """测试初始化连接失败，应抛出 StorageError 并清理资源"""
    storage = RedisStorage("redis://localhost:6379/0")
    mock_redis_client.ping.side_effect = Exception("Connection refused")
    mock_redis_client.aclose = AsyncMock() # 确保 shutdown 能调用
    
    with pytest.raises(StorageError, match="Failed to connect to Redis"):
        await storage.initialize()
    
    # 验证是否调用了清理逻辑
    assert storage.client is None # type: ignore
    assert not storage.is_initialized

@pytest.mark.asyncio
async def test_redis_crud_operations():
    storage = RedisStorage("redis://localhost:6379/0")
    storage.client = mock_redis_client # type: ignore
    
    # Set
    await storage.set("s1", {"a": 1}) # type: ignore
    mock_redis_client.setex.assert_awaited()
    
    # Get
    mock_redis_client.get.return_value = '{"a": 1}'
    assert await storage.get("s1") == {"a": 1} # type: ignore
    
    # Delete
    await storage.delete("s1") # type: ignore
    mock_redis_client.delete.assert_awaited()

@pytest.mark.asyncio
async def test_redis_crud_errors():
    """测试操作过程中的异常包装"""
    storage = RedisStorage("redis://localhost:6379/0")
    storage.client = mock_redis_client # type: ignore
    
    mock_redis_client.get.side_effect = Exception("Redis down")
    with pytest.raises(StorageError, match="Redis get failed"):
        await storage.get("s1") # type: ignore
        
    mock_redis_client.setex.side_effect = Exception("ReadOnly")
    with pytest.raises(StorageError, match="Redis set failed"):
        await storage.set("s1", {}) # type: ignore

    mock_redis_client.delete.side_effect = Exception("Error")
    with pytest.raises(StorageError, match="Redis delete failed"):
        await storage.delete("s1") # type: ignore

    # 未初始化调用
    storage.client = None # type: ignore
    with pytest.raises(StorageError, match="not initialized"):
        await storage.get("s1") # type: ignore

# ================= Factory Tests =================

@pytest.mark.asyncio
async def test_factory_entry_point_loading():
    """测试通过 EntryPoint 加载插件"""
    
    # 模拟一个第三方插件类
    class MockPluginStorage(AbstractStorage):
        async def initialize(self): pass
        async def shutdown(self): pass
        
    # 模拟 EntryPoint 对象
    mock_ep = MagicMock(spec=EntryPoint)
    mock_ep.name = "mockdb"
    # load() 返回注册了该类的模块，或者直接副作用注册
    # 这里我们模拟 load() 动作，并手动 patch registry 来模拟注册成功
    mock_ep.load.side_effect = lambda: None 
    
    # Mock entry_points() 返回列表
    with patch("gecko.plugins.storage.factory.entry_points", return_value=[mock_ep]):
        # 还需要 Mock registry.get_storage_class，第一次返回 None，加载后返回类
        with patch("gecko.plugins.storage.registry.get_storage_class", side_effect=[None, MockPluginStorage]):
            
            storage = await create_storage("mockdb://localhost")
            assert isinstance(storage, MockPluginStorage)
            mock_ep.load.assert_called_once()

@pytest.mark.asyncio
async def test_factory_entry_point_failure():
    """测试 EntryPoint 加载失败的情况"""
    mock_ep = MagicMock()
    mock_ep.name = "faildb"
    mock_ep.load.side_effect = Exception("Load error")
    
    with patch("gecko.plugins.storage.factory.entry_points", return_value=[mock_ep]):
        # 因为加载失败，registry 依然查不到
        with patch("gecko.plugins.storage.registry.get_storage_class", return_value=None):
            with pytest.raises(ConfigurationError, match="Unknown storage scheme"):
                await create_storage("faildb://")

@pytest.mark.asyncio
async def test_factory_builtin_loading():
    """测试内置模块加载 (SQLite)"""
    # 使用 :memory: 确保不产生文件
    storage = await create_storage("sqlite:///:memory:")
    assert storage.__class__.__name__ == "SQLiteStorage"
    await storage.shutdown()

@pytest.mark.asyncio
async def test_factory_invalid_scheme():
    with pytest.raises(ConfigurationError, match="Invalid storage URL"):
        await create_storage("not_a_url")

@pytest.mark.asyncio
async def test_factory_module_import_error():
    """测试内置模块导入失败 (模拟依赖缺失)"""
    # 模拟导入 sqlite 模块时抛出 ImportError
    # [修复] 调整 Context Manager 嵌套顺序，避免 patch 冲突
    with patch.dict(sys.modules, {"gecko.plugins.storage.backends.sqlite": None}): 
        # 先清空 registry
        with patch("gecko.plugins.storage.registry._STORAGE_REGISTRY", {}):
            # 再 patch import_module，且仅针对 create_storage 调用期间
            with patch("importlib.import_module", side_effect=ImportError("No module")):
                with pytest.raises(ConfigurationError, match="Failed to load built-in backend"):
                    await create_storage("sqlite:///:memory:")