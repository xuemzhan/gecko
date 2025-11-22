# tests/plugins/storage/test_redis_and_factory.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import sys

# === Mock Redis Module ===
# 在导入真实模块前进行 Mock，确保测试环境无需 Redis
mock_redis_module = MagicMock()
mock_redis_client = AsyncMock()
mock_redis_module.from_url.return_value = mock_redis_client

# 将 Mock 注入 sys.modules
with patch.dict(sys.modules, {"redis.asyncio": mock_redis_module}):
    from gecko.plugins.storage.backends.redis import RedisStorage

from gecko.plugins.storage.factory import create_storage
from gecko.core.exceptions import ConfigurationError

@pytest.mark.asyncio
async def test_redis_lifecycle():
    """测试 Redis 初始化和关闭流程"""
    url = "redis://localhost:6379/0"
    storage = RedisStorage(url)
    
    # Mock ping success
    mock_redis_client.ping.return_value = True
    
    # Initialize
    await storage.initialize()
    assert storage.is_initialized
    mock_redis_module.from_url.assert_called_once()
    mock_redis_client.ping.assert_awaited_once()
    
    # Shutdown
    await storage.shutdown()
    assert not storage.is_initialized
    mock_redis_client.aclose.assert_awaited_once()

@pytest.mark.asyncio
async def test_redis_operations():
    """测试 Redis CRUD"""
    storage = RedisStorage("redis://localhost:6379/0")
    storage.client = mock_redis_client # type: ignore # 手动注入 Mock client
    
    # Set
    await storage.set("s1", {"a": 1}) # type: ignore
    mock_redis_client.setex.assert_awaited_with(
        "gecko:session:s1", 
        3600*24*7, 
        '{"a": 1}'
    )
    
    # Get
    mock_redis_client.get.return_value = '{"a": 1}'
    data = await storage.get("s1") # type: ignore
    assert data == {"a": 1}
    
    # Delete
    await storage.delete("s1") # type: ignore
    mock_redis_client.delete.assert_awaited_with("gecko:session:s1")

@pytest.mark.asyncio
async def test_factory_sqlite():
    """测试工厂加载 SQLite (这是内置支持的，不需要 Mock import)"""
    url = "sqlite:///:memory:"
    storage = await create_storage(url)
    
    assert storage.is_initialized
    assert storage.__class__.__name__ == "SQLiteStorage"
    
    await storage.shutdown()

@pytest.mark.asyncio
async def test_factory_invalid_scheme():
    """测试无效 Scheme"""
    with pytest.raises(ConfigurationError, match="Unknown storage scheme"):
        await create_storage("invalid://localhost")