# tests/plugins/storage/test_sqlite.py
"""
SQLite 存储后端测试 (Updated for Schema v2)

覆盖范围：
1. 初始化与连接
2. 增删改查 (VectorInterface / SessionInterface)
3. 鲁棒性测试 (非法路径、权限错误)
4. 并发与文件锁集成测试
5. [New] Schema 完整性检查 (expire_at 字段)
"""
import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from sqlalchemy import inspect
from gecko.plugins.storage.backends.sqlite import SQLiteStorage
from gecko.core.exceptions import StorageError

DB_FILE = "test_gecko_sqlite.db"
DB_URL = f"sqlite:///./{DB_FILE}"

@pytest.fixture
async def sqlite_store():
    """Fixture: 创建并自动清理 SQLiteStorage (集成测试用)"""
    # 确保环境干净
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    if os.path.exists(f"{DB_FILE}.lock"):
        try:
            os.remove(f"{DB_FILE}.lock")
        except OSError:
            pass
            
    store = SQLiteStorage(DB_URL)
    await store.initialize()
    yield store
    await store.shutdown()
    
    # 清理
    if os.path.exists(DB_FILE):
        os.remove(DB_FILE)
    if os.path.exists(f"{DB_FILE}.lock"):
        try:
            os.remove(f"{DB_FILE}.lock")
        except OSError:
            pass

@pytest.mark.asyncio
async def test_sqlite_filelock_integration():
    """测试是否尝试配置 FileLock"""
    with patch("gecko.plugins.storage.backends.sqlite.Path") as MockPath:
        mock_instance = MockPath.return_value
        type(mock_instance).parent = PropertyMock(return_value=MagicMock())
        mock_instance.parent.exists.return_value = True 

        with patch("gecko.plugins.storage.backends.sqlite.SQLiteStorage.setup_multiprocess_lock") as mock_setup:
            store = SQLiteStorage(DB_URL)
            assert mock_setup.call_count == 1
            mock_setup.assert_called_with(store.db_path) # type: ignore

@pytest.mark.asyncio
async def test_sqlite_robustness_invalid_path():
    """测试无法创建数据库文件的场景"""
    invalid_url = "sqlite:///some/path/db.sqlite"

    with patch("gecko.plugins.storage.backends.sqlite.Path") as MockPath:
        mock_path_instance = MockPath.return_value
        mock_parent = MagicMock()
        type(mock_path_instance).parent = PropertyMock(return_value=mock_parent)
        mock_parent.exists.return_value = False
        mock_parent.mkdir.side_effect = PermissionError("Access denied")

        with pytest.raises(StorageError, match="Failed to configure SQLite"):
            SQLiteStorage(invalid_url)

@pytest.mark.asyncio
async def test_sqlite_filelock_setup():
    """验证 SQLite 初始化时配置了文件锁"""
    with patch("gecko.plugins.storage.mixins.FILELOCK_AVAILABLE", True):
        with patch("gecko.plugins.storage.mixins.FileLock") as MockFileLock:
            with patch("gecko.plugins.storage.backends.sqlite.Path") as MockPath:
                mock_instance = MockPath.return_value
                type(mock_instance).parent = PropertyMock(return_value=MagicMock())
                mock_instance.parent.exists.return_value = True
                
                url = "sqlite:///./test_lock.db"
                store = SQLiteStorage(url)
    
                assert MockFileLock.call_count == 1
                call_args = MockFileLock.call_args
                assert call_args[0][0].endswith(".lock")
                assert store._file_lock is not None # type: ignore

@pytest.mark.asyncio
async def test_sqlite_crud_operations(sqlite_store):
    """测试基本的增删改查 (集成测试)"""
    session_id = "sess_001"
    data = {"key": "value", "num": 123}
    
    # Set
    await sqlite_store.set(session_id, data)
    
    # Get
    retrieved = await sqlite_store.get(session_id)
    assert retrieved == data
    
    # Update
    data["new"] = "field"
    await sqlite_store.set(session_id, data)
    retrieved_updated = await sqlite_store.get(session_id)
    assert retrieved_updated == data
    
    # Delete
    await sqlite_store.delete(session_id)
    retrieved_deleted = await sqlite_store.get(session_id)
    assert retrieved_deleted is None

@pytest.mark.asyncio
async def test_sqlite_memory_mode():
    """测试内存模式"""
    store = SQLiteStorage("sqlite:///:memory:")
    await store.initialize()
    
    assert store._file_lock is None # type: ignore
    
    await store.set("mem_sess", {"a": 1}) # type: ignore
    res = await store.get("mem_sess") # type: ignore
    assert res == {"a": 1}
    
    await store.shutdown()

@pytest.mark.asyncio
async def test_sqlite_schema_integrity(sqlite_store):
    """
    [New] 验证数据库表结构是否包含新增的 expire_at 字段
    确保 SQLModel 表定义更新已生效
    """
    def _inspect_columns():
        inspector = inspect(sqlite_store.engine)
        columns = inspector.get_columns("gecko_sessions")
        return {col["name"] for col in columns}

    # 在子线程中执行检查
    column_names = await sqlite_store._run_sync(_inspect_columns)
    
    assert "session_id" in column_names
    assert "state_json" in column_names
    assert "expire_at" in column_names, "Missing new 'expire_at' column for TTL support"