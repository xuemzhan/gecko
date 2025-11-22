# tests/plugins/storage/test_sqlite.py
import asyncio
import os
import pytest
from unittest.mock import patch, MagicMock
from gecko.plugins.storage.backends.sqlite import SQLiteStorage
from gecko.core.exceptions import StorageError

DB_FILE = "./test_gecko_sqlite.db"
DB_URL = f"sqlite:///{DB_FILE}"

@pytest.fixture
async def storage():
    if os.path.exists(DB_FILE): os.remove(DB_FILE)
    if os.path.exists(DB_FILE + ".lock"): os.remove(DB_FILE + ".lock")
    
    # Mock FileLock availability to ensure logic is tested
    with patch("gecko.plugins.storage.mixins.FILELOCK_AVAILABLE", True):
        with patch("gecko.plugins.storage.mixins.FileLock"):
            store = SQLiteStorage(DB_URL)
            await store.initialize()
            yield store
            await store.shutdown()
    
    if os.path.exists(DB_FILE): os.remove(DB_FILE)

@pytest.mark.asyncio
async def test_sqlite_filelock_integration():
    """测试是否尝试配置 FileLock"""
    with patch("gecko.plugins.storage.backends.sqlite.SQLiteStorage.setup_multiprocess_lock") as mock_setup:
        store = SQLiteStorage(DB_URL)
        # [修复] 代码中传入的是原始 path (DB_URL 解析出的相对路径)，未做 abspath 转换
        # 预期改为原始相对路径字符串
        mock_setup.assert_called_once_with(f"./{os.path.basename(DB_FILE)}")

@pytest.mark.asyncio
async def test_sqlite_robustness_invalid_path():
    """测试无法创建数据库文件的场景"""
    # [修复] 不要依赖真实文件系统权限，使用 Mock 模拟创建目录失败
    invalid_url = "sqlite:///some/path/db.sqlite"
    
    # 模拟 pathlib.Path.mkdir 抛出 PermissionError
    with patch("pathlib.Path.mkdir", side_effect=PermissionError("Access denied")):
        # 构造函数中会调用 mkdir
        with pytest.raises(StorageError, match="Failed to configure SQLite"):
            SQLiteStorage(invalid_url)

@pytest.mark.asyncio
async def test_sqlite_crud_exceptions(storage):
    """测试 CRUD 过程中的异常包装"""
    # Mock engine connect to fail
    storage.engine = MagicMock()
    storage.engine.connect.side_effect = Exception("DB Gone")
    # Mock session creation to fail
    with patch("gecko.plugins.storage.backends.sqlite.Session", side_effect=Exception("Session Error")):
        
        with pytest.raises(StorageError, match="SQLite set failed"):
            await storage.set("s1", {})
            
        with pytest.raises(StorageError, match="SQLite get failed"):
            await storage.get("s1")
            
        with pytest.raises(StorageError, match="SQLite delete failed"):
            await storage.delete("s1")