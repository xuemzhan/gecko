# tests/plugins/test_storage.py
import pytest
from gecko.plugins.storage.factory import get_storage_by_url
from gecko.plugins.storage.sqlite import SQLiteSessionStorage
from gecko.plugins.storage.utils import parse_storage_url

def test_url_parser():
    scheme, path, params = parse_storage_url("sqlite://./db.sqlite?timeout=10")
    assert scheme == "sqlite"
    assert path == "./db.sqlite"
    assert params["timeout"] == "10"

@pytest.mark.asyncio
async def test_sqlite_storage():
    # 使用 :memory:
    storage = get_storage_by_url("sqlite://:memory:")
    assert isinstance(storage, SQLiteSessionStorage)
    
    await storage.set("sess_1", {"foo": "bar"})
    data = await storage.get("sess_1")
    assert data["foo"] == "bar"
    
    await storage.delete("sess_1")
    assert await storage.get("sess_1") is None