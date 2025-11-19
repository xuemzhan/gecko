# tests/plugins/test_storage.py
import pytest
import os
from gecko.plugins.storage.sqlite import SQLiteSessionStorage
from gecko.plugins.storage.factory import get_storage_by_url

@pytest.mark.asyncio
async def test_sqlite_storage():
    db_path = "./test_storage.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    url = f"sqlite://{db_path}"
    storage = get_storage_by_url(url, required="session")
    
    # Test Set
    session_id = "user_123"
    state = {"messages": [{"role": "user", "content": "hello"}], "metadata": {"tokens": 10}}
    await storage.set(session_id, state)
    
    # Test Get
    retrieved = await storage.get(session_id)
    assert retrieved is not None
    assert retrieved["messages"][0]["content"] == "hello"
    
    # Test Update
    state["messages"].append({"role": "assistant", "content": "hi"})
    await storage.set(session_id, state)
    retrieved_updated = await storage.get(session_id)
    assert len(retrieved_updated["messages"]) == 2
    
    # Test Delete
    await storage.delete(session_id)
    assert await storage.get(session_id) is None
    
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)

@pytest.mark.asyncio
async def test_storage_factory_error():
    with pytest.raises(ValueError, match="无效的存储 URL"):
        get_storage_by_url("invalid_url")
        
    with pytest.raises(ValueError, match="未找到存储实现"):
        get_storage_by_url("unknown://db")
        
# 追加到 tests/plugins/test_storage.py
@pytest.mark.asyncio
async def test_lancedb_storage():
    try:
        import lancedb
    except ImportError:
        pytest.skip("lancedb not installed")

    import shutil
    db_path = "./test_lancedb"
    # 清理旧数据
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
        
    url = f"lancedb://{db_path}"
    # 注意：lancedb 实现了 VectorInterface，所以这里 required="vector"
    storage = get_storage_by_url(url, required="vector", collection_name="test_docs")
    
    # Test Upsert
    docs = [
        {"id": "1", "text": "apple", "embedding": [0.1]*1536, "metadata": {"type": "fruit"}},
        {"id": "2", "text": "banana", "embedding": [0.2]*1536, "metadata": {"type": "fruit"}}
    ]
    await storage.upsert(docs)
    
    # Test Search
    # 模拟搜索向量
    results = await storage.search([0.1]*1536, top_k=1)
    assert len(results) == 1
    assert results[0]["text"] == "apple"
    
    # Cleanup
    if os.path.exists(db_path):
        shutil.rmtree(db_path)