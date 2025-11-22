# tests/plugins/storage/test_vector_backends.py
import asyncio
import os
import shutil
import pytest
import threading
import uuid

# 尝试导入依赖，如果没有则跳过测试
try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    import lancedb
    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False

from gecko.plugins.storage.backends.chroma import ChromaStorage
from gecko.plugins.storage.backends.lancedb import LanceDBStorage

# === Fixtures ===

@pytest.fixture
async def chroma_store(tmp_path):
    """
    修复：使用 tmp_path (pytest 内置 fixture) 
    为每个测试用例生成唯一的临时目录，避免 SQLite 文件锁冲突 (Error 1032)。
    """
    # tmp_path 是一个 pathlib.Path 对象
    # Chroma 需要字符串路径
    db_path = tmp_path / "chroma_db"
    path_str = str(db_path)
        
    store = ChromaStorage(f"chroma://{path_str}")
    await store.initialize()
    yield store
    await store.shutdown()
    
    # Pytest 会自动清理 tmp_path，通常不需要手动 rmtree
    # 但为了确保资源释放，手动显式清理也是可以的（需忽略错误）
    if os.path.exists(path_str):
        try:
            shutil.rmtree(path_str)
        except Exception:
            pass

@pytest.fixture
async def lance_store(tmp_path):
    """
    同样为 LanceDB 使用唯一路径以确保隔离
    """
    db_path = tmp_path / "lance_db"
    path_str = str(db_path)
        
    store = LanceDBStorage(f"lancedb://{path_str}")
    await store.initialize()
    yield store
    await store.shutdown()
    
    if os.path.exists(path_str):
        try:
            shutil.rmtree(path_str)
        except Exception:
            pass

# === Helper ===

def get_dummy_docs():
    return [
        {
            "id": "vec1",
            "embedding": [0.1, 0.2, 0.3],
            "text": "Hello World",
            "metadata": {"source": "test"}
        },
        {
            "id": "vec2",
            "embedding": [0.9, 0.8, 0.7],
            "text": "Gecko AI",
            "metadata": {"source": "doc"}
        }
    ]

# === Chroma Tests ===

@pytest.mark.skipif(not CHROMA_AVAILABLE, reason="chromadb not installed")
@pytest.mark.asyncio
async def test_chroma_vector_ops(chroma_store):
    docs = get_dummy_docs()
    
    # Test Non-blocking Upsert
    # 验证方法不报错即可，内部使用了 ThreadOffload
    await chroma_store.upsert(docs)
    
    # Test Search
    query = [0.1, 0.2, 0.3] # exact match vec1
    results = await chroma_store.search(query, top_k=1)
    
    assert len(results) == 1
    assert results[0]["id"] == "vec1"
    assert results[0]["text"] == "Hello World"
    # Cosine distance of identical vectors is 0, score = 1 - 0 = 1
    assert results[0]["score"] > 0.99

@pytest.mark.skipif(not CHROMA_AVAILABLE, reason="chromadb not installed")
@pytest.mark.asyncio
async def test_chroma_session_ops(chroma_store):
    """Chroma 也实现了 Session 接口"""
    sess_id = "s1"
    state = {"user": "Alice", "age": 30} # 简单类型
    
    await chroma_store.set(sess_id, state)
    retrieved = await chroma_store.get(sess_id)
    
    assert retrieved["user"] == "Alice"
    # 我们在实现中将所有值转为了 str 存入 metadata，或者序列化存入 document
    # 取决于最终采用的实现。最新的实现是 JSON 序列化存 document，应该保持类型。
    # 检查下 ChromaStorage.set 的实现：它使用了 JSONSerializerMixin 存入 document
    # 所以读取出来应该是原始类型
    assert retrieved["age"] == 30

# === LanceDB Tests ===

@pytest.mark.skipif(not LANCEDB_AVAILABLE, reason="lancedb not installed")
@pytest.mark.asyncio
async def test_lance_vector_ops(lance_store):
    docs = get_dummy_docs()
    
    # Test Auto-create table + Upsert
    await lance_store.upsert(docs)
    
    # Test Append
    more_docs = [{
        "id": "vec3",
        "embedding": [0.5, 0.5, 0.5],
        "text": "Append",
        "metadata": {}
    }]
    await lance_store.upsert(more_docs)
    
    # Test Search
    query = [0.9, 0.8, 0.7] # match vec2
    results = await lance_store.search(query, top_k=1)
    
    assert len(results) == 1
    assert results[0]["id"] == "vec2"
    assert results[0]["text"] == "Gecko AI"