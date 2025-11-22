# tests/plugins/storage/test_vector_backends.py
import asyncio
import os
import shutil
import pytest
from unittest.mock import patch, MagicMock

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
from gecko.core.exceptions import StorageError

# === Fixtures ===

@pytest.fixture
async def chroma_store(tmp_path):
    db_path = tmp_path / "chroma_db"
    path_str = str(db_path)
    store = ChromaStorage(f"chroma://{path_str}")
    await store.initialize()
    yield store
    await store.shutdown()

@pytest.fixture
async def lance_store(tmp_path):
    db_path = tmp_path / "lance_db"
    path_str = str(db_path)
    store = LanceDBStorage(f"lancedb://{path_str}")
    await store.initialize()
    yield store
    await store.shutdown()

# === Test Data ===
def get_docs():
    return [
        {"id": "1", "embedding": [0.1, 0.1], "text": "A", "metadata": {"type": "news"}},
        {"id": "2", "embedding": [0.9, 0.9], "text": "B", "metadata": {"type": "blog"}},
        {"id": "3", "embedding": [0.1, 0.2], "text": "C", "metadata": None} # Test None metadata
    ]

# === Chroma Tests ===

@pytest.mark.skipif(not CHROMA_AVAILABLE, reason="chromadb missing")
@pytest.mark.asyncio
async def test_chroma_robustness(chroma_store):
    """测试健壮性：Metadata 为 None"""
    docs = get_docs()
    # 应该能够处理 metadata=None 的情况 (转为 {})
    await chroma_store.upsert(docs)
    
    results = await chroma_store.search([0.1, 0.1], top_k=3)
    assert len(results) == 3
    # 验证 metadata=None 被转为了 {}
    doc3 = next(r for r in results if r["id"] == "3")
    assert doc3["metadata"] == {}

@pytest.mark.skipif(not CHROMA_AVAILABLE, reason="chromadb missing")
@pytest.mark.asyncio
async def test_chroma_filtering(chroma_store):
    """测试元数据过滤"""
    await chroma_store.upsert(get_docs())
    
    # Filter: type=news
    res = await chroma_store.search([0.1, 0.1], top_k=5, filters={"type": "news"})
    assert len(res) == 1
    assert res[0]["id"] == "1"
    
    # Filter: non-exist
    res = await chroma_store.search([0.1, 0.1], top_k=5, filters={"type": "404"})
    assert len(res) == 0

@pytest.mark.skipif(not CHROMA_AVAILABLE, reason="chromadb missing")
@pytest.mark.asyncio
async def test_chroma_exceptions(chroma_store):
    """测试异常包装"""
    # Mock 内部 collection 抛出异常
    chroma_store.vector_col = MagicMock()
    chroma_store.vector_col.query.side_effect = Exception("DB Crash")
    
    with pytest.raises(StorageError, match="Chroma search failed"):
        await chroma_store.search([0.1, 0.1])

# === LanceDB Tests ===

@pytest.mark.skipif(not LANCEDB_AVAILABLE, reason="lancedb missing")
@pytest.mark.asyncio
async def test_lance_robustness(lance_store):
    """测试健壮性：Metadata 为 None"""
    docs = get_docs()
    await lance_store.upsert(docs)
    
    results = await lance_store.search([0.1, 0.1], top_k=3)
    assert len(results) == 3
    doc3 = next(r for r in results if r["id"] == "3")
    
    # [修复] LanceDB 会将缺失字段填充为 None (Schema evolution)
    # Doc1/2 有 'type' 字段，Doc3 没有，所以 Doc3['metadata'] 变为 {'type': None}
    # 我们验证它不是 None 且是一个字典即可，或者验证包含 None
    assert isinstance(doc3["metadata"], dict)
    # 如果有字段，值应为 None
    if doc3["metadata"]:
        assert all(v is None for v in doc3["metadata"].values())

@pytest.mark.skipif(not LANCEDB_AVAILABLE, reason="lancedb missing")
@pytest.mark.asyncio
async def test_lance_filtering(lance_store):
    """测试元数据过滤"""
    await lance_store.upsert(get_docs())
    
    # Filter string
    res = await lance_store.search([0.1, 0.1], top_k=5, filters={"type": "news"})
    assert len(res) == 1
    assert res[0]["id"] == "1"
    
    # 复杂场景：LanceDB SQL 构建测试 (模拟)
    # 注意：真实 lancedb 过滤需要 where 子句
    # 我们已经在 upsert 时验证了写入，现在验证查询返回
    
@pytest.mark.skipif(not LANCEDB_AVAILABLE, reason="lancedb missing")
@pytest.mark.asyncio
async def test_lance_exceptions(lance_store):
    """测试异常包装"""
    lance_store.table = MagicMock()
    lance_store.table.search.side_effect = Exception("Lance Error")
    
    with pytest.raises(StorageError, match="LanceDB search failed"):
        await lance_store.search([0.1, 0.1])