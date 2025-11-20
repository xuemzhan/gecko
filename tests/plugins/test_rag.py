# tests/knowledge/test_rag.py
import pytest
import os
from unittest.mock import MagicMock, AsyncMock
from gecko.plugins.knowledge.document import Document
from gecko.plugins.knowledge.splitters import RecursiveCharacterTextSplitter
from gecko.plugins.knowledge.pipeline import IngestionPipeline
from gecko.plugins.knowledge.tool import RetrievalTool

# Mock Embedder
class MockEmbedder:
    @property
    def dimension(self): return 4
    async def embed_documents(self, texts): return [[0.1, 0.2, 0.3, 0.4] for _ in texts]
    async def embed_query(self, text): return [0.1, 0.2, 0.3, 0.4]

# Mock Vector Store
class MockVectorStore:
    def __init__(self):
        self.docs = {}
    async def upsert(self, documents):
        for d in documents:
            self.docs[d['id']] = d
    async def search(self, query, top_k):
        # 简单返回所有
        return [
            {"text": d["text"], "metadata": d["metadata"], "score": 0.9} 
            for d in list(self.docs.values())[:top_k]
        ]

@pytest.mark.asyncio
async def test_splitter():
    splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    text = "helloworld1234567890"
    chunks = splitter.split_text(text)
    assert len(chunks) >= 2
    assert "hello" in chunks[0]

@pytest.mark.asyncio
async def test_pipeline_end_to_end(tmp_path):
    # 1. 准备测试文件
    test_file = tmp_path / "test.txt"
    test_file.write_text("Gecko is a lightweight agent framework.", encoding="utf-8")
    
    # 2. 初始化组件
    embedder = MockEmbedder()
    vector_store = MockVectorStore()
    pipeline = IngestionPipeline(vector_store, embedder)
    
    # 3. 运行入库
    await pipeline.run([str(test_file)])
    
    # 4. 验证存储
    assert len(vector_store.docs) > 0
    first_doc = list(vector_store.docs.values())[0]
    assert "Gecko" in first_doc["text"]
    assert first_doc["embedding"] == [0.1, 0.2, 0.3, 0.4]

@pytest.mark.asyncio
async def test_retrieval_tool():
    store = MockVectorStore()
    embedder = MockEmbedder()
    # 预存数据
    await store.upsert([{
        "id": "1", "text": "Secret info", "embedding": [0.1]*4, "metadata": {}
    }])
    
    tool = RetrievalTool(vector_store=store, embedder=embedder)
    
    # 测试调用
    result = await tool.execute({"query": "find secret"})
    assert "Secret info" in result
    assert "相关度: 0.90" in result