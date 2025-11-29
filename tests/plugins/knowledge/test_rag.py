# tests/plugins/knowledge/test_rag.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from gecko.plugins.knowledge.splitters import RecursiveCharacterTextSplitter
from gecko.plugins.knowledge.document import Document
from gecko.plugins.knowledge.pipeline import IngestionPipeline

class TestTextSplitter:
    def test_recursive_splitting_basic(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=0)
        text = "1234567890abcdefghij" # 20 chars
        
        # 应该切分为2块
        chunks = splitter.split_text(text)
        assert len(chunks) == 2
        assert chunks[0] == "1234567890"
        assert chunks[1] == "abcdefghij"

    def test_splitting_with_separator(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=10, separators=[" "])
        text = "hello world python"
        
        chunks = splitter.split_text(text)
        # "hello world" (11) > 10 -> "hello", "world"
        # "python"
        assert "hello" in chunks
        assert "world" in chunks 
        assert "python" in chunks

    def test_document_metadata_inheritance(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=100)
        doc = Document(text="content", metadata={"source": "test.txt"})
        
        split_docs = splitter.split_documents([doc])
        
        assert len(split_docs) == 1
        assert split_docs[0].metadata["source"] == "test.txt"
        # 验证注入了 chunk_index
        assert "chunk_index" in split_docs[0].metadata

class TestIngestionPipeline:
    @pytest.mark.asyncio
    async def test_pipeline_execution(self):
        """
        测试 IngestionPipeline 的完整执行流程

        修复点：
        - AutoReader.read 在实现里是同步调用，因此这里用普通 patch/Mock，
          避免返回 AsyncMock 导致 `'coroutine' object is not iterable` 和未 await 警告。
        """
        from gecko.plugins.knowledge.readers import Document

        # 1. 构造向量存储与向量化器的 Mock
        mock_vec_store = MagicMock()
        mock_vec_store.upsert = AsyncMock()  # upsert 在流水线里是 await 的，所以用 AsyncMock 没问题

        mock_embedder = MagicMock()
        # 模拟返回向量: 2 个文档 -> 2 个向量
        mock_embedder.embed_documents = AsyncMock(return_value=[[0.1], [0.2]])

        pipeline = IngestionPipeline(mock_vec_store, mock_embedder)

        # 2. Mock AutoReader.read 返回两个“文档”
        # 注意：此处不再使用 AsyncMock，而是普通同步 Mock
        with patch("gecko.plugins.knowledge.readers.AutoReader.read") as mock_read:
            mock_read.return_value = [
                Document(text="doc1"),
                Document(text="doc2"),
            ]

            # 3. 执行流水线
            await pipeline.run("dummy_path")  # type: ignore

        # 4. 断言向量存储的 upsert 被调用
        mock_vec_store.upsert.assert_awaited_once()
        mock_embedder.embed_documents.assert_awaited_once()
