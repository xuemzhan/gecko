# tests/plugins/knowledge/test_rag.py
import pytest
from unittest.mock import MagicMock, AsyncMock
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
        # Mocks
        mock_vec_store = MagicMock()
        mock_vec_store.upsert = AsyncMock()
        
        mock_embedder = MagicMock()
        # 模拟返回向量: 2个文档 -> 2个向量
        mock_embedder.embed_documents = AsyncMock(return_value=[[0.1], [0.2]])
        
        pipeline = IngestionPipeline(mock_vec_store, mock_embedder)
        
        # 模拟 Reader 返回的文档
        # Mock AutoReader
        with pytest.mock.patch("gecko.plugins.knowledge.readers.AutoReader.read") as mock_read: # type: ignore
            mock_read.return_value = [
                Document(text="doc1"), 
                Document(text="doc2")
            ]
            
            await pipeline.run(["test.txt"])
            
            # 验证流程
            # 1. Embedder 被调用
            assert mock_embedder.embed_documents.called
            
            # 2. Vector Store 被调用
            assert mock_vec_store.upsert.called
            call_args = mock_vec_store.upsert.call_args[0][0] # docs list
            
            assert len(call_args) == 2
            assert call_args[0]["embedding"] == [0.1]