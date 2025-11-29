# tests/plugins/knowledge/test_tool.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from gecko.plugins.knowledge.tool import RetrievalTool
from gecko.plugins.tools.base import ToolResult

@pytest.mark.asyncio
async def test_retrieval_tool_execution():
    """测试 RAG 检索工具的端到端流程"""
    # 1. Mock 依赖
    mock_vector_store = MagicMock()
    mock_embedder = MagicMock()
    
    # 模拟 Embedding 返回
    mock_embedder.embed_query = AsyncMock(return_value=[0.1, 0.2])
    
    # 模拟 Vector Search 返回
    mock_vector_store.search = AsyncMock(return_value=[
        {
            "text": "Gecko is an agent framework.",
            "metadata": {"filename": "doc.md"},
            "score": 0.95
        }
    ])
    
    # 2. 初始化工具
    tool = RetrievalTool(
        vector_store=mock_vector_store,
        embedder=mock_embedder,
        top_k=5
    )
    
    # 3. 验证 PrivateAttr 设置正确
    assert tool._top_k == 5
    
    # 4. 执行工具
    result = await tool.execute({"query": "what is gecko"})
    
    # 5. 验证结果
    assert isinstance(result, ToolResult)
    assert not result.is_error
    assert "Gecko is an agent framework" in result.content
    assert "doc.md" in result.content
    
    # 6. 验证调用链
    mock_embedder.embed_query.assert_awaited_with("what is gecko")
    mock_vector_store.search.assert_awaited_with([0.1, 0.2], top_k=5)

@pytest.mark.asyncio
async def test_retrieval_tool_empty_results():
    """测试无结果的情况"""
    mock_store = MagicMock()
    mock_store.search = AsyncMock(return_value=[]) # 空列表
    
    tool = RetrievalTool(
        vector_store=mock_store,
        embedder=MagicMock()
    )
    # Mock embedder return to avoid error
    tool._embedder.embed_query = AsyncMock(return_value=[0.1])
    
    result = await tool.execute({"query": "unknown"})
    
    assert "未在知识库中找到" in result.content