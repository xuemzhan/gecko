import pytest
from unittest.mock import MagicMock, AsyncMock
from gecko.core.agent import Agent
from gecko.core.toolbox import ToolBox
from gecko.core.memory import TokenMemory
from gecko.core.engine.react import ReActEngine

# 辅助：模拟 Async Generator
async def mock_stream_gen(chunks):
    for c in chunks:
        mock_chunk = MagicMock()
        mock_chunk.choices[0].delta.content = c
        mock_chunk.choices[0].delta.tool_calls = None
        yield mock_chunk

@pytest.mark.asyncio
async def test_agent_streaming_flow():
    # --- Mock Model ---
    mock_model = MagicMock()
    # 模拟返回 "Hello", " ", "World"
    mock_model.astream.return_value = mock_stream_gen(["Hello", " ", "World"])
    
    # --- Setup ---
    agent = Agent(
        model=mock_model,
        toolbox=ToolBox(),
        memory=TokenMemory("test_stream"),
        engine_cls=ReActEngine
    )

    # --- Execute ---
    accumulated = ""
    async for token in agent.stream("Hi"):
        accumulated += token

    # --- Assert ---
    assert accumulated == "Hello World"
    # 验证调用了 astream
    mock_model.astream.assert_called_once()