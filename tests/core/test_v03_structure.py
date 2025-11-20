# tests/core/test_v03_structure.py
import pytest
from typing import List
from pydantic import BaseModel
from unittest.mock import AsyncMock, MagicMock
from gecko.core.engine.react import ReActEngine
from gecko.core.toolbox import ToolBox
from gecko.core.memory import TokenMemory
from gecko.core.message import Message

class UserData(BaseModel):
    name: str
    age: int
    tags: list[str]

@pytest.mark.asyncio
async def test_react_engine_retry_logic():
    # Mock 模型
    mock_model = AsyncMock()

    # --- 第一次调用结果 (Bad Response) ---
    resp1 = MagicMock()
    resp1.choices[0].message.content = "Not JSON"
    resp1.choices[0].message.tool_calls = None
    # [关键修复] 正确 Mock model_dump 返回值，包含 role
    resp1.choices[0].message.model_dump.return_value = {
        "role": "assistant",
        "content": "Not JSON",
        "tool_calls": None
    }

    # --- 第二次调用结果 (Good Response) ---
    resp2 = MagicMock()
    resp2.choices[0].message.content = '{"name": "Fixed", "age": 2, "tags": []}'
    resp2.choices[0].message.tool_calls = None
    # [关键修复] 正确 Mock model_dump 返回值
    resp2.choices[0].message.model_dump.return_value = {
        "role": "assistant",
        "content": '{"name": "Fixed", "age": 2, "tags": []}',
        "tool_calls": None
    }

    mock_model.acompletion.side_effect = [resp1, resp2]

    # Setup Engine
    engine = ReActEngine(mock_model, ToolBox(), TokenMemory("test"))

    # Run (允许 1 次重试)
    result = await engine.step([], response_model=UserData, max_retries=1)

    # Assert
    assert isinstance(result, UserData)
    assert result.name == "Fixed"
    
    # 验证调用次数 (1次失败 + 1次重试)
    assert mock_model.acompletion.call_count == 2