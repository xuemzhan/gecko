# tests/unit/test_tools.py
from typing import Any, Dict
from gecko.plugins.tools.registry import ToolRegistry, tool
from gecko.plugins.tools.base import BaseTool

@tool
class MockTool(BaseTool):
    name: str = "mock_tool"  # 添加 : str 类型注解（关键修复）
    description: str = "test"  # 添加 : str 类型注解
    parameters: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}  # 添加 : Dict[str, Any] 类型注解

    async def execute(self, args):
        return "success"

def test_tool_registry():
    tools = ToolRegistry.list_all()
    assert any(t.name == "mock_tool" for t in tools)