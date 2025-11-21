from typing import Type
import pytest  
from unittest.mock import patch, MagicMock  
  
from gecko.plugins.tools.base import BaseTool, ToolArgsModel, ToolResult  
from gecko.plugins.tools.registry import ToolRegistry, tool  
from gecko.plugins.tools.executor import ToolExecutor  
from gecko.plugins.tools.calculator import CalculatorTool  
from gecko.plugins.tools.duckduckgo import DuckDuckGoSearch  
from pydantic import Field  
  
  
# ---------- 基础测试用工具 ----------  
class EchoArgs(ToolArgsModel):  
    text: str = Field(...)  
  
  
@tool  
class EchoTool(BaseTool):  
    name: str = "echo_tool"  
    description: str = "返回输入文本"  
    args_model: Type[EchoArgs] = EchoArgs  
  
    def _execute_impl(self, args: EchoArgs) -> ToolResult:  
        return ToolResult(content=args.text) 
  
  
@pytest.mark.asyncio  
async def test_base_tool_argument_validation():  
    tool = EchoTool()  
    ok = await tool.execute({"text": "hello"})  
    assert ok.content == "hello"  
    bad = await tool.execute({"text": 123})  
    assert bad.is_error  
  
  
def test_registry_basic_operations():  
    ToolRegistry.register(EchoTool, replace=True)  
    assert "echo_tool" in ToolRegistry.list_all()  
    tool = ToolRegistry.get("echo_tool")  
    assert isinstance(tool, EchoTool)  
  
  
@pytest.mark.asyncio  
async def test_executor_keeps_order():  
    ToolRegistry.register(EchoTool, replace=True)  
    tool_calls = [  
        {"name": "echo_tool", "arguments": {"text": "A"}},  
        {"name": "echo_tool", "arguments": {"text": "B"}},  
    ]  
    results = await ToolExecutor.concurrent_execute(tool_calls, max_concurrent=2)  
    assert [r.content for r in results] == ["A", "B"]  
  
  
@pytest.mark.asyncio  
async def test_executor_missing_tool():  
    ToolRegistry._tools.pop("missing_tool", None)  
    res = await ToolExecutor.concurrent_execute(  
        [{"name": "missing_tool", "arguments": {}}],  
    )  
    assert res[0].is_error  
  
  
@pytest.mark.asyncio  
async def test_calculator_success_and_failure():  
    calc = CalculatorTool()  
    ok = await calc.execute({"expression": "1 + 2"})  
    assert "3" in ok.content  
  
    bad = await calc.execute({"expression": "1 + __import__('os').system('rm -rf /')"})  
    assert bad.is_error  
  
  
@pytest.mark.asyncio  
async def test_duckduckgo_tool(monkeypatch):  
    fake_results = [  
        {"title": "Result 1", "href": "https://example.com/1"},  
        {"title": "Result 2", "href": "https://example.com/2"},  
    ]  
  
    fake_ddgs = MagicMock()  
    fake_ddgs.__enter__.return_value = fake_ddgs  
    fake_ddgs.__exit__.return_value = False  
    fake_ddgs.text.return_value = fake_results  
  
    with patch("gecko.plugins.tools.duckduckgo.DDGS", return_value=fake_ddgs):  
        tool = DuckDuckGoSearch()  
        result = await tool.execute({"query": "gecko"})  
        assert "Result 1" in result.content  
        assert not result.is_error  
  
    # empty query should be error  
    error = await tool.execute({"query": ""})  
    assert error.is_error  
