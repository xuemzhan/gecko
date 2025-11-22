from typing import Type
import pytest
from pydantic import BaseModel
from gecko.plugins.tools.base import BaseTool, ToolResult
from gecko.plugins.tools.registry import ToolRegistry, register_tool
from gecko.plugins.tools.standard.calculator import CalculatorTool
from gecko.core.toolbox import ToolBox

from unittest.mock import MagicMock, patch
from gecko.plugins.tools.standard.duckduckgo import DuckDuckGoSearchTool

# 1. 定义测试用的工具
class EchoArgs(BaseModel):
    msg: str

@register_tool("echo_test")
class EchoTool(BaseTool):
    name: str = "echo_test"
    description: str = "Echoes back the message"
    args_schema: Type[BaseModel] = EchoArgs

    async def _run(self, args: EchoArgs) -> ToolResult: # type: ignore
        return ToolResult(content=f"ECHO: {args.msg}")

# 2. 测试注册表逻辑
def test_registry():
    assert "echo_test" in ToolRegistry.list_tools()
    
    # 测试工厂加载
    tool = ToolRegistry.load_tool("echo_test")
    assert isinstance(tool, EchoTool)
    assert tool.name == "echo_test"

# 3. 测试 BaseTool 校验
@pytest.mark.asyncio
async def test_base_tool_validation():
    tool = EchoTool() # type: ignore
    
    # 正常情况
    res = await tool.execute({"msg": "hello"})
    assert not res.is_error
    assert res.content == "ECHO: hello"
    
    # 缺少参数
    res = await tool.execute({})
    assert res.is_error
    assert "参数校验错误" in res.content

    # OpenAI Schema 生成
    schema = tool.openai_schema
    assert schema["function"]["name"] == "echo_test"
    assert "msg" in schema["function"]["parameters"]["properties"]

# 4. 测试安全计算器
@pytest.mark.asyncio
async def test_calculator_security():
    calc = CalculatorTool() # type: ignore

    # 正常计算
    res = await calc.execute({"expression": "1 + 2 * 3"})
    assert res.content == "7"

    res = await calc.execute({"expression": "sqrt(16)"})
    assert res.content == "4.0"

    # 攻击尝试 1: os 模块 (__import__ chain)
    # 此时 AST 解析为 Call，但 func 是 Attribute，触发 "不支持的复杂函数调用"
    res = await calc.execute({"expression": "__import__('os').system('ls')"})
    assert res.is_error
    # 修正点：增加了对应的错误信息断言
    assert any(msg in res.content for msg in [
        "非法表达式结构", 
        "Name", 
        "不支持的复杂函数调用", # <--- 匹配本次攻击的错误
        "禁止调用的函数"
    ])

    # 攻击尝试 2: 访问属性 (Attribute Access)
    # AST 解析为 Attribute，_safe_eval 未处理 Attribute 节点，触发 "非法表达式结构"
    res = await calc.execute({"expression": "(1).__class__.__bases__[0]"})
    assert res.is_error
    # 这里的错误通常是 "非法表达式结构" (Attribute) 或 "不支持的下标" (Subscript)
    
    # 攻击尝试 3: 超长输入
    res = await calc.execute({"expression": "1" * 1000})
    assert res.is_error
    assert "过长" in res.content

# 5. 测试 ToolBox 集成
@pytest.mark.asyncio
async def test_toolbox_integration():
    # 通过字符串名称初始化
    toolbox = ToolBox(tools=["echo_test", "calculator"]) # type: ignore
    
    assert toolbox.has_tool("echo_test")
    assert toolbox.has_tool("calculator")
    
    # 执行
    res = await toolbox.execute("echo_test", {"msg": "Integration Works"})
    assert res == "ECHO: Integration Works"

# [新增] DuckDuckGo 工具测试套件
class TestDuckDuckGoTool:
    
    @pytest.mark.asyncio
    async def test_ddg_tool_success(self):
        """测试 DDG 搜索成功场景 (Mock 网络请求)"""
        # Mock duckduckgo_search 库
        with patch("gecko.plugins.tools.standard.duckduckgo.DDGS") as mock_ddgs_cls:
            mock_instance = mock_ddgs_cls.return_value
            # 模拟上下文管理器
            mock_instance.__enter__.return_value = mock_instance
            
            # 模拟 text() 方法返回生成器/列表
            mock_instance.text.return_value = [
                {"title": "Gecko Framework", "href": "https://gecko.ai", "body": "AI Agent Framework"}
            ]
            
            tool = DuckDuckGoSearchTool() # type: ignore
            result = await tool.execute({"query": "gecko ai"})
            
            assert not result.is_error
            assert "Gecko Framework" in result.content
            assert "https://gecko.ai" in result.content
            assert result.metadata["count"] == 1

    @pytest.mark.asyncio
    async def test_ddg_tool_missing_dependency(self):
        """测试依赖缺失时的降级处理"""
        tool = DuckDuckGoSearchTool() # type: ignore
        
        # 模拟 sys.modules 中找不到 duckduckgo_search
        # 这会触发 tool._run 中的 ImportError
        with patch.dict("sys.modules", {"duckduckgo_search": None}):
            result = await tool.execute({"query": "test"})
            
            assert result.is_error
            assert "未安装 duckduckgo_search" in result.content

    @pytest.mark.asyncio
    async def test_ddg_tool_search_error(self):
        """测试搜索过程抛出异常"""
        with patch("gecko.plugins.tools.standard.duckduckgo.DDGS") as mock_ddgs_cls:
            mock_instance = mock_ddgs_cls.return_value
            mock_instance.__enter__.return_value = mock_instance
            
            # 模拟网络错误
            mock_instance.text.side_effect = Exception("Network Timeout")
            
            tool = DuckDuckGoSearchTool() # type: ignore
            result = await tool.execute({"query": "fail"})
            
            assert result.is_error
            assert "搜索请求失败" in result.content
            assert "Network Timeout" in result.content

# [修改] DuckDuckGo 工具测试套件
# class TestDuckDuckGoTool:
    
#     @pytest.mark.asyncio
#     async def test_ddg_tool_success(self):
#         """测试 DDG 搜索成功场景 (Mock 网络请求)"""
#         # 1. 构造 Mock 的 duckduckgo_search 模块和 DDGS 类
#         mock_ddgs_module = MagicMock()
#         mock_ddgs_cls = MagicMock()
#         mock_ddgs_module.DDGS = mock_ddgs_cls
        
#         # 2. 模拟 DDGS 实例行为
#         mock_instance = mock_ddgs_cls.return_value
#         # 模拟上下文管理器 (with DDGS() as ddgs:)
#         mock_instance.__enter__.return_value = mock_instance
        
#         # 模拟 text() 方法返回数据
#         mock_instance.text.return_value = [
#             {"title": "Gecko Framework", "href": "https://gecko.ai", "body": "AI Agent Framework"}
#         ]
        
#         # 3. 使用 patch.dict 注入 sys.modules
#         # 这使得工具内部的 `from duckduckgo_search import DDGS` 能成功导入我们的 Mock
#         with patch.dict("sys.modules", {"duckduckgo_search": mock_ddgs_module}):
#             tool = DuckDuckGoSearchTool() # type: ignore
#             result = await tool.execute({"query": "gecko ai"})
            
#             assert not result.is_error
#             assert "Gecko Framework" in result.content
#             assert "https://gecko.ai" in result.content
#             assert result.metadata["count"] == 1

#     @pytest.mark.asyncio
#     async def test_ddg_tool_missing_dependency(self):
#         """测试依赖缺失时的降级处理"""
#         tool = DuckDuckGoSearchTool() # type: ignore
        
#         # 模拟 sys.modules 中找不到 duckduckgo_search (设置为 None 即为 ImportError)
#         with patch.dict("sys.modules", {"duckduckgo_search": None}):
#             result = await tool.execute({"query": "test"})
            
#             assert result.is_error
#             assert "未安装 duckduckgo_search" in result.content

#     @pytest.mark.asyncio
#     async def test_ddg_tool_search_error(self):
#         """测试搜索过程抛出异常"""
#         # 1. 构造 Mock
#         mock_ddgs_module = MagicMock()
#         mock_ddgs_cls = MagicMock()
#         mock_ddgs_module.DDGS = mock_ddgs_cls
        
#         mock_instance = mock_ddgs_cls.return_value
#         mock_instance.__enter__.return_value = mock_instance
        
#         # 2. 模拟网络错误
#         mock_instance.text.side_effect = Exception("Network Timeout")
        
#         # 3. 注入 Mock
#         with patch.dict("sys.modules", {"duckduckgo_search": mock_ddgs_module}):
#             tool = DuckDuckGoSearchTool() # type: ignore
#             result = await tool.execute({"query": "fail"})
            
#             assert result.is_error
#             assert "搜索请求失败" in result.content
#             assert "Network Timeout" in result.content