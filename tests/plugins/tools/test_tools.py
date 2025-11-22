from typing import Type
import pytest
from pydantic import BaseModel
from gecko.plugins.tools.base import BaseTool, ToolResult
from gecko.plugins.tools.registry import ToolRegistry, register_tool
from gecko.plugins.tools.standard.calculator import CalculatorTool
from gecko.core.toolbox import ToolBox

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