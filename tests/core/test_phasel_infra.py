# tests/unit/test_phase1_infra.py
import pytest
from gecko.core.prompt import PromptTemplate
from gecko.core.memory import TokenMemory
from gecko.core.toolbox import ToolBox
from gecko.core.message import Message
from gecko.plugins.tools.base import BaseTool
from gecko.plugins.tools.registry import tool as register_tool # 使用旧装饰器定义 Mock 工具

# --- Mock Data ---
class MockTool(BaseTool):
    name: str = "mock_tool"
    description: str = "A mock tool"
    
    async def execute(self, arguments: dict) -> str:
        return f"Executed with {arguments}"

# --- Tests ---

def test_prompt_template_rendering():
    template_str = "Hello, {{ name }}! Today is {{ day }}."
    prompt = PromptTemplate(template=template_str, input_variables=["name", "day"])
    
    result = prompt.format(name="Gecko", day="Monday")
    assert result == "Hello, Gecko! Today is Monday."

def test_prompt_template_error():
    # 可以在这里测试缺失变量的行为，或者 jinja2 语法错误
    pass

@pytest.mark.asyncio
async def test_token_memory_truncation():
    # 假设 "hello" ~= 1 token
    memory = TokenMemory(session_id="test", max_tokens=10) # 极小的限制
    
    # 构造消息
    sys_msg = {"role": "system", "content": "System"} # ~2 tokens
    user_msg_1 = {"role": "user", "content": "A" * 50} # Long message
    user_msg_2 = {"role": "user", "content": "Short"} # Short message
    
    raw_history = [sys_msg, user_msg_1, user_msg_2]
    
    # 执行修剪
    clean_history = await memory.get_history(raw_history)
    
    # 断言
    # 1. System 必须在
    assert clean_history[0].role == "system"
    # 2. user_msg_1 应该被丢弃，因为它很长且在前面
    # 3. user_msg_2 应该被保留，因为它最近且短
    assert len(clean_history) >= 2
    assert clean_history[-1].content == "Short"
    
    # 验证总 token 数未超标 (粗略验证)
    total_content = "".join([m.content for m in clean_history])
    assert memory.count_tokens(total_content) < 50 

@pytest.mark.asyncio
async def test_toolbox_management():
    tb = ToolBox()
    t = MockTool()
    
    # Register
    tb.register(t)
    assert tb.get("mock_tool") is not None
    
    # Schema
    schema = tb.to_openai_schema()
    assert len(schema) == 1
    assert schema[0]["function"]["name"] == "mock_tool"
    
    # Execute
    res = await tb.execute("mock_tool", {"arg": 1})
    assert res == "Executed with {'arg': 1}"
    
    # Missing
    with pytest.raises(ValueError):
        await tb.execute("non_existent", {})