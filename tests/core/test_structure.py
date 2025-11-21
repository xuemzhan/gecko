# tests/core/test_structure.py
import pytest
from pydantic import BaseModel, Field
from gecko.core.structure import (
    StructureEngine,
    StructureParseError,
    extract_json_from_text
)


# 测试用模型
class User(BaseModel):
    name: str
    age: int


class ComplexModel(BaseModel):
    title: str = Field(description="标题")
    items: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class TestStructureEngine:
    """StructureEngine 测试"""
    
    # ===== Schema 生成 =====
    
    def test_to_openai_tool(self):
        """测试生成 OpenAI Tool Schema"""
        schema = StructureEngine.to_openai_tool(User)
        
        assert schema["type"] == "function"
        assert "function" in schema
        assert "name" in schema["function"]
        assert "parameters" in schema["function"]
        
        params = schema["function"]["parameters"]
        assert "name" in params["properties"]
        assert "age" in params["properties"]
    
    # ===== 解析测试 =====
    
    @pytest.mark.asyncio
    async def test_parse_direct_json(self):
        """测试直接 JSON 解析"""
        json_text = '{"name": "Alice", "age": 25}'
        
        user = await StructureEngine.parse(json_text, User)
        
        assert user.name == "Alice"
        assert user.age == 25
    
    @pytest.mark.asyncio
    async def test_parse_markdown_json(self):
        """测试 Markdown 代码块"""
        markdown = '''
        Some text here
        
        ```json
        {"name": "Bob", "age": 30}
        ```
        
        More text
        '''
        
        user = await StructureEngine.parse(markdown, User)
        
        assert user.name == "Bob"
        assert user.age == 30
    
    @pytest.mark.asyncio
    async def test_parse_from_tool_call(self):
        """测试从工具调用解析"""
        tool_calls = [
            {
                "function": {
                    "arguments": '{"name": "Charlie", "age": 35}'
                }
            }
        ]
        
        user = await StructureEngine.parse(
            "",
            User,
            raw_tool_calls=tool_calls
        )
        
        assert user.name == "Charlie"
        assert user.age == 35
    
    @pytest.mark.asyncio
    async def test_parse_braced_json(self):
        """测试括号匹配提取"""
        text = 'Some random text {"name": "David", "age": 40} more text'
        
        user = await StructureEngine.parse(text, User)
        
        assert user.name == "David"
    
    @pytest.mark.asyncio
    async def test_parse_invalid(self):
        """测试无效输入"""
        with pytest.raises(StructureParseError):
            await StructureEngine.parse("Not valid JSON", User)
    
    @pytest.mark.asyncio
    async def test_parse_with_auto_fix(self):
        """测试自动修复"""
        # 带注释和尾部逗号
        dirty_json = '''
        {
            "name": "Eve",
            "age": 28,
        }
        '''
        
        user = await StructureEngine.parse(
            dirty_json,
            User,
            auto_fix=True
        )
        
        assert user.name == "Eve"
    
    # ===== 验证测试 =====
    
    def test_validate(self):
        """测试数据验证"""
        data = {"name": "Frank", "age": 45}
        
        user = StructureEngine.validate(data, User)
        
        assert user.name == "Frank"
        assert user.age == 45
    
    def test_validate_invalid(self):
        """测试验证失败"""
        data = {"name": "George"}  # 缺少 age
        
        with pytest.raises(Exception):  # Pydantic ValidationError
            StructureEngine.validate(data, User)
    
    # ===== Schema 差异 =====
    
    def test_get_schema_diff(self):
        """测试 Schema 差异检查"""
        data = {
            "name": "Helen",
            "extra": "field"
        }
        
        diff = StructureEngine.get_schema_diff(data, User)
        
        assert "age" in diff["missing_required"]
        assert "extra" in diff["extra_fields"]


class TestHelperFunctions:
    """辅助函数测试"""
    
    def test_extract_json_from_text(self):
        """测试 JSON 提取"""
        text = 'Here is data: {"key": "value"} end'
        
        result = extract_json_from_text(text)
        
        assert result is not None
        assert result["key"] == "value"
    
    def test_extract_json_none(self):
        """测试未找到 JSON"""
        text = "No JSON here"
        
        result = extract_json_from_text(text)
        
        assert result is None


class TestStructureParseError:
    """异常测试"""
    
    @pytest.mark.asyncio
    async def test_detailed_error(self):
        """测试详细错误信息"""
        try:
            await StructureEngine.parse("invalid", User)
        except StructureParseError as e:
            detailed = e.get_detailed_error()
            
            assert "结构化解析失败" in detailed
            assert "尝试的解析策略" in detailed
            assert len(e.attempts) > 0