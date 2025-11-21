# tests/core/test_prompt.py
import pytest
from pathlib import Path
from gecko.core.prompt import PromptTemplate, PromptLibrary


class TestPromptTemplate:
    """PromptTemplate 测试"""
    
    # ===== 基础功能 =====
    
    def test_basic_format(self):
        """测试基础格式化"""
        template = PromptTemplate(
            template="Hello, {{ name }}!",
            input_variables=["name"]
        )
        
        result = template.format(name="Alice")
        assert result == "Hello, Alice!"
    
    def test_multiple_variables(self):
        """测试多个变量"""
        template = PromptTemplate(
            template="{{ greeting }}, {{ name }}! You are {{ age }}.",
            input_variables=["greeting", "name", "age"]
        )
        
        result = template.format(greeting="Hi", name="Bob", age=25)
        assert "Hi" in result
        assert "Bob" in result
        assert "25" in result
    
    def test_missing_variable(self):
        """测试缺少变量"""
        template = PromptTemplate(
            template="Hello, {{ name }}!",
            input_variables=["name"]
        )
        
        with pytest.raises(ValueError, match="缺少必需的模板变量"):
            template.format()
    
    # ===== Jinja2 功能 =====
    
    def test_conditional(self):
        """测试条件语句"""
        template = PromptTemplate(
            template="""
{% if premium %}
Premium User
{% else %}
Regular User
{% endif %}
            """,
            input_variables=["premium"]
        )
        
        result1 = template.format(premium=True)
        assert "Premium User" in result1
        
        result2 = template.format(premium=False)
        assert "Regular User" in result2
    
    def test_loop(self):
        """测试循环"""
        template = PromptTemplate(
            template="""
{% for item in items %}
- {{ item }}
{% endfor %}
            """,
            input_variables=["items"]
        )
        
        result = template.format(items=["a", "b", "c"])
        assert "- a" in result
        assert "- b" in result
        assert "- c" in result
    
    # ===== 变量提取 =====
    
    def test_extract_variables(self):
        """测试变量提取"""
        template = PromptTemplate(
            template="User {{ user }} asked {{ question }}"
        )
        
        variables = template.get_variables_from_template()
        assert "user" in variables
        assert "question" in variables
    
    # ===== 部分填充 =====
    
    def test_partial(self):
        """测试部分填充"""
        template = PromptTemplate(
            template="{{ a }} and {{ b }}",
            input_variables=["a", "b"]
        )
        
        partial = template.partial(a="fixed")
        assert partial.input_variables == ["b"]
    
    # ===== 安全格式化 =====
    
    def test_format_safe(self):
        """测试安全格式化"""
        template = PromptTemplate(
            template="Hello {{ name }}, you are {{ age }}",
            input_variables=["name", "age"]
        )
        
        result = template.format_safe(name="Alice")
        assert "Alice" in result
        assert "<MISSING: age>" in result
    
    # ===== 工厂方法 =====
    
    def test_from_examples(self):
        """测试从示例创建"""
        examples = [
            {"input": "1+1", "output": "2"},
            {"input": "2+2", "output": "4"},
        ]
        
        template = PromptTemplate.from_examples(examples)
        assert "1+1" in template.template
        assert "2" in template.template
    
    # ===== 克隆 =====
    
    def test_clone(self):
        """测试克隆"""
        original = PromptTemplate(
            template="Hello {{ name }}",
            input_variables=["name"]
        )
        
        cloned = original.clone()
        assert cloned.template == original.template
        assert cloned is not original
    
    # ===== 字符串表示 =====
    
    def test_str(self):
        """测试字符串表示"""
        template = PromptTemplate(template="Test")
        str_repr = str(template)
        assert "PromptTemplate" in str_repr


class TestPromptLibrary:
    """PromptLibrary 测试"""
    
    def test_get_react_prompt(self):
        """测试 ReAct 模板"""
        template = PromptLibrary.get_react_prompt()
        
        assert template is not None
        assert "question" in template.input_variables
    
    def test_get_chat_prompt(self):
        """测试对话模板"""
        template = PromptLibrary.get_chat_prompt()
        
        assert template is not None
        
        # 测试1：完整参数
        result1 = template.format(
            user_input="Hello",
            system="You are helpful",
            history=[
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"}
            ]
        )
        assert "Hello" in result1
        
        # 测试2：最小参数（依赖 format_safe 的智能默认值）
        result2 = template.format_safe(user_input="World")
        assert "World" in result2
        assert "User: World" in result2
    
    def test_get_summarization_prompt(self):
        """测试摘要模板"""
        template = PromptLibrary.get_summarization_prompt()
        
        result = template.format(
            text="Long text here...",
            max_words=50
        )
        assert "Long text" in result
        assert "50" in result


class TestEdgeCases:
    """边缘情况测试"""
    
    def test_empty_template(self):
        """测试空模板"""
        template = PromptTemplate(template="")
        result = template.format()
        assert result == ""
    
    def test_no_variables(self):
        """测试无变量模板"""
        template = PromptTemplate(template="Static text")
        result = template.format()
        assert result == "Static text"
    
    def test_unicode(self):
        """测试 Unicode 字符"""
        template = PromptTemplate(
            template="你好，{{ name }}！",
            input_variables=["name"]
        )
        
        result = template.format(name="世界")
        assert "你好" in result
        assert "世界" in result