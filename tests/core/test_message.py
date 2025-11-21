# tests/core/test_message.py
import pytest
import asyncio
from pathlib import Path
from gecko.core.message import Message, MediaResource, ContentBlock


class TestMediaResource:
    """MediaResource 测试"""
    
    def test_url_resource(self):
        """测试 URL 资源"""
        resource = MediaResource(url="https://example.com/image.jpg")
        assert resource.url is not None
        
        openai_format = resource.to_openai_image_url()
        assert openai_format["url"] == "https://example.com/image.jpg"
        assert "detail" in openai_format
    
    def test_base64_resource(self):
        """测试 base64 资源"""
        resource = MediaResource(
            base64_data="abc123",
            mime_type="image/png"
        )
        
        assert resource.base64_data == "abc123"
        
        openai_format = resource.to_openai_image_url()
        assert "data:image/png;base64,abc123" in openai_format["url"]
    
    def test_missing_source(self):
        """测试缺少数据源"""
        with pytest.raises(ValueError, match="必须提供"):
            MediaResource()
    
    def test_size_estimate(self):
        """测试大小估算"""
        # Base64 编码的 "Hello" (SGVsbG8=)
        resource = MediaResource(base64_data="SGVsbG8=")
        size = resource.get_size_estimate()
        assert size > 0


class TestContentBlock:
    """ContentBlock 测试"""
    
    def test_text_block(self):
        """测试文本块"""
        block = ContentBlock(type="text", text="Hello")
        assert block.type == "text"
        assert block.text == "Hello"
        
        openai = block.to_openai_format()
        assert openai["type"] == "text"
        assert openai["text"] == "Hello"
    
    def test_image_block(self):
        """测试图片块"""
        resource = MediaResource(url="https://example.com/img.jpg")
        block = ContentBlock(type="image_url", image_url=resource)
        
        assert block.type == "image_url"
        
        openai = block.to_openai_format()
        assert openai["type"] == "image_url"
    
    def test_invalid_text_block(self):
        """测试无效的文本块"""
        with pytest.raises(ValueError, match="缺少 text"):
            ContentBlock(type="text")
    
    def test_get_text_content(self):
        """测试提取文本"""
        # ✅ 修复：测试文本块
        text_block = ContentBlock(type="text", text="Hello")
        assert text_block.get_text_content() == "Hello"
        
        # ✅ 修复：测试图片块（使用 image_block 而不是 text_block）
        image_block = ContentBlock(
            type="image_url",
            image_url=MediaResource(url="https://example.com/img.jpg")
        )
        assert image_block.get_text_content() == "[image]"  # ✅ 正确


class TestMessage:
    """Message 测试"""
    
    # ===== 工厂方法测试 =====
    
    def test_user_message(self):
        """测试用户消息"""
        msg = Message.user("Hello")
        
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.get_text_content() == "Hello"
    
    def test_assistant_message(self):
        """测试助手消息"""
        msg = Message.assistant("Hi there!")
        
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"
    
    def test_system_message(self):
        """测试系统消息"""
        msg = Message.system("You are helpful")
        
        assert msg.role == "system"
        assert msg.content == "You are helpful"
    
    def test_tool_result_message(self):
        """测试工具返回消息"""
        msg = Message.tool_result(
            tool_call_id="call_123",
            content="Result",
            tool_name="search"
        )
        
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_123"
        assert msg.name == "search"
    
    def test_tool_result_with_dict(self):
        """测试工具返回消息（字典内容）"""
        msg = Message.tool_result(
            tool_call_id="call_123",
            content={"result": "success", "count": 42},
            tool_name="search"
        )
        
        assert msg.role == "tool"
        assert "result" in msg.content
        assert "count" in msg.content
    
    # ===== 转换测试 =====
    
    def test_to_openai_format(self):
        """测试转换为 OpenAI 格式"""
        msg = Message.user("Hello")
        openai = msg.to_openai_format()
        
        assert openai["role"] == "user"
        assert openai["content"] == "Hello"
    
    def test_from_openai(self):
        """测试从 OpenAI 格式解析"""
        openai_msg = {
            "role": "assistant",
            "content": "Hi"
        }
        
        msg = Message.from_openai(openai_msg)
        
        assert msg.role == "assistant"
        assert msg.content == "Hi"
    
    def test_from_openai_with_tool_calls(self):
        """测试解析带 tool_calls 的消息"""
        openai_msg = {
            "role": "assistant",
            "content": "Searching...",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "test"}'
                    }
                }
            ]
        }
        
        msg = Message.from_openai(openai_msg)
        
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
    
    # ===== 工具方法测试 =====
    
    def test_is_empty(self):
        """测试空消息检查"""
        empty_msg = Message.user("")
        assert empty_msg.is_empty()
        
        non_empty = Message.user("Hello")
        assert not non_empty.is_empty()
    
    def test_has_images(self):
        """测试图片检查"""
        text_msg = Message.user("Hello")
        assert not text_msg.has_images()
        
        # 多模态消息需要实际图片文件，这里跳过
    
    def test_clone(self):
        """测试消息克隆"""
        original = Message.user("Hello")
        cloned = original.clone()
        
        assert cloned.role == original.role
        assert cloned.content == original.content
        assert cloned is not original
    
    def test_truncate_content(self):
        """测试内容截断"""
        long_msg = Message.user("A" * 100)
        truncated = long_msg.truncate_content(50)
        
        assert len(truncated.get_text_content()) <= 53  # 50 + "..."
        assert "..." in truncated.get_text_content()
    
    def test_str_repr(self):
        """测试字符串表示"""
        msg = Message.user("Hello")
        
        str_repr = str(msg)
        assert "Message" in str_repr
        assert "user" in str_repr
        
        repr_str = repr(msg)
        assert "Message" in repr_str


class TestMessageAsync:
    """异步功能测试"""
    
    @pytest.mark.asyncio
    async def test_user_async_no_images(self):
        """测试异步用户消息（无图片）"""
        msg = await Message.user_async("Hello")
        
        assert msg.role == "user"
        assert msg.content == "Hello"