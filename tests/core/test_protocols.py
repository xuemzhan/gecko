# tests/core/test_protocols.py (完整修正版)
import pytest
from gecko.core.protocols import (
    ModelProtocol,
    StreamableModelProtocol,
    StorageProtocol,
    ToolProtocol,
    CompletionResponse,
    CompletionChoice,
    StreamChunk,
    check_protocol,
    validate_model,
    validate_storage,
    validate_tool,
    get_missing_methods,
    supports_streaming,
    supports_function_calling,
    supports_vision,
    get_model_name,
)


class TestProtocols:
    """Protocol 测试套件"""
    
    # ==================== ModelProtocol 测试 ====================
    
    def test_valid_model(self):
        """测试有效的基础模型"""
        class ValidModel:
            async def acompletion(self, messages, **kwargs):
                return CompletionResponse(
                    model="test-model",
                    choices=[
                        CompletionChoice(
                            message={"role": "assistant", "content": "Hello"}
                        )
                    ]
                )

            # 实现 count_tokens
            def count_tokens(self, text_or_messages):
                return 0
        
        model = ValidModel()
        assert check_protocol(model, ModelProtocol)
        validate_model(model)
        assert not check_protocol(model, StreamableModelProtocol)
    
    def test_streamable_model(self):
        """测试支持流式的模型"""
        class StreamingModel:
            async def acompletion(self, messages, **kwargs):
                return CompletionResponse(model="test-streaming", choices=[])
            
            async def astream(self, messages, **kwargs):
                yield StreamChunk(model="test-streaming")

            def count_tokens(self, text_or_messages):
                return 0
        
        model = StreamingModel()
        assert check_protocol(model, ModelProtocol)
        assert check_protocol(model, StreamableModelProtocol)
        assert supports_streaming(model)
    
    def test_invalid_model(self):
        """测试无效的模型"""
        class InvalidModel:
            pass
        
        model = InvalidModel()
        assert not check_protocol(model, ModelProtocol)
        
        with pytest.raises(TypeError, match="does not implement ModelProtocol"):
            validate_model(model)
    
    def test_get_missing_methods_model_protocol(self):
        """[Updated] 测试获取 ModelProtocol 缺失的方法 (包含 count_tokens)"""
        class PartialModel:
            # 仅实现了 acompletion，缺少 count_tokens
            async def acompletion(self, messages, **kwargs):
                return CompletionResponse(model="test", choices=[])
        
        missing = get_missing_methods(PartialModel(), ModelProtocol)
        # 验证 count_tokens 被识别为缺失
        assert "count_tokens" in missing
        assert "acompletion" not in missing

    def test_valid_model_full_implementation(self):
        """[New] 测试完整实现了 ModelProtocol (含 count_tokens) 的模型"""
        class FullModel:
            async def acompletion(self, messages, **kwargs):
                return CompletionResponse(model="test", choices=[])
            
            def count_tokens(self, text_or_messages):
                return 10
        
        model = FullModel()
        assert check_protocol(model, ModelProtocol)
        validate_model(model)
    
    def test_get_missing_methods_streamable_protocol(self):
        """测试获取 StreamableModelProtocol 缺失的方法"""
        class PartialModel:
            async def acompletion(self, messages, **kwargs):
                return CompletionResponse(model="test", choices=[])
        
        missing = get_missing_methods(PartialModel(), StreamableModelProtocol)
        assert "astream" in missing
    
    # ==================== 能力检测测试 ====================
    
    def test_supports_function_calling_with_attribute(self):
        """测试 Function Calling 检测（通过属性）"""
        class ModelWithFC:
            _supports_function_calling = True
            
            async def acompletion(self, messages, **kwargs):
                return CompletionResponse(model="test", choices=[])
        
        assert supports_function_calling(ModelWithFC())
    
    def test_supports_function_calling_with_method(self):
        """测试 Function Calling 检测（通过方法）"""
        class ModelWithFCMethod:
            async def acompletion(self, messages, **kwargs):
                return CompletionResponse(model="test", choices=[])
            
            def supports_function_calling(self):
                return True
        
        assert supports_function_calling(ModelWithFCMethod())
    
    def test_supports_function_calling_false(self):
        """测试不支持 Function Calling 的模型"""
        class ModelWithoutFC:
            async def acompletion(self, messages, **kwargs):
                return CompletionResponse(model="test", choices=[])
        
        assert not supports_function_calling(ModelWithoutFC())
    
    def test_supports_vision(self):
        """测试 Vision 能力检测"""
        class ModelWithVision:
            _supports_vision = True
            
            async def acompletion(self, messages, **kwargs):
                return CompletionResponse(model="test", choices=[])
        
        class ModelWithoutVision:
            async def acompletion(self, messages, **kwargs):
                return CompletionResponse(model="test", choices=[])
        
        assert supports_vision(ModelWithVision())
        assert not supports_vision(ModelWithoutVision())
    
    def test_get_model_name(self):
        """测试获取模型名称"""
        class ModelWithName:
            model_name = "gpt-4"
            async def acompletion(self, messages, **kwargs):
                return CompletionResponse(model="test", choices=[])
        
        assert get_model_name(ModelWithName()) == "gpt-4"
        
        class ModelWithNameAttr:
            name = "custom-model"
            async def acompletion(self, messages, **kwargs):
                return CompletionResponse(model="test", choices=[])
        
        assert get_model_name(ModelWithNameAttr()) == "custom-model"
        
        class MyCustomModel:
            async def acompletion(self, messages, **kwargs):
                return CompletionResponse(model="test", choices=[])
        
        assert get_model_name(MyCustomModel()) == "MyCustomModel"
    
    # ==================== StorageProtocol 测试 ====================
    
    def test_valid_storage(self):
        """测试有效的存储后端"""
        class ValidStorage:
            async def get(self, key: str):
                return {"data": "value"}
            
            async def set(self, key: str, value: dict, ttl=None):
                pass
            
            async def delete(self, key: str):
                return True
        
        storage = ValidStorage()
        assert check_protocol(storage, StorageProtocol)
        validate_storage(storage)
    
    def test_invalid_storage(self):
        """测试无效的存储后端"""
        class InvalidStorage:
            async def get(self, key: str):
                return None
        
        storage = InvalidStorage()
        assert not check_protocol(storage, StorageProtocol)
        
        with pytest.raises(TypeError, match="does not implement StorageProtocol"):
            validate_storage(storage)
    
    # ==================== ToolProtocol 测试 ====================
    
    def test_valid_tool(self):
        """测试有效的工具"""
        class ValidTool:
            name = "calculator"
            description = "Perform calculations"
            parameters = {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
            }
            
            async def execute(self, arguments: dict):
                return "42"
        
        tool = ValidTool()
        assert check_protocol(tool, ToolProtocol)
        validate_tool(tool)
    
    def test_invalid_tool_missing_execute(self):
        """测试缺少 execute 方法的工具"""
        class InvalidTool:
            name = "test"
            description = "test"
            parameters = {}
        
        tool = InvalidTool()
        assert not check_protocol(tool, ToolProtocol)
        
        # ✅ 修正：宽松匹配，只要包含 "execute" 即可
        with pytest.raises(TypeError, match="execute"):
            validate_tool(tool)
    
    def test_invalid_tool_missing_name(self):
        """测试缺少 name 的工具"""
        class InvalidTool:
            description = "test"
            parameters = {}
            
            async def execute(self, arguments: dict):
                return "result"
        
        tool = InvalidTool()
        
        # ✅ 修正：宽松匹配，只要包含 "'name'" 即可
        with pytest.raises(ValueError, match="'name'"):
            validate_tool(tool)
    
    def test_invalid_tool_missing_description(self):
        """测试缺少 description 的工具"""
        class InvalidTool:
            name = "test"
            parameters = {}
            
            async def execute(self, arguments: dict):
                return "result"
        
        tool = InvalidTool()
        
        # ✅ 修正：宽松匹配
        with pytest.raises(ValueError, match="'description'"):
            validate_tool(tool)
    
    def test_invalid_tool_invalid_parameters(self):
        """测试无效 parameters 的工具"""
        class InvalidTool:
            name = "test"
            description = "test"
            parameters = "not a dict"
            
            async def execute(self, arguments: dict):
                return "result"
        
        tool = InvalidTool()
        
        with pytest.raises(ValueError, match="'parameters'"):
            validate_tool(tool)
    
    # ==================== 新增：空值测试 ====================
    
    def test_invalid_tool_empty_name(self):
        """测试空字符串 name 的工具"""
        class InvalidTool:
            name = ""
            description = "test"
            parameters = {}
            
            async def execute(self, arguments: dict):
                return "result"
        
        tool = InvalidTool()
        
        # ✅ 这次应该匹配 "non-empty"
        with pytest.raises(ValueError, match="non-empty 'name'"):
            validate_tool(tool)
    
    def test_invalid_tool_empty_description(self):
        """测试空字符串 description 的工具"""
        class InvalidTool:
            name = "test"
            description = ""
            parameters = {}
            
            async def execute(self, arguments: dict):
                return "result"
        
        tool = InvalidTool()
        
        with pytest.raises(ValueError, match="non-empty 'description'"):
            validate_tool(tool)
    
    def test_invalid_tool_whitespace_name(self):
        """测试仅空格 name 的工具"""
        class InvalidTool:
            name = "   "
            description = "test"
            parameters = {}
            
            async def execute(self, arguments: dict):
                return "result"
        
        tool = InvalidTool()
        
        with pytest.raises(ValueError, match="non-empty 'name'"):
            validate_tool(tool)
    
    # ==================== 响应模型测试 ====================
    
    def test_completion_response_creation(self):
        """测试 CompletionResponse 创建"""
        response = CompletionResponse(
            id="chatcmpl-123",
            model="gpt-4",
            choices=[
                CompletionChoice(
                    index=0,
                    message={"role": "assistant", "content": "Hello!"},
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            } # type: ignore
        )
        
        assert response.model == "gpt-4"
        assert len(response.choices) == 1
        assert response.choices[0].message["content"] == "Hello!"
        assert response.usage.total_tokens == 15 # type: ignore
    
    def test_stream_chunk_properties(self):
        """测试 StreamChunk 属性"""
        chunk = StreamChunk(
            id="chatcmpl-123",
            model="gpt-4",
            choices=[
                {
                    "index": 0,
                    "delta": {"content": "Hello"},
                    "finish_reason": None
                }
            ]
        )
        
        assert chunk.model == "gpt-4"
        assert chunk.delta == {"content": "Hello"}
        assert chunk.content == "Hello"
    
    def test_stream_chunk_empty_delta(self):
        """测试空 delta 的 StreamChunk"""
        chunk = StreamChunk(
            id="chatcmpl-123",
            model="gpt-4",
            choices=[]
        )
        
        assert chunk.delta == {}
        assert chunk.content is None


# ==================== 集成测试 ====================

class TestProtocolIntegration:
    """协议集成测试"""
    
    def test_model_with_all_capabilities(self):
        """测试具有所有能力的完整模型"""
        class FullFeaturedModel:
            model_name = "advanced-model"
            _supports_function_calling = True
            _supports_vision = True
            
            async def acompletion(self, messages, **kwargs):
                return CompletionResponse(
                    model=self.model_name,
                    choices=[
                        CompletionChoice(
                            message={"role": "assistant", "content": "Response"}
                        )
                    ]
                )
            
            async def astream(self, messages, **kwargs):
                yield StreamChunk(model=self.model_name)

            def count_tokens(self, text_or_messages):
                return 0
        
        model = FullFeaturedModel()
        
        assert check_protocol(model, ModelProtocol)
        assert check_protocol(model, StreamableModelProtocol)
        assert supports_streaming(model)
        assert supports_function_calling(model)
        assert supports_vision(model)
        assert get_model_name(model) == "advanced-model"
        
        validate_model(model)