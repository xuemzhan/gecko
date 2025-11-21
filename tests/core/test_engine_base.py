# tests/core/engine/test_base.py
import pytest
from gecko.core.engine.base import CognitiveEngine, ExecutionStats
from gecko.core.output import AgentOutput

class TestCognitiveEngine:
    def test_abstract_class(self):
        """测试抽象类不能实例化"""
        with pytest.raises(TypeError):
            CognitiveEngine(model=None, toolbox=None, memory=None)
    
    def test_model_validation(self, toolbox, memory):
        """测试模型验证"""
        class InvalidModel:
            pass
        
        class ConcreteEngine(CognitiveEngine):
            async def step(self, input_messages):
                return AgentOutput(content="test")
        
        with pytest.raises(TypeError, match="ModelProtocol"):
            ConcreteEngine(InvalidModel(), toolbox, memory)
    
    @pytest.mark.asyncio
    async def test_context_manager(self, model, toolbox, memory):  # <--- 修复点：添加缺失的参数
        """测试上下文管理器"""
        initialized = False
        cleaned_up = False
    
        class TestEngine(CognitiveEngine):
            async def initialize(self):
                nonlocal initialized
                initialized = True
    
            async def cleanup(self):
                nonlocal cleaned_up
                cleaned_up = True
    
            async def step(self, input_messages, **kwargs): # 建议加上 **kwargs 以匹配基类签名
                return AgentOutput(content="test")
    
        # 现在 model, toolbox, memory 会由 pytest 自动注入
        async with TestEngine(model, toolbox, memory) as engine:
            assert initialized is True
            assert isinstance(engine, TestEngine)
            
        assert cleaned_up is True