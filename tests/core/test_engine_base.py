# tests/core/engine/test_base.py
import pytest
from gecko.core import memory, toolbox
from gecko.core.engine.base import CognitiveEngine, ExecutionStats
from gecko.core.output import AgentOutput

class TestCognitiveEngine:
    def test_abstract_class(self):
        """测试抽象类不能实例化"""
        with pytest.raises(TypeError):
            CognitiveEngine(model=None, toolbox=None, memory=None)
    
    def test_model_validation(self):
        """测试模型验证"""
        class InvalidModel:
            pass
        
        class ConcreteEngine(CognitiveEngine):
            async def step(self, input_messages):
                return AgentOutput(content="test")
        
        with pytest.raises(TypeError, match="ModelProtocol"):
            ConcreteEngine(InvalidModel(), toolbox, memory)
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
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
            
            async def step(self, input_messages):
                return AgentOutput(content="test")
        
        async with TestEngine(model, toolbox, memory):
            assert initialized
        
        assert cleaned_up