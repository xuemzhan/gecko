# tests/plugins/models/test_models.py
import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace

from gecko.plugins.models.drivers.litellm_driver import LiteLLMDriver
from gecko.plugins.models.config import ModelConfig
from gecko.core.protocols import CompletionResponse, StreamChunk
from gecko.plugins.models.exceptions import (
    AuthenticationError,
    RateLimitError,
    ContextWindowExceededError,
    ServiceUnavailableError,
    ProviderError
)
from gecko.core.resilience import CircuitOpenError, CircuitState

# ===================== 1. 基础功能测试 =====================

@pytest.mark.asyncio
async def test_litellm_driver_completion():
    """[Unit] 测试 LiteLLM 驱动的清洗逻辑"""
    mock_usage = SimpleNamespace(
        prompt_tokens=10, completion_tokens=5, total_tokens=15
    )
    mock_message = SimpleNamespace(
        role="assistant", content="Cleaned Content", tool_calls=None
    )
    mock_choice = SimpleNamespace(
        index=0, finish_reason="stop", message=mock_message, logprobs=None
    )
    mock_obj = SimpleNamespace(
        id="test-id", object="chat.completion", created=1234567890,
        model="gpt-mock", choices=[mock_choice], usage=mock_usage,
        system_fingerprint=None, _hidden_params={}
    )

    config = ModelConfig(model_name="gpt-mock", api_key="mock")
    driver = LiteLLMDriver(config)

    with patch("gecko.plugins.models.drivers.litellm_driver.litellm.acompletion", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_obj
        resp = await driver.acompletion([{"role": "user", "content": "hi"}])
        assert isinstance(resp, CompletionResponse)
        assert resp.choices[0].message["content"] == "Cleaned Content"
        assert resp.usage.total_tokens == 15 # type: ignore

@pytest.mark.asyncio
async def test_litellm_driver_error():
    """[Unit] 测试异常映射"""
    config = ModelConfig(model_name="gpt-mock")
    driver = LiteLLMDriver(config)

    with patch("gecko.plugins.models.drivers.litellm_driver.litellm.acompletion", 
               side_effect=Exception("Generic Error")):
        with pytest.raises(ProviderError) as exc:
            await driver.acompletion([])
        assert "Unknown provider error" in str(exc.value)

# ===================== 2. Token 计数策略测试 =====================

@pytest.mark.asyncio
async def test_litellm_driver_count_tokens_strategies():
    """[New] 测试 LiteLLMDriver 的多级计数策略: Tiktoken 优先"""
    mock_encoding = MagicMock()
    mock_encoding.encode.return_value = [1, 2, 3] 
    
    mock_tiktoken = MagicMock()
    mock_tiktoken.encoding_for_model.return_value = mock_encoding
    mock_tiktoken.get_encoding.return_value = mock_encoding
    
    # [关键] 注入 mock 的 tiktoken
    with patch.dict("sys.modules", {"tiktoken": mock_tiktoken}):
        config_gpt = ModelConfig(model_name="gpt-4")
        driver_gpt = LiteLLMDriver(config_gpt)
        
        assert driver_gpt._tokenizer is not None
        count = driver_gpt.count_tokens("hello world")
        assert count == 3

@pytest.mark.asyncio
async def test_litellm_token_count_fallback():
    """[New] 测试 Token 计数回退逻辑: Tiktoken 缺失 -> Char Estimate"""
    
    # 模拟 Tiktoken 不存在
    with patch.dict("sys.modules", {"tiktoken": None}):
        driver = LiteLLMDriver(ModelConfig(model_name="gpt-4"))
        
        # Tokenizer 应该加载失败
        assert driver._tokenizer is None
        
        text = "hello world"
        count = driver.count_tokens(text)
        
        # 逻辑：tiktoken 缺失 (_tiktoken_available=False) -> 直接走字符估算 (step 2)
        # "hello world" (11 chars) // 3 = 3
        assert count == 3

# ===================== 3. 熔断器集成测试 =====================

@pytest.mark.asyncio
async def test_litellm_driver_circuit_breaker_trigger():
    """[Phase 3] 验证 LiteLLMDriver 集成熔断器"""
    config = ModelConfig(model_name="mock")
    driver = LiteLLMDriver(config)
    driver._circuit_breaker.failure_threshold = 2
    
    # [关键] 使用真实的 ServiceUnavailableError，以便熔断器识别
    error_instance = ServiceUnavailableError("503 Service Unavailable")
    
    with patch("gecko.plugins.models.drivers.litellm_driver.litellm.acompletion", 
               side_effect=error_instance):
        
        # 1. 第一次失败
        with pytest.raises(ServiceUnavailableError):
            await driver.acompletion([])
            
        # 2. 第二次失败 -> 触发熔断
        with pytest.raises(ServiceUnavailableError):
            await driver.acompletion([])
            
        # 验证状态
        assert driver._circuit_breaker._state == CircuitState.OPEN
            
        # 3. 第三次调用 -> 被熔断器拦截
        with pytest.raises(CircuitOpenError):
            await driver.acompletion([])

@pytest.mark.asyncio
async def test_litellm_driver_astream_circuit_breaker():
    """[Gap 4] 验证 astream 方法受熔断器保护"""
    config = ModelConfig(model_name="mock")
    driver = LiteLLMDriver(config)
    driver._circuit_breaker.failure_threshold = 1
    
    # [关键] 使用真实的异常
    error_instance = ServiceUnavailableError("Connection failed")
    
    with patch("gecko.plugins.models.drivers.litellm_driver.litellm.acompletion", 
               side_effect=error_instance):
        
        gen = driver.astream([])
        with pytest.raises(ServiceUnavailableError):
            await gen.__anext__()
            
        assert driver._circuit_breaker._state == CircuitState.OPEN
        
        gen2 = driver.astream([])
        with pytest.raises(CircuitOpenError):
            await gen2.__anext__()