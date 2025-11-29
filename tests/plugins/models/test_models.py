# tests/plugins/models/test_models.py
import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
# 引入 SimpleNamespace 来模拟对象属性访问
from types import SimpleNamespace

from gecko.plugins.models.drivers.litellm_driver import LiteLLMDriver
from gecko.plugins.models.config import ModelConfig
from gecko.plugins.models.presets.zhipu import ZhipuChat
from gecko.core.protocols import CompletionResponse, StreamChunk
from gecko.core.exceptions import ModelError

# ===================== 1. 单元测试 (Mock) =====================

@pytest.mark.asyncio
async def test_litellm_driver_completion():
    """[Unit] 测试 LiteLLM 驱动的清洗逻辑"""
    # [修复] 使用 SimpleNamespace 模拟真实对象结构，避免 MagicMock 的无限递归属性陷阱
    # 这比配置复杂的 MagicMock 更简单且行为更确定
    
    mock_usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15
    )
    
    mock_message = SimpleNamespace(
        role="assistant",
        content="Cleaned Content",
        tool_calls=None
    )
    
    mock_choice = SimpleNamespace(
        index=0,
        finish_reason="stop",
        message=mock_message,
        logprobs=None # 显式设置为 None，满足 safe_access 的预期
    )

    mock_obj = SimpleNamespace(
        id="test-id",
        object="chat.completion",
        created=1234567890,
        model="gpt-mock",
        choices=[mock_choice],
        usage=mock_usage,
        system_fingerprint=None,
        _hidden_params={} # 模拟 litellm 可能存在的私有字段
    )

    config = ModelConfig(model_name="gpt-mock", api_key="mock")
    driver = LiteLLMDriver(config)

    with patch("gecko.plugins.models.drivers.litellm_driver.litellm.acompletion", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = mock_obj
        
        resp = await driver.acompletion([{"role": "user", "content": "hi"}])
        
        assert isinstance(resp, CompletionResponse)
        assert resp.choices[0].message["content"] == "Cleaned Content"
        assert resp.model == "gpt-mock"
        assert resp.usage.total_tokens == 15 # type: ignore

@pytest.mark.asyncio
async def test_litellm_driver_error():
    """[Unit] 测试异常映射"""
    config = ModelConfig(model_name="gpt-mock")
    driver = LiteLLMDriver(config)

    with patch("gecko.plugins.models.drivers.litellm_driver.litellm.acompletion", side_effect=Exception("Auth Failed")):
        with pytest.raises(ModelError) as exc:
            await driver.acompletion([])
        assert "LiteLLM execution failed" in str(exc.value)

# ===================== 2. 集成测试 (Zhipu Live) =====================

ZHIPU_KEY = os.getenv("ZHIPU_API_KEY")
should_run = pytest.mark.skipif(not ZHIPU_KEY, reason="No ZHIPU_API_KEY")

@should_run
@pytest.mark.asyncio
async def test_zhipu_live_completion():
    """[Integration] Zhipu 真实调用"""
    model = ZhipuChat(api_key=ZHIPU_KEY, model="glm-4-flash") # type: ignore
    messages = [{"role": "user", "content": "1+1=?"}]
    
    resp = await model.acompletion(messages)
    
    # 验证是否触发 Pydantic 警告 (如果测试通过且无警告输出，则 Adapter 工作正常)
    assert isinstance(resp, CompletionResponse)
    assert "2" in resp.choices[0].message["content"]

@should_run
@pytest.mark.asyncio
async def test_zhipu_live_stream():
    """[Integration] Zhipu 流式调用"""
    model = ZhipuChat(api_key=ZHIPU_KEY, model="glm-4-flash") # type: ignore
    messages = [{"role": "user", "content": "Hello"}]
    
    text = ""
    async for chunk in model.astream(messages):
        assert isinstance(chunk, StreamChunk)
        if chunk.content:
            text += chunk.content
    
    assert len(text) > 0

@pytest.mark.asyncio
async def test_litellm_driver_count_tokens_strategies():
    """[New] 测试 LiteLLMDriver 的多级计数策略"""
    from gecko.plugins.models.drivers.litellm_driver import LiteLLMDriver
    
    # 1. 测试 Tiktoken 路径 (模拟 gpt-4)
    config_gpt = ModelConfig(model_name="gpt-4")
    driver_gpt = LiteLLMDriver(config_gpt)
    
    # 确保 tokenizer 被加载 (环境中有 tiktoken)
    assert driver_gpt._tokenizer is not None
    count = driver_gpt.count_tokens("hello world")
    assert count > 0
    
    # 2. 测试 Fallback 路径 (模拟 Tiktoken 缺失)
    # ✅ 修复: 模拟 tiktoken 未安装，迫使 _preload_tokenizer 失败，从而 _tokenizer 为 None
    with patch.dict(sys.modules, {"tiktoken": None}):
        config_unknown = ModelConfig(model_name="unknown-model-123")
        driver_unknown = LiteLLMDriver(config_unknown)
        
        # 此时应该为 None (加载失败)
        assert driver_unknown._tokenizer is None
        
        # 调用计数，应该走字符估算或 litellm
        # "hello" (5 chars) // 3 = 1
        count_fallback = driver_unknown.count_tokens("hello") 
        assert count_fallback > 0

@pytest.mark.asyncio
async def test_litellm_response_sanitization():
    """[New] 测试 LiteLLM 响应清洗逻辑 (防止 Pydantic 崩溃)"""
    from gecko.plugins.models.drivers.litellm_driver import LiteLLMDriver
    from gecko.plugins.models.config import ModelConfig
    
    driver = LiteLLMDriver(ModelConfig(model_name="gpt-3.5"))
    
    # 1. 模拟一个损坏的 Pydantic 对象 (model_dump 报错)
    class BrokenObj:
        def model_dump(self, **kwargs):
            raise ValueError("Dump failed")
        
        # 只有属性访问有效
        id = "123"
        object = "chat.completion"
        created = 111
        model = "gpt"
        choices = []
        usage = None
        
    cleaned = driver._sanitize_response(BrokenObj()) # type: ignore
    
    # 验证暴力提取生效
    assert cleaned["id"] == "123"
    assert cleaned["model"] == "gpt"
    assert isinstance(cleaned["choices"], list)

@pytest.mark.asyncio
async def test_litellm_token_count_fallback():
    """[New] 测试 Token 计数回退逻辑"""
    from gecko.plugins.models.drivers.litellm_driver import LiteLLMDriver
    from gecko.plugins.models.config import ModelConfig
    
    # 模拟 Tiktoken 不存在
    with patch.dict("sys.modules", {"tiktoken": None}):
        driver = LiteLLMDriver(ModelConfig(model_name="gpt-4"))
        
        # 确保 tokenizer 为 None
        assert driver._tokenizer is None
        
        # 测试估算逻辑
        text = "hello world"
        count = driver.count_tokens(text)
        
        # "hello world" (11 chars) // 3 = 3
        assert count == 3