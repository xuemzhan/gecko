# gecko/plugins/models/presets/zhipu.py
from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List

from gecko.core.protocols import CompletionResponse, StreamChunk
from gecko.plugins.models.config import ModelConfig
from gecko.plugins.models.drivers.litellm_driver import LiteLLMDriver


class ZhipuChat(LiteLLMDriver):
    """
    智谱 AI (GLM) 预设
    
    继承自 LiteLLMDriver，复用其稳定的清洗逻辑和 OpenAI 协议。
    """
    
    def __init__(self, api_key: str, model: str = "glm-4-plus", **kwargs: Any):
        # 自动判断视觉支持
        is_vision = "v" in model.lower() or "vision" in model.lower()
        
        config = ModelConfig(
            model_name=model,
            # 显式指定使用 litellm 驱动
            driver_type="litellm",
            api_key=api_key,
            # 智谱官方 OpenAI 兼容接口
            base_url="https://open.bigmodel.cn/api/paas/v4/",
            # 强制 LiteLLM 使用 openai 协议处理
            extra_kwargs={"custom_llm_provider": "openai"},
            supports_vision=is_vision,
            supports_function_calling=True,
            **kwargs
        )
        super().__init__(config)
