# gecko/plugins/models/presets/openai.py
from __future__ import annotations

from typing import Any

from gecko.plugins.models.config import ModelConfig
from gecko.plugins.models.drivers.litellm_driver import LiteLLMDriver
from gecko.plugins.models.embedding import LiteLLMEmbedder


class OpenAIChat(LiteLLMDriver):
    """OpenAI Chat 模型预设"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o", **kwargs: Any):
        config = ModelConfig(
            model_name=model,
            driver_type="litellm",
            api_key=api_key,
            supports_vision=True,
            supports_function_calling=True,
            **kwargs
        )
        super().__init__(config)


class OpenAIEmbedder(LiteLLMEmbedder):
    """OpenAI Embedding 模型预设"""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", dimension: int = 1536, **kwargs: Any):
        config = ModelConfig(
            model_name=model,
            api_key=api_key,
            **kwargs
        )
        super().__init__(config, dimension=dimension)