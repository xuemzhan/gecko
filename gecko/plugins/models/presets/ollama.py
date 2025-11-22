# gecko/plugins/models/presets/ollama.py
from __future__ import annotations

from typing import Any

from gecko.plugins.models.config import ModelConfig
from gecko.plugins.models.drivers.litellm_driver import LiteLLMDriver
from gecko.plugins.models.embedding import LiteLLMEmbedder


class OllamaChat(LiteLLMDriver):
    """Ollama 本地 Chat 模型预设"""
    
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434", **kwargs: Any):
        full_model_name = f"ollama/{model}" if not model.startswith("ollama/") else model
        
        config = ModelConfig(
            model_name=full_model_name,
            driver_type="litellm",
            base_url=base_url,
            api_key="ollama",
            timeout=kwargs.pop("timeout", 120.0),
            supports_function_calling=kwargs.pop("supports_function_calling", False),
            **kwargs
        )
        super().__init__(config)


class OllamaEmbedder(LiteLLMEmbedder):
    """Ollama 本地 Embedding 模型预设"""
    
    def __init__(
        self, 
        model: str = "nomic-embed-text", 
        base_url: str = "http://localhost:11434", 
        dimension: int = 768, 
        **kwargs: Any
    ):
        full_model_name = f"ollama/{model}" if not model.startswith("ollama/") else model
        
        config = ModelConfig(
            model_name=full_model_name,
            base_url=base_url,
            api_key="ollama",
            **kwargs
        )
        super().__init__(config, dimension=dimension)