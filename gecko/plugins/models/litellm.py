# gecko/plugins/models/litellm.py
from __future__ import annotations

import os
from typing import Any

import litellm
from pydantic import BaseModel


class LiteLLMModel(BaseModel):
    """
    Gecko 官方推荐模型适配器：支持任意 OpenAI-compatible 私有部署
    用法示例：
        .with_model(
            model="kimi-k2-thinking",
            base_url="http://172.19.37.104:8095/v1",
            api_key="optional",
            temperature=0.7
        )
    """
    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout: float = 300.0
    # 支持所有 litellm 参数
    extra_kwargs: dict[str, Any] = {}

    def model_post_init(self, __context) -> None:
        # 优先使用显式传入的参数
        if self.base_url:
            litellm.api_base = self.base_url
        if self.api_key:
            os.environ.setdefault("OPENAI_API_KEY", self.api_key)

    async def acompletion(self, messages: list[dict], **kwargs) -> Any:
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "custom_llm_provider": "openai",  # 关键：强制走 OpenAI 协议
            **self.extra_kwargs,
            **kwargs,
        }
        if self.base_url:
            params["api_base"] = self.base_url
        if self.api_key:
            params["api_key"] = self.api_key

        return await litellm.acompletion(**params)