# gecko/plugins/models/zhipu.py  —— 删除继承，只保留协议兼容
from __future__ import annotations

import os
from typing import Any

import litellm
from pydantic import BaseModel, Field

ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

class ZhipuGLM(BaseModel):
    model: str = "glm-4.5-air"
    api_key: str = Field(
        default="3bd5e6fdc377489c80dbb435b84d7560.izN8bDXCVR1FNSYS",
        description="智谱官方 API Key"
    )
    base_url: str = Field(default=ZHIPU_BASE_URL, exclude=True)
    temperature: float = 0.7
    max_tokens: int | None = None
    timeout: float = 300.0
    extra_kwargs: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context) -> None:
        os.environ.setdefault("ZHIPU_API_KEY", self.api_key)

    async def acompletion(self, messages: list[dict], **kwargs) -> Any:
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "custom_llm_provider": "openai",
            "api_base": self.base_url,
            "api_key": self.api_key,
            **self.extra_kwargs,
            **kwargs,
        }
        return await litellm.acompletion(**params)

def glm_4_5_air(api_key: str | None = None, **kwargs) -> ZhipuGLM:
    return ZhipuGLM(api_key=api_key or "3bd5e6fdc377489c80dbb435b84d7560.izN8bDXCVR1FNSYS", **kwargs)