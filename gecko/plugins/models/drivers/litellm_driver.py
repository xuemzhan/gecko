# gecko/plugins/models/drivers/litellm_driver.py
from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List

import litellm

from gecko.core.exceptions import ModelError
from gecko.core.logging import get_logger
from gecko.core.protocols import CompletionResponse, StreamChunk
from gecko.plugins.models.adapter import LiteLLMAdapter
from gecko.plugins.models.base import BaseChatModel
from gecko.plugins.models.registry import register_driver

logger = get_logger(__name__)


@register_driver("litellm")
class LiteLLMDriver(BaseChatModel):
    """
    LiteLLM 通用驱动
    
    特点：
    1. 兼容性最强（支持 OpenAI, Zhipu, Ollama 等）。
    2. 使用 LiteLLMAdapter 进行响应清洗，解决 Pydantic 兼容性问题。
    """

    def _get_params(self, messages: List[Dict[str, Any]], stream: bool, **kwargs: Any) -> Dict[str, Any]:
        """构造调用参数"""
        params = {
            "model": self.config.model_name,
            "messages": messages,
            "timeout": self.config.timeout,
            "stream": stream,
            **self.config.extra_kwargs,
            **kwargs,
        }
        # 显式传递参数，确保并发安全
        if self.config.api_key:
            params["api_key"] = self.config.api_key
        if self.config.base_url:
            params["api_base"] = self.config.base_url
        return params

    async def acompletion(self, messages: List[Dict[str, Any]], **kwargs: Any) -> CompletionResponse:
        try:
            params = self._get_params(messages, stream=False, **kwargs)
            resp = await litellm.acompletion(**params)
            # 使用适配器清洗
            return LiteLLMAdapter.to_gecko_response(resp)
        except Exception as e:
            self._handle_error(e)
            raise

    async def astream(self, messages: List[Dict[str, Any]], **kwargs: Any) -> AsyncIterator[StreamChunk]: # type: ignore
        try:
            params = self._get_params(messages, stream=True, **kwargs)
            iterator = await litellm.acompletion(**params)
            
            async for chunk in iterator: # type: ignore
                # 使用适配器清洗
                gecko_chunk = LiteLLMAdapter.to_gecko_chunk(chunk)
                if gecko_chunk:
                    yield gecko_chunk
        except Exception as e:
            self._handle_error(e)

    def _handle_error(self, e: Exception) -> None:
        msg = str(e)
        err_name = type(e).__name__
        
        if "AuthenticationError" in err_name:
            raise ModelError(f"Auth failed: {msg}", error_code="AUTH_ERROR") from e
        if "RateLimitError" in err_name:
            raise ModelError(f"Rate limit: {msg}", error_code="RATE_LIMIT") from e
        if "ContextWindowExceededError" in err_name:
             raise ModelError(f"Context limit: {msg}", error_code="CONTEXT_LIMIT") from e
             
        raise ModelError(f"LiteLLM execution failed ({err_name}): {msg}") from e