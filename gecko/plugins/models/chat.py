# gecko/plugins/models/chat.py
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List

import litellm
from pydantic import ValidationError

from gecko.core.exceptions import ModelError
from gecko.core.logging import get_logger
from gecko.core.protocols import CompletionResponse, StreamChunk
from gecko.plugins.models.base import BaseChatModel

logger = get_logger(__name__)

class LiteLLMChatModel(BaseChatModel):
    """
    基于 LiteLLM 的通用 Chat 模型实现
    """

    def _get_params(self, messages: List[Dict[str, Any]], stream: bool, **kwargs: Any) -> Dict[str, Any]:
        """构造 LiteLLM 调用参数，合并配置与运行时参数"""
        params = {
            "model": self.config.model_name,
            "messages": messages,
            "timeout": self.config.timeout,
            "stream": stream,
            **self.config.extra_kwargs,
            **kwargs,
        }

        if self.config.api_key:
            params["api_key"] = self.config.api_key
        if self.config.base_url:
            params["api_base"] = self.config.base_url

        return params

    def _sanitize_response(self, resp: Any) -> Dict[str, Any]:
        """
        [核心修复] 深度清洗 LiteLLM 响应对象
        
        目标：将不可靠的 Pydantic 对象转换为纯 Python 字典，
        避免因 litellm 内部 Schema 校验失败导致 crash。
        """
        # 策略 1: 尝试通过 JSON 序列化 (通常最稳健，因为会忽略 Pydantic 类型检查)
        try:
            if hasattr(resp, "model_dump_json"):
                # Pydantic v2 JSON 序列化
                return json.loads(resp.model_dump_json())
            if hasattr(resp, "json") and callable(resp.json):
                # Pydantic v1 / 传统 JSON 方法
                return json.loads(resp.json()) # type: ignore
        except Exception:
            pass

        # 策略 2: 尝试标准 model_dump (如果 JSON 失败)
        try:
            if hasattr(resp, "model_dump"):
                return resp.model_dump(mode='json') # mode='json' 强制转换类型
            if hasattr(resp, "dict"):
                return resp.dict()
        except Exception:
            pass

        # 策略 3: 手动暴力提取 (Ultimate Fallback)
        # 当 litellm 对象损坏无法 dump 时，手动提取关键字段
        try:
            data = {
                "id": getattr(resp, "id", ""),
                "object": getattr(resp, "object", "chat.completion"),
                "created": getattr(resp, "created", 0),
                "model": getattr(resp, "model", ""),
                "choices": [],
                "usage": None
            }
            
            # 提取 Choices
            raw_choices = getattr(resp, "choices", [])
            if isinstance(raw_choices, list):
                for c in raw_choices:
                    c_dict = {
                        "index": getattr(c, "index", 0),
                        "finish_reason": getattr(c, "finish_reason", "stop"),
                        "message": {}
                    }
                    # 提取 Message
                    msg = getattr(c, "message", None)
                    if msg:
                        c_dict["message"] = {
                            "role": getattr(msg, "role", "assistant"),
                            "content": getattr(msg, "content", None),
                            "tool_calls": getattr(msg, "tool_calls", None)
                        }
                    data["choices"].append(c_dict)
            
            # 提取 Usage
            usage = getattr(resp, "usage", None)
            if usage:
                data["usage"] = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                    "completion_tokens": getattr(usage, "completion_tokens", 0),
                    "total_tokens": getattr(usage, "total_tokens", 0)
                }
            
            return data
            
        except Exception as e:
            logger.error("Manual response extraction failed", error=str(e))
            # 如果到了这一步，说明对象完全不可读，返回空结构防止 crash
            return {"choices": [], "model": "unknown-error"}

    async def acompletion(self, messages: List[Dict[str, Any]], **kwargs: Any) -> CompletionResponse:
        try:
            kwargs.pop("stream", None)

            params = self._get_params(messages, stream=False, **kwargs)
            resp = await litellm.acompletion(**params)
            
            # 使用新的清洗逻辑
            data = self._sanitize_response(resp)
            
            return CompletionResponse(**data)

        except (ValidationError, TypeError) as e:
            logger.error("Failed to parse model response", error=str(e))
            raise ModelError(f"Response parsing failed: {e}") from e
        except Exception as e:
            self._handle_litellm_error(e)
            raise

    async def astream(self, messages: List[Dict[str, Any]], **kwargs: Any) -> AsyncIterator[StreamChunk]: # type: ignore
        try:
            kwargs.pop("stream", None)

            params = self._get_params(messages, stream=True, **kwargs)
            response_iterator = await litellm.acompletion(**params)
            
            async for chunk in response_iterator: # type: ignore
                # 流式 Chunk 同样需要清洗
                data = self._sanitize_response(chunk)
                # 确保 data 不为空且有 choices (防止空包)
                if data and data.get("choices") or data.get("id"):
                    yield StreamChunk(**data)

        except Exception as e:
            self._handle_litellm_error(e)

    def _handle_litellm_error(self, e: Exception) -> None:
        """统一异常映射"""
        msg = str(e)
        error_type = type(e).__name__
        
        if "AuthenticationError" in error_type:
            raise ModelError(f"Authentication failed: {msg}", error_code="AUTH_ERROR") from e
        if "RateLimitError" in error_type:
            raise ModelError(f"Rate limit exceeded: {msg}", error_code="RATE_LIMIT") from e
        if "ContextWindowExceededError" in error_type:
             raise ModelError(f"Context window exceeded: {msg}", error_code="CONTEXT_LIMIT") from e
             
        raise ModelError(f"Model execution failed ({error_type}): {msg}") from e