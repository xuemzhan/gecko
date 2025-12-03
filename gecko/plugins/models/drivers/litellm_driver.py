# gecko/plugins/models/drivers/litellm_driver.py
"""
LiteLLM 驱动 (v0.4 Phase 3 Complete)

集成特性：
1. Tokenizer 预加载与多级回退计数 (Tiktoken -> Char Estimation)。
2. 熔断器 (Circuit Breaker) 保护，防止服务雪崩。
3. 统一异常映射 (ProviderError 体系)。
4. 响应适配 (LiteLLMAdapter) 清洗 Pydantic 数据。
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Union

import litellm

from gecko.core.exceptions import ModelError, GeckoError
from gecko.core.logging import get_logger
from gecko.core.protocols import CompletionResponse, StreamChunk
from gecko.core.resilience import CircuitBreaker
from gecko.plugins.models.adapter import LiteLLMAdapter
from gecko.plugins.models.base import BaseChatModel
from gecko.plugins.models.config import ModelConfig
from gecko.plugins.models.registry import register_driver
from gecko.plugins.models.exceptions import (
    AuthenticationError,
    RateLimitError,
    ContextWindowExceededError,
    ServiceUnavailableError,
    ProviderError
)

logger = get_logger(__name__)


@register_driver("litellm")
class LiteLLMDriver(BaseChatModel):
    """
    LiteLLM 通用驱动实现
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._tokenizer = None
        # 标记 tiktoken 是否可用/加载成功，用于 count_tokens 决策
        self._tiktoken_available: bool = True
        
        # 1. 预加载 Tokenizer (优化冷启动性能)
        self._preload_tokenizer()
        
        # 2. 初始化熔断器
        # 仅针对 服务不可用(5xx) 和 速率限制(429) 进行熔断
        # 认证错误(401)和上下文超限(400)属于不可恢复的业务/配置错误，不应触发熔断
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
            monitor_exceptions=(ServiceUnavailableError, RateLimitError)
        )

    def _preload_tokenizer(self):
        """
        预加载 Tokenizer 到内存
        
        尝试根据模型名称加载最合适的 tiktoken 编码器。
        如果失败，标记 _tiktoken_available = False。
        """
        model_lower = self.config.model_name.lower()

        try:
            import tiktoken
            self._tiktoken_available = True

            # 启发式匹配编码器
            if any(k in model_lower for k in ["gpt-4", "gpt-3.5", "gpt-4o"]):
                try:
                    self._tokenizer = tiktoken.encoding_for_model(self.config.model_name)
                except KeyError:
                    self._tokenizer = tiktoken.get_encoding("cl100k_base")
            
            elif "text-embedding" in model_lower:
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            
            elif any(k in model_lower for k in ["claude", "anthropic", "llama", "mistral", "glm"]):
                # 非 OpenAI 模型使用 cl100k_base 作为近似估算 (足够通用)
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            
            else:
                # 默认 fallback
                self._tokenizer = tiktoken.get_encoding("cl100k_base")

        except ImportError:
            self._tiktoken_available = False
            logger.debug("tiktoken not installed, token counting will use estimation")
        except Exception as e:
            self._tiktoken_available = False
            logger.debug(f"Tokenizer preload failed: {e}")

    def count_tokens(self, text_or_messages: Union[str, List[Dict[str, Any]]]) -> int:
        """
        计算 Token 数量 (多级回退策略)
        """
        try:
            text = self._to_text(text_or_messages)

            # 1. 优先使用本地 Tokenizer (最快，C++实现)
            if self._tokenizer is not None:
                return len(self._tokenizer.encode(text))

            # 2. 如果 tiktoken 根本不可用，直接走字符估算
            if not getattr(self, "_tiktoken_available", True):
                # 中文约 0.6 token/char, 英文约 0.25 -> 平均保守取 0.33
                return len(text) // 3

            # 3. 尝试调用 LiteLLM 的 encode (较慢，有 Python 开销)
            # 仅对短文本使用，避免阻塞 Event Loop
            if len(text) < 10000:
                try:
                    return len(litellm.encode(model=self.config.model_name, text=text)) # type: ignore
                except Exception:
                    pass

            # 4. 最终兜底：字符估算
            return len(text) // 3

        except Exception:
            # 最后的防线
            return len(str(text_or_messages)) // 3

    def _to_text(self, inp: Union[str, List[Dict[str, Any]]]) -> str:
        """辅助函数：将输入转为字符串"""
        if isinstance(inp, str):
            return inp
        if isinstance(inp, list):
            return "".join(str(m.get("content", "")) for m in inp)
        return str(inp)

    def _get_params(self, messages: List[Dict[str, Any]], stream: bool, **kwargs: Any) -> Dict[str, Any]:
        """构造 LiteLLM 调用参数"""
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

    async def acompletion(self, messages: List[Dict[str, Any]], **kwargs: Any) -> CompletionResponse:
        """
        单次生成 (集成熔断器)
        """
        # 定义受保护的执行逻辑
        async def _execute():
            try:
                # 复制参数防止副作用
                call_kwargs = kwargs.copy()
                call_kwargs.pop("stream", None)

                params = self._get_params(messages, stream=False, **call_kwargs)
                resp = await litellm.acompletion(**params)
                
                # 使用适配器清洗响应数据
                return LiteLLMAdapter.to_gecko_response(resp)
            except Exception as e:
                # 将 LiteLLM 异常转换为 Gecko 异常并重新抛出
                self._handle_error(e)
                raise e

        # [Phase 3] 通过熔断器执行
        return await self._circuit_breaker.call(_execute)

    async def astream(self, messages: List[Dict[str, Any]], **kwargs: Any) -> AsyncIterator[StreamChunk]: # type: ignore
        """
        流式生成 (集成熔断器)
        
        注意：熔断器保护的是**连接建立**阶段。
        """
        # 定义生成器工厂
        async def _create_generator():
            call_kwargs = kwargs.copy()
            call_kwargs.pop("stream", None)
            params = self._get_params(messages, stream=True, **call_kwargs)
            return await litellm.acompletion(**params)

        try:
            # 1. 建立连接 (受熔断器保护)
            # 如果此时服务端 503/429，_create_generator 会抛错，触发熔断
            response_iterator = await self._circuit_breaker.call(_create_generator)
            
            # 2. 迭代数据 (流式传输)
            async for chunk in response_iterator: # type: ignore
                gecko_chunk = LiteLLMAdapter.to_gecko_chunk(chunk)
                if gecko_chunk:
                    yield gecko_chunk
        except Exception as e:
            self._handle_error(e)

    def _handle_error(self, e: Exception) -> None:
        """
        统一异常映射逻辑
        """
        # [Fix] 如果已经是框架内部异常（如 CircuitOpenError 或 已转换过的 ModelError），直接透传
        if isinstance(e, GeckoError):
            raise e

        msg = str(e)
        err_name = type(e).__name__
        
        logger.warning(f"LiteLLM raw error: {err_name} - {msg}")

        if "AuthenticationError" in err_name:
            raise AuthenticationError(f"Auth failed: {msg}") from e
            
        if "RateLimitError" in err_name:
            raise RateLimitError(f"Rate limit exceeded: {msg}") from e
            
        if "ContextWindowExceededError" in err_name:
            raise ContextWindowExceededError(f"Context limit exceeded: {msg}") from e
            
        if "ServiceUnavailableError" in err_name or "APIConnectionError" in err_name:
            raise ServiceUnavailableError(f"Service unavailable: {msg}") from e
            
        if "Timeout" in err_name:
            raise ServiceUnavailableError(f"Request timeout: {msg}") from e

        # 兜底
        raise ProviderError(f"Unknown provider error ({err_name}): {msg}") from e