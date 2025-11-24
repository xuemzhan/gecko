# gecko/plugins/models/drivers/litellm_driver.py
from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Union

import litellm
import tiktoken

from gecko.core.exceptions import ModelError
from gecko.core.logging import get_logger
from gecko.core.protocols import CompletionResponse, StreamChunk
from gecko.plugins.models.adapter import LiteLLMAdapter
from gecko.plugins.models.base import BaseChatModel
from gecko.plugins.models.config import ModelConfig
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
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._tokenizer = None
        # [优化] 初始化时预加载 Tokenizer，避免在推理热路径中触发文件 IO
        self._preload_tokenizer()

    def _preload_tokenizer(self):
        """预加载 Tokenizer 到内存 (主要针对 OpenAI/Tiktoken 兼容模型)"""
        try:
            # 如果是 GPT 系列或兼容模型，预加载 tiktoken encoding
            if any(k in self.config.model_name.lower() for k in ["gpt", "text-embedding", "claude"]):
                # 注意：Claude 其实不完全用 cl100k_base，但这里仅作示例优化
                # 生产环境可根据 model_name 映射不同的 encoding
                self._tokenizer = tiktoken.encoding_for_model(self.config.model_name)
        except Exception:
            # 加载失败不报错，运行时降级处理
            pass

    # [实现] 高性能、非阻塞的计数方法
    def count_tokens(self, text_or_messages: Union[str, List[Dict[str, Any]]]) -> int:
        try:
            # 1. 统一转换为文本字符串
            text = self._to_text(text_or_messages)

            # 2. 快速路径：如果预加载了 tiktoken，直接在 C++ 层计算 (极快)
            if self._tokenizer:
                return len(self._tokenizer.encode(text))

            # 3. 慢速路径：LiteLLM encode
            # 仅当文本较短 (<10k chars) 时调用，防止大文本触发 heavy computation 阻塞 Loop
            if len(text) < 10000:
                # litellm.encode 内部有缓存，但仍有 Python 层开销
                return len(litellm.encode(model=self.config.model_name, text=text)) # type: ignore
            
            # 4. 兜底策略：字符长度估算 (避免卡死)
            # 中文约 0.6 token/char, 英文约 0.25 -> 平均取 0.33
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