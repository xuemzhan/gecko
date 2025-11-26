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
        """
        预加载 Tokenizer 到内存
        
        修复: 改进模型匹配逻辑，支持更多模型类型
        """
        model_lower = self.config.model_name.lower()
        
        try:
            import tiktoken
            
            # 修复: 更精确的模型匹配
            if any(k in model_lower for k in ["gpt-4", "gpt-3.5", "gpt-4o"]):
                # OpenAI GPT 系列 - 尝试精确匹配
                try:
                    self._tokenizer = tiktoken.encoding_for_model(self.config.model_name)
                except KeyError:
                    # 回退到 cl100k_base (GPT-4 默认)
                    self._tokenizer = tiktoken.get_encoding("cl100k_base")
                    
            elif "text-embedding" in model_lower:
                # OpenAI Embedding 模型
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
                
            elif any(k in model_lower for k in ["claude", "anthropic"]):
                # 修复: Claude 使用近似编码
                # 注意: 这不是精确的，但对于 token 计数足够用
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
                logger.debug("Using cl100k_base as approximation for Claude")
                
            elif any(k in model_lower for k in ["llama", "mistral", "qwen", "yi"]):
                # 修复: 开源模型使用通用编码
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
                logger.debug(f"Using cl100k_base as approximation for {self.config.model_name}")
                
            elif any(k in model_lower for k in ["glm", "zhipu", "chatglm"]):
                # 修复: 智谱模型
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
                logger.debug("Using cl100k_base as approximation for GLM")
                
            else:
                # 未知模型，尝试通用编码
                logger.debug(f"Unknown model {self.config.model_name}, using cl100k_base")
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
                
        except ImportError:
            logger.debug("tiktoken not available, token counting will use estimation")
        except Exception as e:
            logger.debug(f"Tokenizer preload failed: {e}")

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