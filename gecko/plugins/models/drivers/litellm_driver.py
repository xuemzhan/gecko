# gecko/plugins/models/drivers/litellm_driver.py
from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Union

import litellm

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
        """
        LiteLLM 通用驱动

        修复点：
        - 增加 _tiktoken_available 标记，用于区分“未安装 tiktoken”与“仅未成功预加载 Tokenizer”
        """
        super().__init__(config)
        self._tokenizer = None
        # 标记当前环境是否可用 tiktoken，用于 count_tokens 回退策略
        self._tiktoken_available: bool = True
        # [优化] 初始化时预加载 Tokenizer，避免在推理热路径中触发文件 IO
        self._preload_tokenizer()

    def _preload_tokenizer(self):
        """
        预加载 Tokenizer 到内存

        修复:
        - 改进模型匹配逻辑，支持更多模型类型
        - 显式设置 _tiktoken_available 标记，为回退逻辑提供依据
        """
        model_lower = self.config.model_name.lower()

        try:
            import tiktoken

            # 只要 import 成功，认为环境“支持 tiktoken”
            self._tiktoken_available = True

            # 修复: 更精确的模型匹配
            if any(k in model_lower for k in ["gpt-4", "gpt-3.5", "gpt-4o"]):
                # OpenAI GPT 系列 - 尝试精确匹配
                try:
                    self._tokenizer = tiktoken.encoding_for_model(
                        self.config.model_name
                    )
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
                logger.debug(
                    "Using cl100k_base as approximation for Claude"
                )

            elif any(k in model_lower for k in ["llama", "mistral", "qwen", "yi"]):
                # 修复: 开源模型使用通用编码
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
                logger.debug(
                    f"Using cl100k_base as approximation for {self.config.model_name}"
                )

            elif any(k in model_lower for k in ["glm", "zhipu", "chatglm"]):
                # 修复: 智谱模型
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
                logger.debug("Using cl100k_base as approximation for GLM")

            else:
                # 未知模型，尝试通用编码
                logger.debug(
                    f"Unknown model {self.config.model_name}, using cl100k_base"
                )
                self._tokenizer = tiktoken.get_encoding("cl100k_base")

        except ImportError:
            # 修复点：明确标记 tiktoken 不可用，count_tokens 直接走估算分支
            self._tiktoken_available = False
            logger.debug(
                "tiktoken not available, token counting will use estimation"
            )
        except Exception as e:
            # 其它异常也视为“当前环境不适合使用 tiktoken”
            self._tiktoken_available = False
            logger.debug(f"Tokenizer preload failed: {e}")

    # [实现] 高性能、非阻塞的计数方法
    def count_tokens(self, text_or_messages: Union[str, List[Dict[str, Any]]]) -> int:
        """
        Token 计数逻辑（多级回退）

        策略：
        1. 若已成功预加载 tiktoken Tokenizer -> 直接 encode（最快）
        2. 若 tiktoken 不可用 (_tiktoken_available=False) -> 直接按字符估算
        3. 若 tiktoken 可用但未成功预加载 -> 尝试 litellm.encode（相对精确）
        4. 若 litellm.encode 失败或大文本 -> 按字符估算兜底
        """
        try:
            # 1. 统一转换为文本字符串
            text = self._to_text(text_or_messages)

            # 2. 快速路径：如果预加载了 tiktoken，直接在 C++ 层计算 (极快)
            if self._tokenizer is not None:
                return len(self._tokenizer.encode(text))

            # 3. 若 tiktoken 确认不可用，则直接走估算逻辑
            #    这是测试中模拟 "tiktoken 不存在" 的主要分支
            if not getattr(self, "_tiktoken_available", True):
                return len(text) // 3

            # 4. 慢速路径：调用 LiteLLM 的 encode 做尽量精确的估算
            #    仅当文本较短 (<10k chars) 时调用，防止大文本触发 heavy computation 阻塞 Loop
            if len(text) < 10000:
                # litellm.encode 内部有缓存，但仍有 Python 层开销
                return len(
                    litellm.encode( # type: ignore
                        model=self.config.model_name, text=text
                    )  # type: ignore
                )

            # 5. 兜底策略：字符长度估算 (避免卡死)
            #    中文约 0.6 token/char, 英文约 0.25 -> 平均取 0.33
            return len(text) // 3

        except Exception:
            # 最后的防线：任意异常都退回字符估算
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
    
    def _sanitize_response(self, raw: Any) -> Dict[str, Any]:
        """
        兼容旧版测试的响应清洗方法（LiteLLMDriver 私有方法）

        注意：
        - 实际生产路径已经通过 LiteLLMAdapter 进行标准化。
        - 这里主要用于单元测试中构造的“损坏对象”场景：
          - model_dump() 抛异常
          - 但基本属性仍然可通过 getattr 访问
        """
        # 1. 优先尝试 model_dump / dict 等“标准方式”
        try:
            if hasattr(raw, "model_dump"):
                return raw.model_dump()  # type: ignore[no-any-return]
            if hasattr(raw, "dict"):
                return raw.dict()  # type: ignore[no-any-return]
        except Exception:
            # model_dump / dict 失败则退回属性访问
            pass

        # 2. 暴力属性访问 fallback：尽量拼出一个 OpenAI 风格的响应字典
        def g(attr: str, default: Any = None) -> Any:
            return getattr(raw, attr, default)

        return {
            "id": g("id"),
            "object": g("object"),
            "created": g("created"),
            "model": g("model"),
            "choices": g("choices", []),
            "usage": g("usage", None),
        }
