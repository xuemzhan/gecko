# gecko/plugins/models/exceptions.py
from gecko.core.exceptions import GeckoError, ErrorCode, ModelError

class ProviderError(ModelError):
    """
    模型服务商基础异常 (v0.4)
    继承自 ModelError 以保持向后兼容，确保上层业务逻辑能捕获所有模型错误。
    """
    def __init__(self, message: str, provider: str = "unknown", **kwargs):
        # 默认使用 MODEL_UNAVAILABLE，子类可覆盖
        code = kwargs.pop("code", ErrorCode.MODEL_UNAVAILABLE)
        super().__init__(message, code=code, **kwargs)
        self.context["provider"] = provider

class ContextWindowExceededError(ProviderError):
    """上下文窗口超限 (错误码: MODEL_CONTEXT_EXCEEDED)"""
    def __init__(self, message: str, max_tokens: int = 0, **kwargs):
        kwargs["code"] = ErrorCode.MODEL_CONTEXT_EXCEEDED
        super().__init__(message, **kwargs)
        self.context["max_tokens"] = max_tokens

class RateLimitError(ProviderError):
    """速率限制 (错误码: MODEL_RATE_LIMITED)"""
    def __init__(self, message: str, retry_after: float = 0.0, **kwargs):
        kwargs["code"] = ErrorCode.MODEL_RATE_LIMITED
        super().__init__(message, **kwargs)
        self.context["retry_after"] = retry_after

class AuthenticationError(ProviderError):
    """鉴权失败 (错误码: MODEL_AUTH_FAILED)"""
    def __init__(self, message: str, **kwargs):
        kwargs["code"] = ErrorCode.MODEL_AUTH_FAILED
        super().__init__(message, **kwargs)

class ServiceUnavailableError(ProviderError):
    """服务不可用 (5xx)"""
    pass