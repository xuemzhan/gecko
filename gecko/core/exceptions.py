# gecko/core/exceptions.py
"""
统一异常与错误码系统

- 提供 GeckoError 及各子类异常（AgentError、ModelError、WorkflowError 等）
- 复用 gecko.core.error_codes 中定义的 ErrorCode / ErrorCategory 等
"""

from __future__ import annotations

from typing import Any, Optional

from gecko.core.error_codes import (  # 复用已有错误码定义
    ErrorCode,
    ErrorCategory,
    ErrorInfo,
    get_error_info,
    is_retryable,
    get_message,
)


class GeckoError(Exception):
    """
    Gecko 框架所有自定义异常的基类。

    属性:
        code: ErrorCode（逻辑错误码，默认为 UNKNOWN）
        context: 额外上下文信息（调试/日志）
        error_code: 兼容部分调用点传入的自定义字符串 error_code
    """

    def __init__(self, message: str, **kwargs: Any):
        # 标准化字段
        self.code: ErrorCode = kwargs.pop("code", ErrorCode.UNKNOWN)
        self.context: dict[str, Any] = kwargs.pop("context", {}) or {}
        # 某些调用点会传入 error_code="AUTH_ERROR" 这类字符串，这里兼容保留
        self.error_code: Any = kwargs.pop("error_code", None)

        super().__init__(message)

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} (code={self.code})"


class AgentError(GeckoError):
    """Agent 级别错误（输入不合法、内部状态异常等）"""
    pass


class ModelError(GeckoError):
    """模型调用相关错误（鉴权失败、限流、上下文超限等）"""
    pass


class ConfigurationError(GeckoError):
    """配置错误（缺少必要参数、无效配置项等）"""
    pass


class ToolError(GeckoError):
    """工具执行错误"""
    pass


class ToolNotFoundError(GeckoError):
    """工具未找到"""

    def __init__(self, tool_name: str, **kwargs: Any):
        msg = f"Tool '{tool_name}' not found"
        super().__init__(msg, **kwargs)
        self.context.setdefault("tool_name", tool_name)


class WorkflowError(GeckoError):
    """Workflow 编排/执行过程中的错误"""
    pass


class WorkflowCycleError(WorkflowError):
    """Workflow 结构中检测到环"""
    pass


class StorageError(GeckoError):
    """存储后端读写错误"""
    pass


__all__ = [
    # 异常类
    "GeckoError",
    "AgentError",
    "ModelError",
    "ConfigurationError",
    "ToolError",
    "ToolNotFoundError",
    "WorkflowError",
    "WorkflowCycleError",
    "StorageError",
    # 错误码相关（从 error_codes 复用）
    "ErrorCode",
    "ErrorCategory",
    "ErrorInfo",
    "get_error_info",
    "is_retryable",
    "get_message",
]