# gecko/core/error_codes.py
"""
统一错误码系统

为框架提供结构化的错误分类和标识。
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, NamedTuple, Optional


class ErrorCategory(str, Enum):
    """错误类别"""
    GENERAL = "general"
    MODEL = "model"
    TOOL = "tool"
    STORAGE = "storage"
    WORKFLOW = "workflow"
    VALIDATION = "validation"
    SECURITY = "security"


class ErrorInfo(NamedTuple):
    """错误信息"""
    category: ErrorCategory
    message: str
    retryable: bool
    severity: str  # "info", "warning", "error", "critical"


class ErrorCode(str, Enum):
    """错误码枚举"""
    # 通用
    UNKNOWN = "UNKNOWN"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"

    # 验证
    VALIDATION_FAILED = "VALIDATION_FAILED"
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_CONFIG = "INVALID_CONFIG"

    # 模型
    MODEL_AUTH_FAILED = "MODEL_AUTH_FAILED"
    MODEL_RATE_LIMITED = "MODEL_RATE_LIMITED"
    MODEL_CONTEXT_EXCEEDED = "MODEL_CONTEXT_EXCEEDED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"

    # 工具
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    TOOL_EXECUTION_FAILED = "TOOL_EXECUTION_FAILED"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"

    # 存储
    STORAGE_CONNECTION_FAILED = "STORAGE_CONNECTION_FAILED"
    STORAGE_READ_FAILED = "STORAGE_READ_FAILED"
    STORAGE_WRITE_FAILED = "STORAGE_WRITE_FAILED"

    # 工作流
    WORKFLOW_VALIDATION_FAILED = "WORKFLOW_VALIDATION_FAILED"
    WORKFLOW_CYCLE_DETECTED = "WORKFLOW_CYCLE_DETECTED"
    WORKFLOW_MAX_STEPS_EXCEEDED = "WORKFLOW_MAX_STEPS_EXCEEDED"

    # 安全
    SECURITY_INJECTION_DETECTED = "SECURITY_INJECTION_DETECTED"


# 错误码元数据
_ERROR_INFO: Dict[ErrorCode, ErrorInfo] = {
    ErrorCode.UNKNOWN: ErrorInfo(
        ErrorCategory.GENERAL, "An unknown error occurred", False, "error"
    ),
    ErrorCode.TIMEOUT: ErrorInfo(
        ErrorCategory.GENERAL, "Operation timed out", True, "warning"
    ),
    ErrorCode.CANCELLED: ErrorInfo(
        ErrorCategory.GENERAL, "Operation was cancelled", False, "info"
    ),
    ErrorCode.VALIDATION_FAILED: ErrorInfo(
        ErrorCategory.VALIDATION, "Validation failed", False, "error"
    ),
    ErrorCode.INVALID_INPUT: ErrorInfo(
        ErrorCategory.VALIDATION, "Invalid input provided", False, "error"
    ),
    ErrorCode.INVALID_CONFIG: ErrorInfo(
        ErrorCategory.VALIDATION, "Invalid configuration", False, "error"
    ),
    ErrorCode.MODEL_AUTH_FAILED: ErrorInfo(
        ErrorCategory.MODEL, "Model authentication failed", False, "error"
    ),
    ErrorCode.MODEL_RATE_LIMITED: ErrorInfo(
        ErrorCategory.MODEL, "Model rate limit exceeded", True, "warning"
    ),
    ErrorCode.MODEL_CONTEXT_EXCEEDED: ErrorInfo(
        ErrorCategory.MODEL, "Model context window exceeded", False, "error"
    ),
    ErrorCode.MODEL_UNAVAILABLE: ErrorInfo(
        ErrorCategory.MODEL, "Model service unavailable", True, "error"
    ),
    ErrorCode.TOOL_NOT_FOUND: ErrorInfo(
        ErrorCategory.TOOL, "Tool not found", False, "error"
    ),
    ErrorCode.TOOL_EXECUTION_FAILED: ErrorInfo(
        ErrorCategory.TOOL, "Tool execution failed", False, "error"
    ),
    ErrorCode.TOOL_TIMEOUT: ErrorInfo(
        ErrorCategory.TOOL, "Tool execution timed out", True, "warning"
    ),
    ErrorCode.STORAGE_CONNECTION_FAILED: ErrorInfo(
        ErrorCategory.STORAGE, "Storage connection failed", True, "error"
    ),
    ErrorCode.STORAGE_READ_FAILED: ErrorInfo(
        ErrorCategory.STORAGE, "Storage read operation failed", True, "error"
    ),
    ErrorCode.STORAGE_WRITE_FAILED: ErrorInfo(
        ErrorCategory.STORAGE, "Storage write operation failed", True, "error"
    ),
    ErrorCode.WORKFLOW_VALIDATION_FAILED: ErrorInfo(
        ErrorCategory.WORKFLOW, "Workflow validation failed", False, "error"
    ),
    ErrorCode.WORKFLOW_CYCLE_DETECTED: ErrorInfo(
        ErrorCategory.WORKFLOW, "Workflow cycle detected", False, "error"
    ),
    ErrorCode.WORKFLOW_MAX_STEPS_EXCEEDED: ErrorInfo(
        ErrorCategory.WORKFLOW, "Workflow max steps exceeded", False, "error"
    ),
    ErrorCode.SECURITY_INJECTION_DETECTED: ErrorInfo(
        ErrorCategory.SECURITY, "Potential injection attack detected", False, "critical"
    ),
}


def get_error_info(code: ErrorCode) -> ErrorInfo:
    """获取错误信息"""
    return _ERROR_INFO.get(
        code,
        ErrorInfo(ErrorCategory.GENERAL, str(code), False, "error")
    )


def is_retryable(code: ErrorCode) -> bool:
    """判断错误是否可重试"""
    return get_error_info(code).retryable


def get_message(code: ErrorCode) -> str:
    """获取错误消息"""
    return get_error_info(code).message


__all__ = [
    "ErrorCode",
    "ErrorCategory",
    "ErrorInfo",
    "get_error_info",
    "is_retryable",
    "get_message",
]