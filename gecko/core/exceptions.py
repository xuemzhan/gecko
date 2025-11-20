# gecko/core/exceptions.py
"""
Gecko 异常体系（改进版）

改进：移除装饰器，提倡显式错误处理
"""
from __future__ import annotations
from typing import Optional, Dict, Any

# ========== 异常基类 ==========

class GeckoError(Exception):
    """
    Gecko 统一异常基类
    
    设计原则：
    1. 包含结构化上下文
    2. 便于日志记录
    3. 支持异常链（from）
    """
    def __init__(
        self,
        message: str,
        *args,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(message, *args, **kwargs)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.context = context or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（便于日志/API 返回）"""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
        }
    
    def __str__(self) -> str:
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} [{ctx_str}]"
        return self.message

# ========== 领域异常 ==========

class AgentError(GeckoError):
    """Agent 执行异常"""
    pass

class ModelError(GeckoError):
    """模型调用异常"""
    pass

class ToolError(GeckoError):
    """工具执行异常"""
    pass

class ToolNotFoundError(ToolError):
    """工具未找到"""
    def __init__(self, tool_name: str):
        super().__init__(
            f"Tool '{tool_name}' not found in registry",
            error_code="TOOL_NOT_FOUND",
            context={"tool_name": tool_name}
        )

class ToolTimeoutError(ToolError):
    """工具超时"""
    def __init__(self, tool_name: str, timeout: float):
        super().__init__(
            f"Tool '{tool_name}' timed out after {timeout}s",
            error_code="TOOL_TIMEOUT",
            context={"tool_name": tool_name, "timeout": timeout}
        )

class WorkflowError(GeckoError):
    """工作流异常"""
    pass

class WorkflowCycleError(WorkflowError):
    """工作流循环依赖"""
    pass

class StorageError(GeckoError):
    """存储异常"""
    pass

class ConfigurationError(GeckoError):
    """配置错误"""
    pass

class ValidationError(GeckoError):
    """验证错误"""
    pass