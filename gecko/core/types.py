# gecko/core/types.py
"""
类型定义模块

提供框架级别的类型定义。
"""
from __future__ import annotations

from typing import Any, Dict, Generic, List, Literal, Optional, TypeVar, Union

try:
    from typing import TypedDict, NotRequired
except ImportError:
    from typing_extensions import TypedDict, NotRequired


# ==================== OpenAI 消息格式 ====================

class ToolFunctionDict(TypedDict):
    """工具函数"""
    name: str
    arguments: str


class ToolCallDict(TypedDict):
    """工具调用"""
    id: str
    type: Literal["function"]
    function: ToolFunctionDict


class MessageDict(TypedDict, total=False):
    """消息字典"""
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]]
    name: NotRequired[str]
    tool_calls: NotRequired[List[ToolCallDict]]
    tool_call_id: NotRequired[str]


class UsageDict(TypedDict, total=False):
    """Token 使用"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# ==================== Result 类型 ====================

T = TypeVar("T")
E = TypeVar("E", bound=Exception)


class Result(Generic[T]):
    """
    结果包装器
    
    示例:
        ```python
        def divide(a: int, b: int) -> Result[float]:
            if b == 0:
                return Result.err("Division by zero")
            return Result.ok(a / b)
        
        result = divide(10, 2)
        if result.is_ok:
            print(result.unwrap())
        ```
    """

    __slots__ = ("_value", "_error", "_is_ok")

    def __init__(
        self,
        value: Optional[T] = None,
        error: Optional[str] = None,
        is_ok: bool = True
    ):
        self._value = value
        self._error = error
        self._is_ok = is_ok

    @classmethod
    def ok(cls, value: T) -> "Result[T]":
        """创建成功结果"""
        return cls(value=value, is_ok=True)

    @classmethod
    def err(cls, error: str) -> "Result[T]":
        """创建错误结果"""
        return cls(error=error, is_ok=False)

    @property
    def is_ok(self) -> bool:
        return self._is_ok

    @property
    def is_err(self) -> bool:
        return not self._is_ok

    @property
    def error(self) -> Optional[str]:
        return self._error

    def unwrap(self) -> T:
        """获取值（错误时抛异常）"""
        if not self._is_ok:
            raise ValueError(f"Cannot unwrap error: {self._error}")
        return self._value  # type: ignore

    def unwrap_or(self, default: T) -> T:
        """获取值或默认值"""
        return self._value if self._is_ok else default  # type: ignore

    def __repr__(self) -> str:
        if self._is_ok:
            return f"Ok({self._value!r})"
        return f"Err({self._error!r})"


# ==================== 类型别名 ====================

MessageList = List[MessageDict]
ToolCallList = List[ToolCallDict]
Embedding = List[float]
EmbeddingBatch = List[Embedding]


__all__ = [
    "MessageDict",
    "ToolCallDict",
    "ToolFunctionDict",
    "UsageDict",
    "Result",
    "MessageList",
    "ToolCallList",
    "Embedding",
    "EmbeddingBatch",
]