"""
LLM 模型相关协议与数据结构
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Protocol, AsyncIterator, runtime_checkable
from pydantic import BaseModel, Field
from gecko.core.protocols.base import check_protocol, get_missing_methods

# ====================== 模型响应格式 ======================

class CompletionChoice(BaseModel):
    """单个补全选择"""
    index: int = 0
    message: Dict[str, Any]
    finish_reason: Optional[str] = None
    logprobs: Optional[Dict[str, Any]] = None

class CompletionUsage(BaseModel):
    """Token 使用统计"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class CompletionResponse(BaseModel):
    """标准的模型补全响应格式"""
    id: str = Field(default="", description="响应 ID")
    object: str = Field(default="chat.completion", description="对象类型")
    created: int = Field(default=0, description="创建时间戳")
    model: str = Field(default="", description="模型名称")
    choices: List[CompletionChoice] = Field(default_factory=list)
    usage: Optional[CompletionUsage] = Field(default=None)
    system_fingerprint: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class StreamChunk(BaseModel):
    """流式响应的单个数据块"""
    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: List[Dict[str, Any]] = Field(default_factory=list)
    
    @property
    def delta(self) -> Dict[str, Any]:
        if self.choices:
            return self.choices[0].get("delta", {})
        return {}
    
    @property
    def content(self) -> Optional[str]:
        return self.delta.get("content")

# ====================== 模型协议 ======================

@runtime_checkable
class ModelProtocol(Protocol):
    """LLM 模型核心协议"""
    async def acompletion(self, messages: List[Dict[str, Any]], **kwargs) -> CompletionResponse:
        ...

    # [新增] 同步 Token 计数接口
    # 允许上层模块（如 Memory）在不阻塞 Event Loop 的前提下获取 Token 数
    def count_tokens(self, text_or_messages: str | List[Dict[str, Any]]) -> int:
        """
        计算输入内容的 Token 数量。
        应尽量使用本地 Tokenizer (如 tiktoken) 以保证性能。
        """
        ...

@runtime_checkable
class StreamableModelProtocol(ModelProtocol, Protocol):
    """支持流式输出的模型协议"""
    async def astream(self, messages: List[Dict[str, Any]], **kwargs) -> AsyncIterator[StreamChunk]:
        ...
        yield # type: ignore

# ====================== 工具函数 ======================

def supports_streaming(model: Any) -> bool:
    return isinstance(model, StreamableModelProtocol)

def supports_function_calling(model: Any) -> bool:
    if hasattr(model, "_supports_function_calling"):
        return model._supports_function_calling
    if hasattr(model, "supports_function_calling"):
        method = getattr(model, "supports_function_calling")
        if callable(method):
            return method() # type: ignore
    return False

def supports_vision(model: Any) -> bool:
    if hasattr(model, "_supports_vision"):
        return model._supports_vision
    if hasattr(model, "supports_vision"):
        method = getattr(model, "supports_vision")
        if callable(method):
            return method() # type: ignore
    return False

def get_model_name(model: Any) -> str:
    if hasattr(model, "model_name"): return model.model_name
    if hasattr(model, "name"): return model.name
    return model.__class__.__name__

def validate_model(model: Any) -> None:
    """
    验证模型是否满足 ModelProtocol 所需的方法/属性（鸭子类型检查）

    不再依赖 isinstance(model, ModelProtocol)，而是基于 get_missing_methods，
    这样自定义模型只要实现必要方法即可通过验证。
    """
    missing = get_missing_methods(model, ModelProtocol)
    if missing:
        raise TypeError(
            "Model does not implement ModelProtocol. "
            f"Missing methods: {', '.join(missing)}"
        )