"""
协议模块入口
重新导出所有子模块内容，确保外部引用 gecko.core.protocols.ModelProtocol 依然有效。
"""
from gecko.core.protocols.base import check_protocol, get_missing_methods
from gecko.core.protocols.model import (
    ModelProtocol, StreamableModelProtocol, 
    CompletionResponse, CompletionChoice, CompletionUsage, StreamChunk,
    supports_streaming, supports_function_calling, supports_vision,
    get_model_name, validate_model
)
from gecko.core.protocols.storage import StorageProtocol, validate_storage
from gecko.core.protocols.tool import ToolProtocol, validate_tool
from gecko.core.protocols.embedder import EmbedderProtocol
from gecko.core.protocols.runnable import RunnableProtocol, StreamableRunnableProtocol
from gecko.core.protocols.vector import VectorStoreProtocol

# 类型别名 (保持兼容)
from typing import Dict, List, Any
ModelResponse = CompletionResponse
MessageDict = Dict[str, Any]
MessageList = List[MessageDict]
ToolCall = Dict[str, Any]
ToolCallList = List[ToolCall]
StorageValue = Dict[str, Any]
Vector = List[float]
VectorList = List[Vector]

__all__ = [
    "ModelProtocol", "StreamableModelProtocol",
    "CompletionResponse", "CompletionChoice", "CompletionUsage", "StreamChunk",
    "StorageProtocol", "ToolProtocol", "EmbedderProtocol",
    "RunnableProtocol", "StreamableRunnableProtocol", "VectorStoreProtocol",
    "check_protocol", "validate_model", "validate_storage", "validate_tool",
    "supports_streaming", "supports_function_calling", "supports_vision",
    "get_model_name", "get_missing_methods",
    "ModelResponse", "MessageDict", "ToolCall", "Vector"
]