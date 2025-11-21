# gecko/core/protocols.py (完整修正版)
"""
Gecko 核心协议定义

定义框架中各组件的标准接口，使用 Protocol 实现鸭子类型。

核心设计原则：
1. Protocol 仅定义接口契约，不包含实现
2. 默认行为通过工具函数或基类提供
3. 支持运行时类型检查

核心协议：
- ModelProtocol: LLM 模型核心接口
- StreamableModelProtocol: 支持流式的模型接口
- StorageProtocol: 存储后端接口
- ToolProtocol: 工具接口
- EmbedderProtocol: 嵌入模型接口
- RunnableProtocol: 可运行对象接口
- VectorStoreProtocol: 向量存储接口
"""
from __future__ import annotations

from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

from pydantic import BaseModel, Field


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
        """获取增量内容"""
        if self.choices:
            return self.choices[0].get("delta", {})
        return {}
    
    @property
    def content(self) -> Optional[str]:
        """获取文本内容"""
        return self.delta.get("content")


# ====================== 模型协议 ======================

@runtime_checkable
class ModelProtocol(Protocol):
    """
    LLM 模型核心协议
    
    定义所有模型必须实现的接口。
    
    设计原则:
        - 仅定义必需的方法
        - 不包含任何默认实现
        - 支持运行时类型检查
    
    示例:
        ```python
        class MyModel:
            async def acompletion(
                self, 
                messages: List[Dict[str, Any]], 
                **kwargs
            ) -> CompletionResponse:
                # 调用 API
                response = await api_call(messages)
                return CompletionResponse(**response)
        
        # 验证
        assert isinstance(MyModel(), ModelProtocol)  # ✅ True
        ```
    """
    
    async def acompletion(
        self, 
        messages: List[Dict[str, Any]], 
        **kwargs
    ) -> CompletionResponse:
        """
        异步补全接口（必需）
        
        参数:
            messages: 消息列表（OpenAI 格式）
            **kwargs: 模型参数（temperature, max_tokens 等）
        
        返回:
            CompletionResponse 标准响应
        """
        ...


@runtime_checkable
class StreamableModelProtocol(ModelProtocol, Protocol):
    """
    支持流式输出的模型协议
    
    继承自 ModelProtocol，额外要求实现 astream 方法。
    
    示例:
        ```python
        class StreamingModel:
            async def acompletion(self, messages, **kwargs):
                ...
            
            async def astream(self, messages, **kwargs):
                async for chunk in api_stream(messages):
                    yield StreamChunk(**chunk)
        
        model = StreamingModel()
        assert isinstance(model, StreamableModelProtocol)  # ✅ True
        ```
    """
    
    async def astream(
        self, 
        messages: List[Dict[str, Any]], 
        **kwargs
    ) -> AsyncIterator[StreamChunk]:
        """
        异步流式补全接口
        
        参数:
            messages: 消息列表
            **kwargs: 模型参数
        
        返回:
            StreamChunk 异步生成器
        """
        ...
        # Protocol 要求有 yield，但不能真的执行
        yield  # type: ignore


# ====================== 存储协议 ======================

@runtime_checkable
class StorageProtocol(Protocol):
    """
    存储后端协议
    
    定义键值存储的标准接口。
    """
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """获取数据"""
        ...
    
    async def set(
        self, 
        key: str, 
        value: Dict[str, Any], 
        ttl: Optional[int] = None
    ) -> None:
        """存储数据"""
        ...
    
    async def delete(self, key: str) -> bool:
        """删除数据"""
        ...


# ====================== 工具协议 ======================

@runtime_checkable
class ToolProtocol(Protocol):
    """
    工具协议
    
    定义 Agent 可使用的工具的标准接口。
    """
    
    name: str
    description: str
    parameters: Dict[str, Any]
    
    async def execute(self, arguments: Dict[str, Any]) -> str:
        """执行工具"""
        ...


# ====================== 嵌入模型协议 ======================

@runtime_checkable
class EmbedderProtocol(Protocol):
    """嵌入模型协议"""
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文本"""
        ...
    
    def get_dimension(self) -> int:
        """获取向量维度"""
        ...


# ====================== 可运行对象协议 ======================

@runtime_checkable
class RunnableProtocol(Protocol):
    """可运行对象协议"""
    
    async def run(self, input: Any) -> Any:
        """运行对象"""
        ...


@runtime_checkable
class StreamableRunnableProtocol(RunnableProtocol, Protocol):
    """支持流式输出的可运行对象"""
    
    async def stream(self, input: Any) -> AsyncIterator[str]:
        """流式运行"""
        ...
        yield  # type: ignore


# ====================== 向量存储协议 ======================

@runtime_checkable
class VectorStoreProtocol(Protocol):
    """向量数据库协议"""
    
    async def add(
        self,
        ids: List[str],
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """添加向量"""
        ...
    
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """检索相似向量"""
        ...
    
    async def delete(self, ids: List[str]) -> None:
        """删除向量"""
        ...


# ====================== 工具函数 ======================

def check_protocol(obj: Any, protocol: type) -> bool:
    """
    检查对象是否实现了指定协议
    
    参数:
        obj: 要检查的对象
        protocol: 协议类型
    
    返回:
        是否实现了协议
    """
    return isinstance(obj, protocol)


def supports_streaming(model: Any) -> bool:
    """
    检查模型是否支持流式输出
    
    参数:
        model: 模型对象
    
    返回:
        是否支持
    
    实现逻辑:
        检查是否实现了 StreamableModelProtocol
    """
    return isinstance(model, StreamableModelProtocol)


def supports_function_calling(model: Any) -> bool:
    """
    检查模型是否支持 Function Calling
    
    参数:
        model: 模型对象
    
    返回:
        是否支持
    
    实现逻辑:
        检查 _supports_function_calling 属性或 supports_function_calling 方法
    """
    # 优先检查属性
    if hasattr(model, "_supports_function_calling"):
        return model._supports_function_calling
    
    # 其次检查方法
    if hasattr(model, "supports_function_calling"):
        method = getattr(model, "supports_function_calling")
        if callable(method):
            return method()
    
    return False


def supports_vision(model: Any) -> bool:
    """
    检查模型是否支持视觉输入
    
    参数:
        model: 模型对象
    
    返回:
        是否支持
    """
    if hasattr(model, "_supports_vision"):
        return model._supports_vision
    
    if hasattr(model, "supports_vision"):
        method = getattr(model, "supports_vision")
        if callable(method):
            return method()
    
    return False


def get_model_name(model: Any) -> str:
    """
    获取模型名称
    
    参数:
        model: 模型对象
    
    返回:
        模型名称
    """
    # 优先使用 model_name 属性
    if hasattr(model, "model_name"):
        return model.model_name
    
    # 其次使用 name 属性
    if hasattr(model, "name"):
        return model.name
    
    # 最后使用类名
    return model.__class__.__name__


def get_missing_methods(obj: Any, protocol: type) -> List[str]:
    """
    获取对象未实现的协议方法
    
    参数:
        obj: 要检查的对象
        protocol: 协议类型
    
    返回:
        缺失的方法名列表
    """
    import inspect as insp
    
    missing = []
    
    # 获取协议的所有成员
    for name, value in insp.getmembers(protocol):
        # 跳过私有成员和特殊成员
        if name.startswith("_"):
            continue
        
        # 检查是否是方法或属性
        if insp.isfunction(value) or insp.ismethod(value):
            if not hasattr(obj, name):
                missing.append(name)
            elif not callable(getattr(obj, name)):
                missing.append(name)
        elif insp.isdatadescriptor(value):
            # 属性检查
            if not hasattr(obj, name):
                missing.append(name)
    
    return missing


def validate_model(model: Any) -> None:
    """
    验证模型是否符合 ModelProtocol
    
    参数:
        model: 模型对象
    
    异常:
        TypeError: 模型不符合协议
    """
    if not isinstance(model, ModelProtocol):
        missing = get_missing_methods(model, ModelProtocol)
        raise TypeError(
            f"Model does not implement ModelProtocol. "
            f"Missing: {', '.join(missing) if missing else 'unknown'}"
        )


def validate_storage(storage: Any) -> None:
    """验证存储后端是否符合 StorageProtocol"""
    if not isinstance(storage, StorageProtocol):
        missing = get_missing_methods(storage, StorageProtocol)
        raise TypeError(
            f"Storage does not implement StorageProtocol. "
            f"Missing: {', '.join(missing) if missing else 'unknown'}"
        )


def validate_tool(tool: Any) -> None:
    """
    验证工具是否符合 ToolProtocol
    
    验证步骤：
    1. name: 必需、非空、字符串、去除空格后非空
    2. description: 必需、非空、字符串、去除空格后非空
    3. parameters: 必需、字典类型
    4. execute: 必需、异步方法
    
    参数:
        tool: 工具对象
    
    异常:
        ValueError: 属性无效（缺失、为空、类型错误）
        TypeError: 方法未实现（缺少 execute）
    """
    # 1. 验证 name
    if not hasattr(tool, "name"):
        raise ValueError("Tool must have a 'name' attribute")
    
    if not isinstance(tool.name, str) or not tool.name.strip():
        raise ValueError("Tool must have a non-empty 'name' attribute")
    
    # 2. 验证 description
    if not hasattr(tool, "description"):
        raise ValueError("Tool must have a 'description' attribute")
    
    if not isinstance(tool.description, str) or not tool.description.strip():
        raise ValueError("Tool must have a non-empty 'description' attribute")
    
    # 3. 验证 parameters
    if not hasattr(tool, "parameters"):
        raise ValueError("Tool must have a 'parameters' attribute")
    
    if not isinstance(tool.parameters, dict):
        raise ValueError("Tool must have a 'parameters' dict attribute")
    
    # 4. 验证 execute 方法
    if not hasattr(tool, "execute"):
        raise TypeError("Tool must have an 'execute' method")
    
    if not callable(getattr(tool, "execute")):
        raise TypeError("Tool 'execute' must be callable")
    
    # 5. 完整协议检查（作为最后的保障）
    if not isinstance(tool, ToolProtocol):
        missing = get_missing_methods(tool, ToolProtocol)
        missing_methods = [m for m in missing if m not in ["name", "description", "parameters"]]
        
        if missing_methods:
            raise TypeError(
                f"Tool does not implement ToolProtocol. "
                f"Missing methods: {', '.join(missing_methods)}"
            )


# ====================== 类型别名 ======================

ModelResponse = CompletionResponse
MessageDict = Dict[str, Any]
MessageList = List[MessageDict]
ToolCall = Dict[str, Any]
ToolCallList = List[ToolCall]
StorageValue = Dict[str, Any]
Vector = List[float]
VectorList = List[Vector]


# ====================== 导出 ======================

__all__ = [
    # 模型相关
    "ModelProtocol",
    "StreamableModelProtocol",
    "CompletionResponse",
    "CompletionChoice",
    "CompletionUsage",
    "StreamChunk",
    # 存储相关
    "StorageProtocol",
    # 工具相关
    "ToolProtocol",
    # 嵌入相关
    "EmbedderProtocol",
    # 可运行对象
    "RunnableProtocol",
    "StreamableRunnableProtocol",
    # 向量存储
    "VectorStoreProtocol",
    # 工具函数
    "check_protocol",
    "supports_streaming",
    "supports_function_calling",
    "supports_vision",
    "get_model_name",
    "get_missing_methods",
    "validate_model",
    "validate_storage",
    "validate_tool",
    # 类型别名
    "ModelResponse",
    "MessageDict",
    "MessageList",
    "ToolCall",
    "ToolCallList",
    "StorageValue",
    "Vector",
    "VectorList",
]