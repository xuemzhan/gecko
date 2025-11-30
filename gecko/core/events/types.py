# gecko/core/events/types.py
"""
事件类型定义模块
定义了系统内流转的标准事件结构。
"""
from __future__ import annotations
import time
from typing import Any, Dict, Optional, Literal
from pydantic import BaseModel, Field

class BaseEvent(BaseModel):
    """
    通用事件基类
    用于 EventBus 的内部消息分发。
    """
    type: str
    timestamp: float = Field(default_factory=time.time)
    data: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    
    model_config = {"arbitrary_types_allowed": True}

class AgentStreamEvent(BaseModel):
    """
    Agent 流式输出的标准协议。
    
    设计目的：
    解决流式生成过程中 yield 类型不统一的问题。
    前端或调用方应根据 `type` 字段决定如何渲染或处理。
    
    字段说明：
    - token:       LLM 生成的文本片段（增量）
    - tool_input:  Agent 决定调用工具，包含工具名和参数
    - tool_output: 工具执行完毕，包含执行结果
    - error:       执行过程中发生的非致命错误或警告
    - result:      执行结束，包含最终的 AgentOutput 对象
    """
    type: Literal["token", "tool_input", "tool_output", "error", "result"] = Field(
        ..., description="事件类型"
    )
    content: str = Field(default="", description="文本内容（如 Token 片段、错误详情、工具结果摘要）")
    data: Dict[str, Any] = Field(default_factory=dict, description="结构化载荷（如完整的工具调用对象、AgentOutput对象）")

    model_config = {
        "arbitrary_types_allowed": True,
        "populate_by_name": True
    }