# gecko/plugins/tools/base.py
from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable
from pydantic import BaseModel, Field

@runtime_checkable
class ToolProtocol(Protocol):
    """所有工具必须实现的协议（duck typing）"""
    name: str
    description: str
    parameters: Dict[str, Any]

    async def execute(self, arguments: Dict[str, Any]) -> str: ...

class ToolResponse(BaseModel):
    """工具执行返回结构"""
    tool_name: str
    content: str
    is_error: bool = False

class BaseTool(BaseModel):
    """
    所有工具的基类，继承自 Pydantic BaseModel
    关键：name/description/parameters 必须在子类中显式带类型注解覆盖
    """
    name: str = Field(..., description="工具唯一名称（子类必须覆盖）")
    description: str = Field(..., description="工具功能描述（子类必须覆盖）")
    parameters: Dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
            "required": []
        },
        description="OpenAI 风格的参数 schema"
    )

    async def execute(self, arguments: Dict[str, Any]) -> str:
        """子类必须实现此方法"""
        raise NotImplementedError("子类必须实现 execute 方法")