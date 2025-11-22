# gecko/core/protocols/tool.py
"""工具相关协议"""
from __future__ import annotations
from typing import Any, Dict, Protocol, runtime_checkable
from gecko.core.protocols.base import get_missing_methods

@runtime_checkable
class ToolProtocol(Protocol):
    """工具协议"""
    name: str
    description: str
    parameters: Dict[str, Any]
    async def execute(self, arguments: Dict[str, Any]) -> str: ...

def validate_tool(tool: Any) -> None:
    """
    验证工具是否符合 ToolProtocol
    
    修复点：
    1. 补全 description 和 parameters 的类型/内容检查。
    2. 调整错误提示语序，以匹配单元测试的正则断言 (e.g. "non-empty 'name'").
    """
    # 1. 属性存在性检查
    for attr in ["name", "description", "parameters"]:
        if not hasattr(tool, attr):
            # 测试要求：ValueError(match="'name'") 或 ("Tool must have a 'name' attribute")
            raise ValueError(f"Tool must have a '{attr}' attribute")
    
    # 2. Name 内容检查
    # 测试要求正则: "non-empty 'name'"
    if not isinstance(tool.name, str) or not tool.name.strip():
        raise ValueError("Tool must have a non-empty 'name' attribute")
        
    # 3. Description 内容检查 (之前漏掉了)
    # 测试要求正则: "non-empty 'description'"
    if not isinstance(tool.description, str) or not tool.description.strip():
        raise ValueError("Tool must have a non-empty 'description' attribute")

    # 4. Parameters 类型检查 (之前漏掉了)
    # 测试要求正则: "'parameters'"
    if not isinstance(tool.parameters, dict):
        raise ValueError("Tool must have a 'parameters' dict attribute")

    # 5. 方法检查
    # 测试要求正则: "execute"
    if not hasattr(tool, "execute"):
        raise TypeError("Tool must have an 'execute' method")
    
    if not callable(getattr(tool, "execute")):
        raise TypeError("Tool 'execute' must be callable")