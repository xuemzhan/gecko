# gecko/plugins/tools/registry.py
from __future__ import annotations

from typing import Dict, List
from gecko.plugins.tools.base import BaseTool, ToolProtocol

class ToolRegistry:
    """全局工具注册表（单例）"""
    _tools: Dict[str, ToolProtocol] = {}

    @classmethod
    def register(cls, tool: ToolProtocol):
        if tool.name in cls._tools:
            raise ValueError(f"工具 '{tool.name}' 已注册")
        cls._tools[tool.name] = tool

    @classmethod
    def get(cls, name: str) -> ToolProtocol | None:
        return cls._tools.get(name)

    @classmethod
    def list_all(cls) -> List[ToolProtocol]:
        return list(cls._tools.values())

def tool(cls):
    """
    装饰器：自动注册工具实例
    用法：@tool 放在 BaseTool 子类上
    """
    if not issubclass(cls, BaseTool):
        raise TypeError("@tool 只能用于 BaseTool 子类")
    
    instance = cls()
    ToolRegistry.register(instance)
    return cls