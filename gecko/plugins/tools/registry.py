# gecko/plugins/tools/registry.py
"""
工具注册表

提供工具的自动发现、注册与工厂化创建能力。
"""
from __future__ import annotations

from typing import Dict, Type, List, Any, Optional

from gecko.core.logging import get_logger
from gecko.plugins.tools.base import BaseTool

logger = get_logger(__name__)


class ToolRegistry:
    """全局工具注册中心"""
    
    _registry: Dict[str, Type[BaseTool]] = {}

    @classmethod
    def register(cls, name: str):
        """
        装饰器：注册工具类
        
        示例:
            @register_tool("my_tool")
            class MyTool(BaseTool): ...
        """
        def decorator(tool_cls: Type[BaseTool]):
            if not issubclass(tool_cls, BaseTool):
                raise TypeError(f"Registered class {tool_cls.__name__} must inherit from BaseTool")
            
            if name in cls._registry:
                logger.warning(f"Tool '{name}' is being overwritten in registry.")
            
            cls._registry[name] = tool_cls
            logger.debug(f"Tool registered: {name} -> {tool_cls.__name__}")
            return tool_cls
        return decorator

    @classmethod
    def load_tool(cls, name: str, **kwargs: Any) -> BaseTool:
        """
        工厂方法：根据名称加载并实例化工具
        
        参数:
            name: 工具名称
            **kwargs: 传递给工具构造函数的参数
            
        异常:
            ValueError: 工具未找到
        """
        if name not in cls._registry:
            raise ValueError(f"Tool '{name}' not found in registry. Available: {list(cls._registry.keys())}")
        
        tool_cls = cls._registry[name]
        try:
            # 实例化工具
            # 注意：BaseTool 是 Pydantic 模型，kwargs 会被 validate
            return tool_cls(**kwargs)
        except Exception as e:
            raise ValueError(f"Failed to instantiate tool '{name}': {e}") from e

    @classmethod
    def list_tools(cls) -> List[str]:
        """列出所有已注册工具"""
        return list(cls._registry.keys())


# 便捷导出
register_tool = ToolRegistry.register
load_tool = ToolRegistry.load_tool