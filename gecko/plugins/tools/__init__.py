# gecko/plugins/tools/__init__.py
"""
Gecko 工具插件系统

提供工具的定义、注册和发现机制。

核心组件：
- BaseTool: 所有工具的基类
- ToolResult: 标准化执行结果
- register_tool: 工具注册装饰器
- load_tool: 工具加载工厂
- ToolRegistry: 注册表管理类
"""
from gecko.plugins.tools.base import BaseTool, ToolResult
from gecko.plugins.tools.registry import ToolRegistry, load_tool, register_tool

__all__ = [
    "BaseTool",
    "ToolResult",
    "ToolRegistry",
    "register_tool",
    "load_tool",
]