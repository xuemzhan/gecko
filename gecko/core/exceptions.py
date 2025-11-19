# gecko/core/exceptions.py
from __future__ import annotations

class GeckoError(Exception):
    """Gecko 框架所有异常的基类"""
    pass

class AgentError(GeckoError):
    """Agent 执行过程中发生的错误"""
    pass

class ModelError(GeckoError):
    """模型调用相关错误"""
    pass

class ToolError(GeckoError):
    """工具执行错误"""
    pass

class PluginNotFoundError(GeckoError):
    """插件未注册"""
    def __init__(self, plugin_type: str, name: str):
        super().__init__(f"{plugin_type.capitalize()} '{name}' not found in registry")