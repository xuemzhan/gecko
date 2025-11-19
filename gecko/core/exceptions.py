# gecko/core/exceptions.py
from __future__ import annotations

class GeckoError(Exception):
    """
    Gecko 框架统一异常基类
    - 所有 Gecko 相关异常继承自此
    - 便于全局捕获：except GeckoError as e
    - 企业级日志/监控推荐捕获此基类
    """
    pass

class AgentError(GeckoError):
    """Agent 执行相关异常（如模型调用失败、工具错误）"""
    pass

class WorkflowError(GeckoError):
    """Workflow 执行相关异常（如 DAG 循环、节点失败） - 本次新增"""
    pass

class StorageError(GeckoError):
    """存储插件相关异常（如 Session/Vector 读写失败） - 预留"""
    pass

class ToolError(GeckoError):
    """工具执行相关异常（如参数解析失败） - 预留"""
    pass

# 可选：添加通用工厂（方便日志）
def raise_workflow_error(message: str) -> None:
    raise WorkflowError(f"[Workflow] {message}")

def raise_agent_error(message: str) -> None:
    raise AgentError(f"[Agent] {message}")

class PluginNotFoundError(GeckoError):
    """插件未注册"""
    def __init__(self, plugin_type: str, name: str):
        super().__init__(f"{plugin_type.capitalize()} '{name}' not found in registry")