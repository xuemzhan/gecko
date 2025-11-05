# agno/models/registry.py
from typing import Dict, Type
from .base import Model

# 全局的模型供应商注册表
# 键 (str): 供应商的唯一标识符，例如 'openai', 'anthropic'
# 值 (Type[Model]): 对应的模型类
PROVIDER_REGISTRY: Dict[str, Type[Model]] = {}

def register_model(name: str):
    """
    一个装饰器工厂，用于将模型类注册到全局注册表中。

    通过这种方式，我们遵循了开闭原则：当添加一个新的模型供应商时，
    只需要在模型类定义上添加此装饰器即可，无需修改任何工厂函数代码。

    Args:
        name (str): 要注册的供应商的名称。此名称将用作在注册表中的键。

    Returns:
        Callable: 一个接收类并将其注册的装饰器。
    """
    def decorator(cls: Type[Model]):
        """
        实际的装饰器，负责执行注册操作。
        """
        # 检查是否已存在同名供应商，防止命名冲突
        if name in PROVIDER_REGISTRY:
            raise ValueError(f"模型供应商 '{name}' 已经被注册。")
        
        # 将类注册到字典中
        PROVIDER_REGISTRY[name] = cls
        
        # 返回原始类，以便可以正常使用
        return cls
    return decorator