# agno/models/utils.py
from typing import Optional, Union

# 导入核心基类和新的注册表
from .base import Model
from .registry import PROVIDER_REGISTRY

def _get_model_class(model_id: str, model_provider: str) -> Model:
    """
    根据供应商名称从注册表中获取模型类并实例化。
    
    这个函数是重构的核心。它不再需要知道所有具体的模型实现，
    而是通过查询 PROVIDER_REGISTRY 来动态加载，从而实现了完全的解耦。
    这使得添加新的模型供应商变得非常简单和安全。

    Args:
        model_id (str): 要传递给模型实例的模型ID。
        model_provider (str): 模型供应商的名称，用作在注册表中的查找键。

    Returns:
        Model: 相应供应商的模型类的实例。

    Raises:
        ValueError: 如果在注册表中找不到指定的供应商。
    """
    # 从注册表中查找模型类
    model_class = PROVIDER_REGISTRY.get(model_provider.lower().strip())
    
    # 如果未找到，则抛出错误
    if model_class is None:
        raise ValueError(f"不支持的模型供应商 '{model_provider}'。")
        
    # 实例化并返回模型对象
    return model_class(id=model_id)


def _parse_model_string(model_string: str) -> Model:
    """
    解析格式为 '<provider>:<model_id>' 的模型字符串。
    (此函数逻辑保持不变)
    """
    if not model_string or not isinstance(model_string, str):
        raise ValueError(f"模型字符串必须是非空字符串，得到: {model_string}")

    if ":" not in model_string:
        raise ValueError(
            f"无效的模型字符串格式: '{model_string}'。模型字符串应为 '<provider>:<model_id>' 格式，例如 'openai:gpt-4o'"
        )

    parts = model_string.split(":", 1)
    if len(parts) != 2:
        raise ValueError(
            f"无效的模型字符串格式: '{model_string}'。模型字符串应为 '<provider>:<model_id>' 格式，例如 'openai:gpt-4o'"
        )

    model_provider, model_id = parts
    model_provider = model_provider.strip().lower()
    model_id = model_id.strip()

    if not model_provider or not model_id:
        raise ValueError(
            f"无效的模型字符串格式: '{model_string}'。模型字符串应为 '<provider>:<model_id>' 格式，例如 'openai:gpt-4o'"
        )

    return _get_model_class(model_id, model_provider)


def get_model(model: Union[Model, str, None]) -> Optional[Model]:
    """
    获取模型实例的入口函数。
    (此函数逻辑保持不变)
    """
    if model is None:
        return None
    elif isinstance(model, Model):
        return model
    elif isinstance(model, str):
        return _parse_model_string(model)
    else:
        raise ValueError("模型必须是 Model 实例、字符串或 None")