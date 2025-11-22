# gecko/plugins/models/registry.py
from __future__ import annotations

from typing import Callable, Dict, Optional, Type, TypeVar

from gecko.core.logging import get_logger
from gecko.plugins.models.base import BaseChatModel

logger = get_logger(__name__)

_DRIVER_REGISTRY: Dict[str, Type[BaseChatModel]] = {}

# [修复] 定义一个泛型变量，限定范围是 BaseChatModel 的子类
T = TypeVar("T", bound=Type[BaseChatModel])

def register_driver(name: str) -> Callable[[T], T]:
    """
    装饰器：注册模型驱动
    
    使用泛型 T 确保类型推断能够透传：
    输入是具体的 Driver 类，返回的也是具体的 Driver 类（保留了具体实现信息）。
    """
    def decorator(cls: T) -> T:
        if name in _DRIVER_REGISTRY:
            logger.warning(f"Driver '{name}' already registered, overwriting with {cls.__name__}")
        _DRIVER_REGISTRY[name] = cls
        logger.debug(f"Registered model driver: {name}")
        return cls
    return decorator


def get_driver_class(name: str) -> Optional[Type[BaseChatModel]]:
    """获取驱动类"""
    return _DRIVER_REGISTRY.get(name)