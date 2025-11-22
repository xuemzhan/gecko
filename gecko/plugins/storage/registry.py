# gecko/plugins/storage/registry.py
"""
存储插件注册器

负责管理 URL Scheme 到存储后端类的映射。
采用装饰器模式进行注册。
"""
from __future__ import annotations

from typing import Callable, Dict, Type

from gecko.core.logging import get_logger
from gecko.plugins.storage.abc import AbstractStorage

logger = get_logger(__name__)

# 存储后端类注册表
# Key: URL scheme (e.g., "sqlite", "redis")
# Value: Storage Class
_STORAGE_REGISTRY: Dict[str, Type[AbstractStorage]] = {}


def register_storage(scheme: str) -> Callable[[Type[AbstractStorage]], Type[AbstractStorage]]:
    """
    装饰器：注册存储后端实现
    
    参数:
        scheme: URL 协议前缀 (如 'sqlite', 'redis')
    
    示例:
        @register_storage("redis")
        class RedisStorage(AbstractStorage):
            ...
    """
    def decorator(cls: Type[AbstractStorage]) -> Type[AbstractStorage]:
        if scheme in _STORAGE_REGISTRY:
            logger.warning(
                "Storage scheme already registered, overwriting",
                scheme=scheme,
                existing=_STORAGE_REGISTRY[scheme].__name__,
                new=cls.__name__
            )
        
        _STORAGE_REGISTRY[scheme] = cls
        logger.debug("Registered storage backend", scheme=scheme, cls=cls.__name__)
        return cls
    
    return decorator


def get_storage_class(scheme: str) -> Type[AbstractStorage] | None:
    """获取已注册的存储类"""
    return _STORAGE_REGISTRY.get(scheme)