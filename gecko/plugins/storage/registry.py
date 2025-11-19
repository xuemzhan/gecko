# gecko/plugins/storage/registry.py
from __future__ import annotations
from typing import Dict, Callable

_STORAGE_FACTORIES: Dict[str, Callable] = {}

def register_storage(scheme: str):
    """
    装饰器：注册存储实现类
    - scheme: URL 前缀，如 "sqlite"
    - 返回工厂函数：接收 storage_url + overrides，返回实例
    用法：@register_storage("sqlite")
    class MyStorage: ...
    """
    def decorator(cls):
        if scheme in _STORAGE_FACTORIES:
            raise ValueError(f"方案 {scheme} 已注册")
        def factory(storage_url: str, **overrides):
            return cls(storage_url=storage_url, **overrides)
        _STORAGE_FACTORIES[scheme] = factory
        return cls
    return decorator

def get_storage_factory(scheme: str) -> Callable | None:
    """根据 scheme 获取注册的工厂函数"""
    return _STORAGE_FACTORIES.get(scheme)