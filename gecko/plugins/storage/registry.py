# gecko/plugins/storage/registry.py
from __future__ import annotations
from typing import Dict, Callable, Type, Any

# 存储后端工厂注册表
_STORAGE_FACTORIES: Dict[str, Callable] = {}

def register_storage(scheme: str):
    """
    装饰器：注册存储后端实现
    :param scheme: URL 协议前缀，如 'sqlite', 'redis', 'postgres'
    """
    def decorator(cls):
        if scheme in _STORAGE_FACTORIES:
            raise ValueError(f"存储方案 '{scheme}' 已注册")
        
        # 包装为工厂函数
        def factory(storage_url: str, **overrides):
            return cls(storage_url=storage_url, **overrides)
            
        _STORAGE_FACTORIES[scheme] = factory
        return cls
    return decorator

def get_storage_factory(scheme: str) -> Callable | None:
    """获取指定协议的工厂函数"""
    return _STORAGE_FACTORIES.get(scheme)