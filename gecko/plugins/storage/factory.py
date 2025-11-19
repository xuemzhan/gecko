# gecko/plugins/storage/factory.py
from __future__ import annotations
from typing import Dict, Any, Callable
from importlib.metadata import entry_points

# 本地注册表（快速开发插件）
_LOCAL_FACTORIES: Dict[str, Callable] = {}

def _register_local(scheme: str, cls: Any):
    """内部注册函数，供 __init__.py 使用"""
    if scheme in _LOCAL_FACTORIES:
        raise ValueError(f"本地存储 {scheme} 已注册")
    def factory(storage_url: str, **overrides):
        return cls(storage_url=storage_url, **overrides)
    _LOCAL_FACTORIES[scheme] = factory

def get_storage_by_url(storage_url: str, required: str = "any", **overrides) -> Any:
    """
    终极工厂：本地优先 + entry_points 兜底
    """
    scheme = storage_url.split("://")[0].split("+")[0]
    
    # 1. 优先本地注册（快速开发）
    factory = _LOCAL_FACTORIES.get(scheme)
    if factory:
        return factory(storage_url, **overrides)
    
    # 2. 其次 entry_points（生产插件）
    for ep in entry_points(group="gecko.storage"):
        if ep.name == scheme:
            factory_cls = ep.load()
            return factory_cls(storage_url, **overrides)
    
    raise ValueError(f"未找到存储实现: {scheme}\n"
                     f"  - 快速开发支持: sqlite, lancedb\n"
                     f"  - 生产插件需 pip install gecko-postgres / gecko-milvus 等")