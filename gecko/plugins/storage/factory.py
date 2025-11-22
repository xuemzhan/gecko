# gecko/plugins/storage/factory.py
"""
存储工厂

负责解析 URL，自动加载对应的后端模块，实例化并初始化存储对象。
实现了懒加载机制、注册表刷新和回退扫描机制，确保在各种环境下都能正确加载后端。
"""
from __future__ import annotations

import importlib
import inspect
import sys
from typing import Any, Optional, Type

from gecko.core.exceptions import ConfigurationError
from gecko.core.logging import get_logger
from gecko.plugins.storage.abc import AbstractStorage
# [重要] 导入模块本身，防止闭包引用陈旧的 _STORAGE_REGISTRY 字典
import gecko.plugins.storage.registry as storage_registry
from gecko.plugins.storage.utils import parse_storage_url

logger = get_logger(__name__)

# 模块映射表：Scheme -> Module Path
# 用于懒加载，防止 Import Hell
_BACKEND_MODULES = {
    "sqlite": "gecko.plugins.storage.backends.sqlite",
    "redis": "gecko.plugins.storage.backends.redis",
    "chroma": "gecko.plugins.storage.backends.chroma",
    "lancedb": "gecko.plugins.storage.backends.lancedb",
    "postgres": "gecko.plugins.storage.backends.postgres",
    "qdrant": "gecko.plugins.storage.backends.qdrant",
    "milvus": "gecko.plugins.storage.backends.milvus",
}


async def create_storage(url: str, **kwargs: Any) -> AbstractStorage:
    """
    创建并初始化存储后端
    
    流程：
    1. 解析 URL 获取协议 (scheme)。
    2. 检查注册表是否已有对应类。
    3. 如果没有，尝试动态导入对应的模块。
    4. 如果导入后仍未注册（常见于测试 Mock 环境），手动扫描模块寻找类。
    5. 实例化并异步初始化。
    
    参数:
        url: 存储 URL (e.g., "sqlite:///data.db")
        **kwargs: 传递给存储后端的额外参数
    
    返回:
        已初始化的存储对象
        
    异常:
        ConfigurationError: 无法加载或初始化存储后端
    """
    try:
        scheme, _, _ = parse_storage_url(url)
        # 处理特殊变体 (如 postgres+pgvector -> postgres)
        clean_scheme = scheme.split("+")[0]
    except Exception as e:
        raise ConfigurationError(f"Invalid storage URL: {e}") from e

    # 1. 尝试从注册表获取
    cls: Optional[Type[AbstractStorage]] = storage_registry.get_storage_class(scheme)

    # 2. 如果未注册，尝试动态导入
    if not cls:
        module_path = _BACKEND_MODULES.get(clean_scheme)
        if not module_path:
            raise ConfigurationError(
                f"Unknown storage scheme: '{scheme}'. "
                f"Supported: {list(_BACKEND_MODULES.keys())}"
            )
        
        try:
            logger.debug("Lazy loading storage backend", module=module_path)
            
            # 如果模块已在 sys.modules 中（可能是之前的测试加载过），强制重载以触发注册副作用
            if module_path in sys.modules:
                module = importlib.reload(sys.modules[module_path])
            else:
                module = importlib.import_module(module_path)
                
        except ImportError as e:
            raise ConfigurationError(
                f"Failed to load storage backend for '{scheme}'.\n"
                f"Missing dependency? Try installing: pip install gecko-ai[{clean_scheme}]\n"
                f"Error: {e}"
            ) from e
        
        # 3. 再次尝试从注册表获取 (正常流程)
        cls = storage_registry.get_storage_class(scheme)
        
        # 4. 回退机制：如果装饰器注册失败（常见于测试环境 patch 导致注册表不一致），
        # 手动扫描模块寻找 AbstractStorage 的子类
        if not cls:
            logger.warning(
                f"Registry lookup failed for {scheme}, scanning module {module_path}..."
            )
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) 
                    and issubclass(obj, AbstractStorage) 
                    and obj is not AbstractStorage):
                    
                    cls = obj
                    # 手动补注册，避免下次扫描
                    storage_registry._STORAGE_REGISTRY[scheme] = cls
                    logger.info(f"Manually registered {name} for {scheme}")
                    break
            
        if not cls:
            # 调试信息
            current_keys = list(storage_registry._STORAGE_REGISTRY.keys())
            raise ConfigurationError(
                f"Module '{module_path}' imported but no storage registered for '{scheme}'.\n"
                f"Current registry keys: {current_keys}"
            )

    # 5. 实例化与初始化
    # 预先定义 instance 防止 unbound variable error
    instance: Optional[AbstractStorage] = None
    
    try:
        instance = cls(url, **kwargs)
        await instance.initialize()
        return instance
    except Exception as e:
        # 如果初始化失败，尝试清理资源
        if instance:
            await instance.shutdown()
        raise ConfigurationError(f"Failed to initialize {cls.__name__}: {e}") from e