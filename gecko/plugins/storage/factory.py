# gecko/plugins/storage/factory.py
"""
存储工厂

负责解析 URL，自动加载对应的后端模块，实例化并初始化存储对象。
实现了懒加载机制，避免导入未使用的依赖。
"""
from __future__ import annotations

import importlib
from typing import Any

from gecko.core.exceptions import ConfigurationError
from gecko.core.logging import get_logger
from gecko.plugins.storage.abc import AbstractStorage
from gecko.plugins.storage.registry import get_storage_class
from gecko.plugins.storage.utils import parse_storage_url

logger = get_logger(__name__)

# 模块映射表：Scheme -> Module Path
# 用于懒加载，防止 import hell
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
    
    步骤:
    1. 解析 URL 获取 scheme
    2. 动态导入对应的后端模块
    3. 从注册表获取类
    4. 实例化并调用 initialize()
    
    参数:
        url: 存储 URL (e.g., "sqlite:///data.db")
        **kwargs: 传递给存储后端的额外参数
    
    返回:
        已初始化的存储对象
        
    异常:
        ConfigurationError: 无法加载存储后端
    """
    try:
        scheme, _, _ = parse_storage_url(url)
        # 处理特殊变体 (如 postgres+pgvector -> postgres)
        clean_scheme = scheme.split("+")[0]
    except Exception as e:
        raise ConfigurationError(f"Invalid storage URL: {e}") from e

    # 1. 检查注册表
    cls = get_storage_class(scheme)

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
            importlib.import_module(module_path)
        except ImportError as e:
            raise ConfigurationError(
                f"Failed to load storage backend for '{scheme}'.\n"
                f"Missing dependency? Try installing: pip install gecko-ai[{clean_scheme}]\n"
                f"Error: {e}"
            ) from e
        
        # 再次尝试获取
        cls = get_storage_class(scheme)
        if not cls:
            raise ConfigurationError(
                f"Module '{module_path}' imported but no storage registered for '{scheme}'"
            )

    # 3. 实例化
    try:
        instance = cls(url, **kwargs)
    except Exception as e:
        raise ConfigurationError(f"Failed to instantiate {cls.__name__}: {e}") from e

    # 4. 异步初始化
    try:
        await instance.initialize()
    except Exception as e:
        await instance.shutdown() # 确保清理
        raise ConfigurationError(f"Failed to initialize {cls.__name__}: {e}") from e

    return instance