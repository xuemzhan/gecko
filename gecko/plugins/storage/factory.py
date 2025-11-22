# gecko/plugins/storage/factory.py
"""
存储工厂

负责解析 URL，自动加载对应的后端模块，实例化并初始化存储对象。

更新日志：
- [Arch] 引入 importlib.metadata (entry_points) 支持第三方插件自动发现。
- [Refactor] 优化加载顺序：Entry Points -> Built-in -> Registry Check。
"""
from __future__ import annotations

import importlib
import inspect
import sys
# 引入 metadata 用于发现插件
from importlib.metadata import entry_points
from typing import Any, Optional, Type

from gecko.core.exceptions import ConfigurationError
from gecko.core.logging import get_logger
from gecko.plugins.storage.abc import AbstractStorage
# [重要] 导入模块本身，防止闭包引用陈旧的 _STORAGE_REGISTRY 字典
import gecko.plugins.storage.registry as storage_registry
from gecko.plugins.storage.utils import parse_storage_url

logger = get_logger(__name__)

# 模块映射表：Scheme -> Module Path (内置后端)
_BACKEND_MODULES = {
    "sqlite": "gecko.plugins.storage.backends.sqlite",
    "redis": "gecko.plugins.storage.backends.redis",
    "chroma": "gecko.plugins.storage.backends.chroma",
    "lancedb": "gecko.plugins.storage.backends.lancedb",
    "postgres": "gecko.plugins.storage.backends.postgres",
    "qdrant": "gecko.plugins.storage.backends.qdrant",
    "milvus": "gecko.plugins.storage.backends.milvus",
}

# 插件组名称
PLUGIN_GROUP = "gecko.storage.backends"


def _load_from_entry_point(scheme: str) -> bool:
    """
    尝试从 Entry Points 加载插件
    
    返回: 是否成功加载
    """
    # 兼容 Python 3.10+ 和旧版本的 entry_points API
    try:
        # Python 3.10+
        eps = entry_points(group=PLUGIN_GROUP)
    except TypeError:
        # Python 3.8/3.9: entry_points() 返回字典
        eps = entry_points().get(PLUGIN_GROUP, []) # type: ignore

    # 查找匹配 scheme 的插件
    # 假设 entry point name 即为 scheme (例如: "mongo = my_package.mongo:MongoStorage")
    target_ep = next((ep for ep in eps if ep.name == scheme), None)

    if target_ep:
        try:
            logger.info(f"Loading storage plugin via entry point: {scheme}")
            # load() 会导入模块并返回对象 (类或函数)
            # 导入模块时，其内部的 @register_storage 装饰器应该会被触发
            target_ep.load()
            return True
        except Exception as e:
            logger.error(f"Failed to load plugin for '{scheme}': {e}")
            # 这里不抛出异常，允许回退到内置模块
            return False
    
    return False


async def create_storage(url: str, **kwargs: Any) -> AbstractStorage:
    """
    创建并初始化存储后端
    
    流程：
    1. 解析 URL 获取协议 (scheme)。
    2. 检查注册表是否已有对应类。
    3. 如果没有，按优先级尝试动态加载：
       a. Entry Points (第三方插件)
       b. Built-in Modules (内置支持)
    4. 再次检查注册表。
    5. 回退机制：手动扫描模块。
    6. 实例化并异步初始化。
    
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

    # 2. 如果未注册，尝试动态加载
    if not cls:
        module: Any = None
        loaded = False

        # 2a. 优先尝试 Entry Points (允许插件覆盖内置)
        if _load_from_entry_point(clean_scheme):
            loaded = True
        
        # 2b. 如果插件未处理，尝试内置模块
        if not loaded:
            module_path = _BACKEND_MODULES.get(clean_scheme)
            if not module_path:
                # 如果既没有插件也没有内置支持
                raise ConfigurationError(
                    f"Unknown storage scheme: '{scheme}'. "
                    f"Supported built-ins: {list(_BACKEND_MODULES.keys())}, "
                    f"Plugins: Check '{PLUGIN_GROUP}' entry points."
                )
            
            try:
                logger.debug("Lazy loading built-in backend", module=module_path)
                
                # [Fix] 增加对 None 的检查。
                # 如果测试 Mock 将模块置为 None，或者模块加载失败残留为 None，应视为未加载，
                # 从而进入 import_module 尝试加载（并触发预期的 ImportError）。
                if module_path in sys.modules and sys.modules[module_path] is not None:
                    module = importlib.reload(sys.modules[module_path])
                else:
                    module = importlib.import_module(module_path)
                
                loaded = True
            except ImportError as e:
                raise ConfigurationError(
                    f"Failed to load built-in backend for '{scheme}'.\n"
                    f"Missing dependency? Try installing: pip install gecko-ai[{clean_scheme}]\n"
                    f"Error: {e}"
                ) from e
        
        # 3. 再次尝试从注册表获取
        cls = storage_registry.get_storage_class(scheme)
        
        # 4. 回退机制：手动扫描模块 (针对内置模块加载后装饰器失效的罕见情况)
        if not cls and module:
            logger.warning(
                f"Registry lookup failed for {scheme}, scanning module..."
            )
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) 
                    and issubclass(obj, AbstractStorage) 
                    and obj is not AbstractStorage):
                    
                    cls = obj
                    # 手动补注册
                    storage_registry._STORAGE_REGISTRY[scheme] = cls
                    logger.info(f"Manually registered {name} for {scheme}")
                    break
            
        if not cls:
            current_keys = list(storage_registry._STORAGE_REGISTRY.keys())
            raise ConfigurationError(
                f"Backend loaded but no storage class registered for '{scheme}'.\n"
                f"Current registry keys: {current_keys}"
            )

    # 5. 实例化与初始化
    instance: Optional[AbstractStorage] = None
    
    try:
        instance = cls(url, **kwargs)
        await instance.initialize()
        return instance
    except Exception as e:
        if instance:
            await instance.shutdown()
        raise ConfigurationError(f"Failed to initialize {cls.__name__}: {e}") from e