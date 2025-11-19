# gecko/plugins/storage/factory.py
from __future__ import annotations
import importlib
import logging
from urllib.parse import urlparse
from gecko.plugins.storage.registry import _STORAGE_FACTORIES

logger = logging.getLogger(__name__)

def get_storage_by_url(storage_url: str, required: str = "any", **overrides) -> Any:
    """
    根据 URL 自动初始化对应的存储后端
    """
    if "://" in storage_url:
        scheme = storage_url.split("://")[0]
    else:
        raise ValueError(f"无效的存储 URL: {storage_url}")

    # 1. 查找已注册的工厂
    factory = _STORAGE_FACTORIES.get(scheme)

    # 2. [优化] 如果未找到，尝试动态导入同名模块 (gecko.plugins.storage.{scheme})
    if not factory:
        try:
            module_name = f"gecko.plugins.storage.{scheme}"
            logger.debug(f"尝试动态加载存储插件: {module_name}")
            importlib.import_module(module_name)
            # 重新获取
            factory = _STORAGE_FACTORIES.get(scheme)
        except ImportError as e:
            # 如果是因为缺少依赖包（如 lancedb），抛出更明确的错误
            if scheme in str(e) and "No module named" not in str(e):
                raise ImportError(f"加载存储插件 {scheme} 失败，请安装对应依赖: {e}")
            pass
        except Exception as e:
            logger.warning(f"动态加载存储插件 {scheme} 失败: {e}")

    if not factory:
        # 尝试加载外部插件（entry_points）
        raise ValueError(f"未找到存储实现: {scheme}。\n"
                         f"请确保：\n"
                         f"1. 已安装对应依赖 (如 `rye add lancedb`)\n"
                         f"2. URL 协议头正确 (如 lancedb://)")

    instance = factory(storage_url, **overrides)
    
    # 简单的接口检查
    if required == "session" and not hasattr(instance, "get"):
         raise TypeError(f"存储 {scheme} 不支持 Session 接口")
    if required == "vector" and not hasattr(instance, "search"):
         raise TypeError(f"存储 {scheme} 不支持 Vector 接口")

    return instance