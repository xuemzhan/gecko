# gecko/plugins/storage/abc.py
"""
存储抽象基类

定义所有存储后端必须实现的生命周期方法。
确保所有插件都有统一的初始化和关闭流程。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class AbstractStorage(ABC):
    """
    存储后端抽象基类
    
    所有具体的存储实现（SQLite, Redis, Chroma 等）都必须继承此类，
    并实现异步的生命周期管理方法。
    """
    
    def __init__(self, url: str, **kwargs: Any):
        """
        初始化存储后端配置
        
        参数:
            url: 连接字符串 (例如: sqlite:///./data.db)
            **kwargs: 额外的配置参数
        """
        self.url = url
        self.config = kwargs
        self._is_initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """
        异步初始化
        
        用于建立数据库连接、创建表结构、检查索引等耗时 IO 操作。
        必须确保此方法是幂等的（多次调用不会出错）。
        """
        pass

    @abstractmethod
    async def shutdown(self) -> None:
        """
        异步关闭
        
        用于释放连接池、关闭文件句柄、清理临时资源。
        """
        pass
    
    @property
    def is_initialized(self) -> bool:
        """检查是否已初始化"""
        return self._is_initialized

    async def __aenter__(self):
        """支持上下文管理器"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时自动关闭"""
        await self.shutdown()