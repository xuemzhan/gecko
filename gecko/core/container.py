# gecko/core/container.py
"""
轻量级依赖注入容器

提供服务的注册、解析和生命周期管理。
"""
from __future__ import annotations

import asyncio
import inspect
from typing import (
    Any, Callable, Dict, Generic, Optional, Type, TypeVar, Union, 
    get_type_hints, overload
)
from enum import Enum
from contextlib import asynccontextmanager

from gecko.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class Lifetime(str, Enum):
    """服务生命周期"""
    TRANSIENT = "transient"    # 每次解析创建新实例
    SINGLETON = "singleton"    # 全局单例
    SCOPED = "scoped"          # 作用域内单例


class ServiceDescriptor(Generic[T]):
    """服务描述符"""
    
    def __init__(
        self,
        service_type: Type[T],
        implementation: Union[Type[T], Callable[..., T], T],
        lifetime: Lifetime = Lifetime.TRANSIENT,
    ):
        self.service_type = service_type
        self.implementation = implementation
        self.lifetime = lifetime
        
        # 单例实例缓存
        self._instance: Optional[T] = None
    
    def is_instance(self) -> bool:
        """是否是直接注册的实例"""
        return not (
            inspect.isclass(self.implementation) or 
            callable(self.implementation)
        )


class Container:
    """
    依赖注入容器
    
    示例:
        ```python
        container = Container()
        
        # 注册服务
        container.register(DatabaseService, SQLiteDatabase, Lifetime.SINGLETON)
        container.register_instance(Config, my_config)
        container.register_factory(HttpClient, lambda c: HttpClient(c.resolve(Config)))
        
        # 解析服务
        db = container.resolve(DatabaseService)
        
        # 作用域
        async with container.create_scope() as scope:
            scoped_service = scope.resolve(ScopedService)
        ```
    """
    
    def __init__(self, parent: Optional["Container"] = None):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._parent = parent
        self._scoped_instances: Dict[Type, Any] = {}
        self._is_scope = parent is not None
    
    # ==================== 注册方法 ====================
    
    def register(
        self,
        service_type: Type[T],
        implementation: Optional[Type[T]] = None,
        lifetime: Lifetime = Lifetime.TRANSIENT,
    ) -> "Container":
        """
        注册服务类型
        
        参数:
            service_type: 服务接口/基类
            implementation: 具体实现类（None 则使用 service_type 自身）
            lifetime: 生命周期
        """
        impl = implementation or service_type
        self._services[service_type] = ServiceDescriptor(
            service_type=service_type,
            implementation=impl,
            lifetime=lifetime,
        )
        logger.debug(
            "Service registered",
            service=service_type.__name__,
            implementation=impl.__name__ if inspect.isclass(impl) else str(impl),
            lifetime=lifetime.value
        )
        return self
    
    def register_singleton(
        self,
        service_type: Type[T],
        implementation: Optional[Type[T]] = None,
    ) -> "Container":
        """注册单例服务"""
        return self.register(service_type, implementation, Lifetime.SINGLETON)
    
    def register_scoped(
        self,
        service_type: Type[T],
        implementation: Optional[Type[T]] = None,
    ) -> "Container":
        """注册作用域服务"""
        return self.register(service_type, implementation, Lifetime.SCOPED)
    
    def register_instance(self, service_type: Type[T], instance: T) -> "Container":
        """
        注册已创建的实例（作为单例）
        """
        descriptor = ServiceDescriptor(
            service_type=service_type,
            implementation=instance,  # type: ignore
            lifetime=Lifetime.SINGLETON,
        )
        descriptor._instance = instance
        self._services[service_type] = descriptor
        logger.debug("Instance registered", service=service_type.__name__)
        return self
    
    def register_factory(
        self,
        service_type: Type[T],
        factory: Callable[["Container"], T],
        lifetime: Lifetime = Lifetime.TRANSIENT,
    ) -> "Container":
        """
        注册工厂函数
        """
        self._services[service_type] = ServiceDescriptor(
            service_type=service_type,
            implementation=factory,
            lifetime=lifetime,
        )
        logger.debug("Factory registered", service=service_type.__name__)
        return self
    
    # ==================== 解析方法 ====================
    
    def resolve(self, service_type: Type[T]) -> T:
        """
        解析服务实例
        
        参数:
            service_type: 服务类型
            
        返回:
            服务实例
            
        异常:
            KeyError: 服务未注册
        """
        # 查找描述符
        descriptor = self._get_descriptor(service_type)
        if descriptor is None:
            raise KeyError(f"Service not registered: {service_type.__name__}")
        
        return self._create_instance(descriptor)
    
    def resolve_optional(self, service_type: Type[T]) -> Optional[T]:
        """解析服务（未注册时返回 None）"""
        try:
            return self.resolve(service_type)
        except KeyError:
            return None
    
    def resolve_all(self, service_type: Type[T]) -> list[T]:
        """解析所有匹配的服务（包括子类）"""
        instances = []
        for registered_type, descriptor in self._services.items():
            if issubclass(registered_type, service_type):
                instances.append(self._create_instance(descriptor))
        return instances
    
    # ==================== 作用域 ====================
    
    @asynccontextmanager
    async def create_scope(self):
        """
        创建作用域容器
        
        示例:
            async with container.create_scope() as scope:
                service = scope.resolve(ScopedService)
        """
        scope = Container(parent=self)
        try:
            yield scope
        finally:
            # 清理作用域实例
            await scope._cleanup_scoped()
    
    async def _cleanup_scoped(self):
        """清理作用域内的实例"""
        for instance in self._scoped_instances.values():
            if hasattr(instance, "close"):
                try:
                    result = instance.close()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.warning("Error closing scoped instance", error=str(e))
        self._scoped_instances.clear()
    
    # ==================== 内部方法 ====================
    
    def _get_descriptor(self, service_type: Type) -> Optional[ServiceDescriptor]:
        """查找服务描述符（包括父容器）"""
        if service_type in self._services:
            return self._services[service_type]
        if self._parent:
            return self._parent._get_descriptor(service_type)
        return None
    
    def _create_instance(self, descriptor: ServiceDescriptor[T]) -> T:
        """创建服务实例"""
        # 单例：返回缓存的实例
        if descriptor.lifetime == Lifetime.SINGLETON:
            if descriptor._instance is not None:
                return descriptor._instance
        
        # 作用域：在当前作用域内缓存
        if descriptor.lifetime == Lifetime.SCOPED:
            if descriptor.service_type in self._scoped_instances:
                return self._scoped_instances[descriptor.service_type]
        
        # 创建实例
        impl = descriptor.implementation
        
        if descriptor.is_instance():
            # 直接注册的实例
            instance = impl
        elif callable(impl):
            if inspect.isclass(impl):
                # 类：自动注入构造函数参数
                instance = self._create_with_injection(impl)
            else:
                # 工厂函数
                instance = impl(self)
        else:
            raise TypeError(f"Cannot create instance from {impl}")
        
        # 缓存
        if descriptor.lifetime == Lifetime.SINGLETON:
            descriptor._instance = instance # type: ignore
        elif descriptor.lifetime == Lifetime.SCOPED:
            self._scoped_instances[descriptor.service_type] = instance
        
        return instance  # type: ignore
    
    def _create_with_injection(self, cls: Type[T]) -> T:
        """创建实例并自动注入依赖"""
        # 获取构造函数参数类型
        try:
            hints = get_type_hints(cls.__init__)
        except Exception:
            hints = {}
        
        # 排除 self 和 return
        hints.pop("return", None)
        
        # 解析依赖
        kwargs = {}
        for param_name, param_type in hints.items():
            if param_type in self._services or (self._parent and param_type in self._parent._services):
                kwargs[param_name] = self.resolve(param_type)
        
        return cls(**kwargs)
    
    # ==================== 工具方法 ====================
    
    def has(self, service_type: Type) -> bool:
        """检查服务是否已注册"""
        return self._get_descriptor(service_type) is not None
    
    def get_registered_services(self) -> list[Type]:
        """获取所有已注册的服务类型"""
        services = list(self._services.keys())
        if self._parent:
            services.extend(self._parent.get_registered_services())
        return services


# 全局默认容器
_default_container: Optional[Container] = None


def get_container() -> Container:
    """获取默认容器"""
    global _default_container
    if _default_container is None:
        _default_container = Container()
    return _default_container


def configure_container(container: Container) -> None:
    """设置默认容器"""
    global _default_container
    _default_container = container


# ==================== 导出 ====================

__all__ = [
    "Container",
    "Lifetime",
    "ServiceDescriptor",
    "get_container",
    "configure_container",
]