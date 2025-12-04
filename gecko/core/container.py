# gecko/core/container.py
"""
轻量级依赖注入容器

提供服务的注册、解析和生命周期管理。
"""
from __future__ import annotations

import asyncio
import inspect
import sys
from typing import (
    Any, Callable, Dict, Generic, Optional, Type, TypeVar, Union, 
    get_type_hints, overload, get_origin, get_args
)
from threading import RLock
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
        impl = self.implementation
        # 如果实现是类 -> 不是实例
        if inspect.isclass(impl):
            return False
        # 如果实现是该服务类型的实例 -> 视为实例
        try:
            if isinstance(impl, self.service_type):
                return True
        except Exception:
            # isinstance 可能因为 typing generics 等抛异常，忽略
            pass
        # 可调用对象通常是工厂/函数，不能视为已预创建的实例
        if callable(impl):
            return False
        # 否则视为实例
        return True


from gecko.core.exceptions import (
    ServiceNotRegisteredError,
    CircularDependencyError,
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
        # 用于检测解析时的循环依赖（保存正在解析的服务类型）
        self._resolving: set[Type] = set()
        # 保护单例初始化的同步锁，防止并发创建多个单例
        self._singleton_lock = RLock()
        # 异步路径的锁（用于 async resolve）
        self._async_lock = asyncio.Lock()
    
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
            raise ServiceNotRegisteredError(f"Service not registered: {service_type.__name__}")

        # 循环依赖检测
        if service_type in self._resolving:
            raise CircularDependencyError(f"Circular dependency detected while resolving {service_type.__name__}")

        try:
            self._resolving.add(service_type)
            return self._create_instance(descriptor)
        finally:
            self._resolving.discard(service_type)
    
    def resolve_optional(self, service_type: Type[T]) -> Optional[T]:
        """解析服务（未注册时返回 None）"""
        try:
            return self.resolve(service_type)
        except KeyError:
            return None
    
    def resolve_all(self, service_type: Type[T]) -> list[T]:
        """解析所有匹配的服务（包括子类）"""
        instances = []
        # 遍历当前容器及父容器，收集所有匹配项
        for registered_type, descriptor in self._services.items():
            try:
                if inspect.isclass(registered_type) and issubclass(registered_type, service_type):
                    instances.append(self._create_instance(descriptor))
            except Exception:
                # 若 registered_type 不是类或比较失败，则跳过
                continue
        if self._parent:
            instances.extend(self._parent.resolve_all(service_type))
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

    async def shutdown(self):
        """
        清理容器中所有单例实例的生命周期资源（如果有 close 或 aclose 方法）。
        这是一个异步方法，适合在应用退出时调用，确保引用的资源被正确释放。
        """
        # 清理当前容器注册的单例实例
        for descriptor in self._services.values():
            inst = descriptor._instance
            if not inst:
                continue
            try:
                # 优先支持异步关闭方法 `aclose`
                if hasattr(inst, "aclose"):
                    res = inst.aclose()
                    if asyncio.iscoroutine(res):
                        await res
                elif hasattr(inst, "close"):
                    res = inst.close()
                    if asyncio.iscoroutine(res):
                        await res
            except Exception as e:
                logger.warning("Error shutting down singleton instance", error=str(e))
    
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
            # 使用锁保护单例创建，避免并发重复创建
            with self._singleton_lock:
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
                # 类：自动注入构造函数参数（同步路径）
                instance = self._create_with_injection(impl)
            else:
                # 工厂函数，调用并允许工厂返回协程（但同步 resolve 不 await）
                instance = impl(self)
                # 如果工厂返回协程，提示使用 async resolve 更合适
                if asyncio.iscoroutine(instance):
                    raise RuntimeError("Factory returned a coroutine. Use 'resolve_async' for async factories.")
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
        # 获取构造函数参数类型，尽量在正确的模块命名空间解析前向引用
        try:
            module = sys.modules.get(cls.__module__)
            globalns = getattr(module, "__dict__", None)
            hints = get_type_hints(cls.__init__, globalns=globalns)
        except Exception:
            # 若 get_type_hints 失败（例如局部/前向引用在当前全局命名空间不可解析），
            # 回退到直接使用 __annotations__（可能包含字符串或 ForwardRef）
            hints = getattr(cls.__init__, "__annotations__", {})

        # 排除 self 和 return
        hints.pop("return", None)

        # 解析依赖，支持 Optional[...] 等 typing 包装类型
        kwargs = {}
        for param_name, param_type in hints.items():
            origin = get_origin(param_type)
            target_type = None
            if origin is Union:
                # 处理 Optional[T] => Union[T, NoneType]
                args = get_args(param_type)
                non_none = [a for a in args if a is not type(None)]  # noqa: E721
                if len(non_none) == 1:
                    target_type = non_none[0]
            elif origin is not None:
                # 其他泛型/容器类型，暂不支持注入
                target_type = None
            else:
                target_type = param_type

            # 如果 target_type 不是具体类型（例如字符串或 ForwardRef），尝试按名称匹配已注册类型
            if not isinstance(target_type, type):
                name = None
                if isinstance(target_type, str):
                    name = target_type
                else:
                    name = getattr(target_type, "__forward_arg__", None)

                if name:
                    for registered in self.get_registered_services():
                        if getattr(registered, "__name__", None) == name:
                            target_type = registered
                            break
                if not isinstance(target_type, type):
                    # 仍然无法解析为类型，跳过注入
                    continue

            # 使用 _get_descriptor 支持父链查找
            if self._get_descriptor(target_type) is not None:
                kwargs[param_name] = self.resolve(target_type)

        return cls(**kwargs)

    async def resolve_async(self, service_type: Type[T]) -> T:
        """
        异步解析服务实例，支持工厂返回协程或实例的异步初始化场景。
        在遇到同步工厂/实现时也能正常工作（会在同步路径创建实例）。
        """
        descriptor = self._get_descriptor(service_type)
        if descriptor is None:
            raise ServiceNotRegisteredError(f"Service not registered: {service_type.__name__}")

        # 循环依赖检测
        if service_type in self._resolving:
            raise CircularDependencyError(f"Circular dependency detected while resolving {service_type.__name__}")

        try:
            self._resolving.add(service_type)
            return await self._create_instance_async(descriptor)
        finally:
            self._resolving.discard(service_type)

    async def _create_instance_async(self, descriptor: ServiceDescriptor[T]) -> T:
        """异步创建实例的实现，支持 async 工厂与 async 关闭"""
        # 单例检查/创建在异步路径使用 async 锁
        if descriptor.lifetime == Lifetime.SINGLETON:
            async with self._async_lock:
                if descriptor._instance is not None:
                    return descriptor._instance

        if descriptor.lifetime == Lifetime.SCOPED:
            if descriptor.service_type in self._scoped_instances:
                return self._scoped_instances[descriptor.service_type]

        impl = descriptor.implementation

        if descriptor.is_instance():
            instance = impl
        elif callable(impl):
            if inspect.isclass(impl):
                # 类构造为同步，复用同步注入路径
                instance = self._create_with_injection(impl)
            else:
                # 工厂函数，可能返回协程
                instance = impl(self)
                if asyncio.iscoroutine(instance):
                    instance = await instance
        else:
            raise TypeError(f"Cannot create instance from {impl}")

        # 缓存
        if descriptor.lifetime == Lifetime.SINGLETON:
            descriptor._instance = instance # type: ignore
        elif descriptor.lifetime == Lifetime.SCOPED:
            self._scoped_instances[descriptor.service_type] = instance

        return instance  # type: ignore
    
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