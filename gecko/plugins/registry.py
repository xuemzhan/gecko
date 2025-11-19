from typing import Any, Callable, Dict, Type, Optional, List
from pluggy import PluginManager

# 单例注册表（线程安全）
class Registry:
    _instances: Dict[str, 'Registry'] = {}

    def __new__(cls, name: str):
        if name not in cls._instances:
            cls._instances[name] = super(Registry, cls).__new__(cls)
            cls._instances[name]._registry: Dict[str, Any] = {} # type: ignore
            cls._instances[name]._pm = PluginManager(name)  # 未来 entry_points
        return cls._instances[name]

    def register(self, key: str, item: Any):
        if key in self._registry:
            raise ValueError(f"Key {key} already registered")
        self._registry[key] = item

    def get(self, key: str) -> Optional[Any]:
        return self._registry.get(key)

    def list(self) -> List[str]:
        return list(self._registry.keys())

# 全局实例
model_registry = Registry("models")
tool_registry = Registry("tools")
storage_registry = Registry("storage")

# 装饰器工厂
def register_model(name: str) -> Callable:
    def decorator(cls: Type):
        model_registry.register(name, cls)
        return cls
    return decorator

def register_tool(name: str) -> Callable:
    def decorator(cls: Type):
        tool_registry.register(name, cls)
        return cls
    return decorator

def register_storage(name: str) -> Callable:
    def decorator(cls: Type):
        storage_registry.register(name, cls)
        return cls
    return decorator

# 在 registry.py 的 get 方法中添加
def get(self, key: str) -> Any:
    item = self._registry.get(key)
    if item is None:
        from gecko.core.exceptions import PluginNotFoundError
        raise PluginNotFoundError(self._pm.project_name, key)
    return item() if isinstance(item, type) else item  # 支持类懒实例化

# 示例：未来用 pluggy.load_setuptools_entrypoints() 自动加载社区插件