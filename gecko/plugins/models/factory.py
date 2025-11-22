# gecko/plugins/models/factory.py
from gecko.core.exceptions import ConfigurationError
from gecko.plugins.models.base import BaseChatModel
from gecko.plugins.models.config import ModelConfig
from gecko.plugins.models.registry import get_driver_class

# 导入默认驱动以触发注册
import gecko.plugins.models.drivers.litellm_driver  # noqa: F401


def create_model(config: ModelConfig) -> BaseChatModel:
    """
    根据配置创建模型实例 (工厂方法)
    """
    driver_cls = get_driver_class(config.driver_type)
    
    if not driver_cls:
        raise ConfigurationError(
            f"Driver '{config.driver_type}' not found. "
            f"Available drivers: {list(gecko.plugins.models.registry._DRIVER_REGISTRY.keys())}. " # type: ignore
            f"Have you registered it?"
        )
    
    return driver_cls(config)