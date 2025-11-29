# gecko/plugins/models/factory.py
from gecko.core.exceptions import ConfigurationError
from gecko.plugins.models.base import BaseChatModel
from gecko.plugins.models.config import ModelConfig
from gecko.plugins.models.registry import get_driver_class, list_drivers

# 导入默认驱动以触发注册
import gecko.plugins.models.drivers.litellm_driver  # noqa: F401


def create_model(config: ModelConfig) -> BaseChatModel:
    """
    根据配置创建模型实例 (工厂方法)
    """
    driver_cls = get_driver_class(config.driver_type)
    
    if not driver_cls:
        available = list_drivers()
        raise ConfigurationError(
            f"Driver '{config.driver_type}' not found. "
            f"Available drivers: {available}. "
            f"Have you registered it with @register_driver('{config.driver_type}')?"
        )
    
    return driver_cls(config)