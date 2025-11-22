# gecko/plugins/models/__init__.py
"""
Gecko Models Plugin

提供统一的模型接入层。
架构：配置 (Config) -> 工厂 (Factory) -> 注册表 (Registry) -> 驱动 (Driver)
"""
import litellm

# ================= Global Configuration =================
# 关闭 LiteLLM 的遥测和冗余打印，保持控制台整洁
litellm.suppress_debug_info = True
litellm.telemetry = False
# ========================================================

from gecko.plugins.models.base import AbstractModel, BaseChatModel, BaseEmbedder
from gecko.plugins.models.config import ModelConfig
from gecko.plugins.models.drivers.litellm_driver import LiteLLMDriver
from gecko.plugins.models.embedding import LiteLLMEmbedder
from gecko.plugins.models.factory import create_model
from gecko.plugins.models.presets.ollama import OllamaChat, OllamaEmbedder
from gecko.plugins.models.presets.openai import OpenAIChat, OpenAIEmbedder
from gecko.plugins.models.presets.zhipu import ZhipuChat

# 确保默认驱动被注册
import gecko.plugins.models.drivers.litellm_driver  # noqa: F401

__all__ = [
    "ModelConfig",
    "AbstractModel",
    "BaseChatModel",
    "BaseEmbedder",
    "LiteLLMDriver",
    "LiteLLMEmbedder",
    "create_model",
    "OpenAIChat",
    "OpenAIEmbedder",
    "OllamaChat",
    "OllamaEmbedder",
    "ZhipuChat",
]