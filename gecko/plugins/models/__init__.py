# gecko/plugins/models/__init__.py
"""
Gecko Models Plugin

提供统一的模型接入层。
架构：配置 (Config) -> 工厂 (Factory) -> 注册表 (Registry) -> 驱动 (Driver)
"""
try:
    import litellm  # type: ignore

    # ================= Global Configuration =================
    # 关闭 LiteLLM 的遥测和冗余打印，保持控制台整洁
    litellm.suppress_debug_info = True
    litellm.telemetry = False
    # ========================================================
except ImportError:
    # 未安装 litellm 时允许导入本模块，但实际使用驱动/Embedder 时仍会抛错
    litellm = None  # type: ignore

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