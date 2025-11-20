# gecko/config.py
"""
Gecko 配置管理（改进版）

改进：
1. 保留 pydantic-settings
2. 支持依赖注入（不强制全局单例）
3. 提供默认实例便于快速使用
"""
from __future__ import annotations
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict # type: ignore

class GeckoSettings(BaseSettings):
    """
    Gecko 配置
    
    使用方式 1（快速开发，全局配置）:
        from gecko.config import settings
        print(settings.default_model)
    
    使用方式 2（测试/多实例，依赖注入）:
        config = GeckoSettings(default_model="gpt-4")
        agent = Agent(config=config)
    """
    
    # ========== LLM 配置 ==========
    default_model: str = Field(
        default="gpt-3.5-turbo",
        description="默认模型名称"
    )
    
    default_api_key: str = Field(
        default="",
        description="API Key"
    )
    
    default_base_url: Optional[str] = Field(
        default=None,
        description="API Base URL"
    )
    
    default_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="生成温度"
    )
    
    # ========== 引擎配置 ==========
    max_turns: int = Field(
        default=5,
        ge=1,
        le=50,
        description="最大推理轮次"
    )
    
    max_context_tokens: int = Field(
        default=4000,
        ge=100,
        description="上下文最大 Token 数"
    )
    
    # ========== 存储配置 ==========
    default_storage_url: str = Field(
        default="sqlite://./gecko_data.db",
        description="存储后端 URL"
    )
    
    # ========== 日志配置 ==========
    log_level: str = Field(
        default="INFO",
        description="日志级别"
    )
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """验证日志级别"""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v_upper
    
    log_format: str = Field(
        default="text",
        description="日志格式：text 或 json"
    )
    
    @field_validator('log_format')
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        """验证日志格式"""
        if v not in {"text", "json"}:
            raise ValueError("log_format must be 'text' or 'json'")
        return v
    
    # ========== 性能配置 ==========
    enable_cache: bool = Field(
        default=True,
        description="启用请求级缓存"
    )
    
    tool_execution_timeout: float = Field(
        default=30.0,
        ge=1.0,
        description="工具执行超时（秒）"
    )
    
    # ========== Pydantic 配置 ==========
    model_config = SettingsConfigDict(
        env_prefix="GECKO_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

# ========== 全局默认实例（可选使用） ==========

_default_settings: Optional[GeckoSettings] = None

def get_settings() -> GeckoSettings:
    """
    获取全局配置实例
    
    注意：这是便捷方法，推荐在生产代码中使用依赖注入
    """
    global _default_settings
    if _default_settings is None:
        _default_settings = GeckoSettings()
    return _default_settings

# 默认实例（快速开发用）
settings = get_settings()