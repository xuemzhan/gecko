# gecko/config.py
"""
全局配置系统 (Final Production Version)

变更记录：
- [Add] Memory: 补充 LRU 缓存大小和摘要预留 Token 配置
- [Add] Telemetry: 补充服务名称与全局开关
"""

from __future__ import annotations

from typing import Optional, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GeckoSettings(BaseSettings):
    # ================= 1. Model & Inference (模型与推理) =================
    default_model: str = Field(default="gpt-4o", description="默认 LLM 模型名称")
    default_api_key: str = Field(default="", description="默认 API Key")
    default_base_url: Optional[str] = Field(default=None, description="自定义 API Base URL")
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    default_model_timeout: float = Field(default=30.0, ge=5.0, description="LLM 请求超时(秒)")

    # ================= 2. Agent Execution (智能体运行时) =================
    max_turns: int = Field(default=10, ge=1, le=100, description="最大对话轮数")
    max_context_tokens: int = Field(default=4000, ge=100, description="上下文窗口限制")

    # ================= 3. Workflow Engine (工作流引擎) =================
    workflow_checkpoint_strategy: Literal["always", "final", "manual"] = Field(
        default="final", 
        description="持久化策略: always(每步), final(仅结束), manual(手动)"
    )
    workflow_history_retention: int = Field(
        default=20, 
        ge=1, 
        description="持久化时保留的历史记录步数(防止 IO 爆炸)"
    )

    # ================= 4. Storage & Database (存储层) =================
    default_storage_url: str = Field(
        default="sqlite:///./gecko_data.db", 
        description="数据库连接串 (sqlite/redis/postgres/chroma)"
    )
    storage_pool_size: int = Field(default=5, ge=1, description="连接池大小")
    storage_max_overflow: int = Field(default=10, ge=0, description="连接池最大溢出")

    # ================= 5. Memory Management (记忆模块) =================
    memory_summary_interval: float = Field(default=30.0, ge=5.0, description="摘要更新防抖间隔(秒)")
    # [新增]
    memory_cache_size: int = Field(default=2000, ge=100, description="Token 计数器的 LRU 缓存大小")
    # [新增]
    memory_summary_reserve_tokens: int = Field(default=500, ge=0, description="为摘要预留的 Token 数")

    # ================= 6. Telemetry & Observability (遥测) =================
    # [新增]
    telemetry_enabled: bool = Field(default=True, description="是否启用 OpenTelemetry")
    # [新增]
    telemetry_service_name: str = Field(default="gecko-app", description="Trace 服务名称")

    # ================= 7. System (系统层) =================
    log_level: str = Field(default="INFO", description="日志级别")
    log_format: Literal["text", "json"] = Field(default="text", description="日志格式")
    enable_cache: bool = Field(default=True, description="是否启用全局缓存")
    tool_execution_timeout: float = Field(default=30.0, ge=1.0, description="工具执行超时(秒)")

    model_config = SettingsConfigDict(
        env_prefix="GECKO_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return v.upper()


# Singleton
_default_settings: Optional[GeckoSettings] = None

def get_settings(force_reload: bool = False) -> GeckoSettings:
    global _default_settings
    if _default_settings is None or force_reload:
        _default_settings = GeckoSettings()
    return _default_settings

def configure_settings(**overrides) -> GeckoSettings:
    global _default_settings
    _default_settings = GeckoSettings(**overrides)
    return _default_settings

def reset_settings():
    global _default_settings
    _default_settings = None

settings = get_settings()