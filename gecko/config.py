# gecko/config.py  
"""  
配置系统（改进版）  
  
- 增加 configure_settings / reset_settings，便于测试重载  
- 使用 Lazy 初始化，防止导入 config 时立刻读取 .env  
- Docstring 更新，避免示例与 Agent API 不匹配  
"""  
  
from __future__ import annotations  
  
from typing import Optional  
  
from pydantic import Field, field_validator  
from pydantic_settings import BaseSettings, SettingsConfigDict  
  
  
class GeckoSettings(BaseSettings):  
    default_model: str = Field(default="gpt-3.5-turbo")  
    default_api_key: str = Field(default="")  
    default_base_url: Optional[str] = None  
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)  
  
    max_turns: int = Field(default=5, ge=1, le=50)  
    max_context_tokens: int = Field(default=4000, ge=100)  
  
    default_storage_url: str = Field(default="sqlite://./gecko_data.db")  
    log_level: str = Field(default="INFO")  
    log_format: str = Field(default="text")  
  
    enable_cache: bool = True  
    tool_execution_timeout: float = Field(default=30.0, ge=1.0)  
  
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
  
    @field_validator("log_format")  
    @classmethod  
    def validate_log_format(cls, v: str) -> str:  
        if v not in {"text", "json"}:  
            raise ValueError("log_format must be 'text' or 'json'")  
        return v  
  
  
_default_settings: Optional[GeckoSettings] = None  
  
  
def get_settings(force_reload: bool = False) -> GeckoSettings:  
    global _default_settings  
    if _default_settings is None or force_reload:  
        _default_settings = GeckoSettings()  
    return _default_settings  
  
  
def configure_settings(**overrides) -> GeckoSettings:  
    """  
    允许测试/脚本传入覆盖参数，例如：  
        configure_settings(default_model="gpt-4")  
    """  
    global _default_settings  
    _default_settings = GeckoSettings(**overrides)  
    return _default_settings  
  
  
def reset_settings():  
    global _default_settings  
    _default_settings = None  
  
  
settings = get_settings()  
