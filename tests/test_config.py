# tests/test_config.py
import pytest
import os
from gecko.config import GeckoSettings

def test_config_validation():
    # 正常配置
    conf = GeckoSettings(log_level="DEBUG", log_format="json")
    assert conf.log_level == "DEBUG"
    
    # 错误 Log Level
    with pytest.raises(ValueError):
        GeckoSettings(log_level="INVALID")

    # 错误 Log Format
    with pytest.raises(ValueError):
        GeckoSettings(log_format="xml")

def test_env_loading(monkeypatch):
    monkeypatch.setenv("GECKO_DEFAULT_MODEL", "env-model-v1")
    monkeypatch.setenv("GECKO_MAX_TURNS", "10")
    
    conf = GeckoSettings()
    assert conf.default_model == "env-model-v1"
    assert conf.max_turns == 10