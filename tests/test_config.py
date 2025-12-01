# tests/core/test_config.py
"""
Gecko 配置模块单元测试

测试目标：
1. 验证所有配置项的默认值是否符合预期。
2. 验证环境变量 (GECKO_*) 是否能正确覆盖配置。
3. 验证 configure_settings 运行时覆盖机制。
4. 验证 Pydantic 校验逻辑 (数值范围、枚举值、日志级别)。
5. 验证单例模式与重置逻辑。
"""
import os
import pytest
from pydantic import ValidationError
from gecko.config import (
    GeckoSettings,
    get_settings,
    configure_settings,
    reset_settings,
)

@pytest.fixture
def clean_settings():
    """
    Fixture: 在测试前后清理配置单例和环境变量，防止测试污染
    """
    # Teardown (pre-cleanup)
    reset_settings()
    # 备份当前环境变量
    old_environ = dict(os.environ)
    # 清理所有 GECKO_ 开头的环境变量
    for key in list(os.environ.keys()):
        if key.startswith("GECKO_"):
            del os.environ[key]
            
    yield
    
    # Teardown (restore)
    os.environ.clear()
    os.environ.update(old_environ)
    reset_settings()

def test_default_values(clean_settings):
    """测试所有配置项的默认值"""
    settings = get_settings()
    
    # Model
    assert settings.default_model == "gpt-4o"
    assert settings.default_api_key == ""
    assert settings.default_base_url is None
    assert settings.default_temperature == 0.7
    assert settings.default_model_timeout == 30.0
    
    # Agent
    assert settings.max_turns == 10
    assert settings.max_context_tokens == 4000
    
    # Workflow
    assert settings.workflow_checkpoint_strategy == "final"
    assert settings.workflow_history_retention == 20
    
    # Storage
    assert settings.default_storage_url == "sqlite:///./gecko_data.db"
    assert settings.storage_pool_size == 5
    assert settings.storage_max_overflow == 10
    
    # Memory
    assert settings.memory_summary_interval == 30.0
    assert settings.memory_cache_size == 2000
    assert settings.memory_summary_reserve_tokens == 500
    
    # Telemetry
    assert settings.telemetry_enabled is True
    assert settings.telemetry_service_name == "gecko-app"
    
    # System
    assert settings.log_level == "INFO"
    assert settings.log_format == "text"
    assert settings.enable_cache is True
    assert settings.tool_execution_timeout == 30.0

def test_environment_variable_override(clean_settings, monkeypatch):
    """测试环境变量覆盖默认值"""
    # 设置环境变量
    monkeypatch.setenv("GECKO_DEFAULT_MODEL", "claude-3-opus")
    monkeypatch.setenv("GECKO_MAX_TURNS", "50")
    monkeypatch.setenv("GECKO_WORKFLOW_CHECKPOINT_STRATEGY", "always")
    monkeypatch.setenv("GECKO_STORAGE_POOL_SIZE", "20")
    monkeypatch.setenv("GECKO_TELEMETRY_ENABLED", "False")
    monkeypatch.setenv("GECKO_DEFAULT_TEMPERATURE", "0.1")
    
    # 必须 force_reload=True 才能重新读取环境变量
    settings = get_settings(force_reload=True)
    
    assert settings.default_model == "claude-3-opus"
    assert settings.max_turns == 50
    assert settings.workflow_checkpoint_strategy == "always"
    assert settings.storage_pool_size == 20
    assert settings.telemetry_enabled is False
    assert settings.default_temperature == 0.1

def test_configure_settings_override(clean_settings):
    """测试代码运行时覆盖 (configure_settings)"""
    # 1. 初始获取
    s1 = get_settings()
    assert s1.default_api_key == ""
    
    # 2. 运行时配置
    s2 = configure_settings(
        default_api_key="sk-test-key",
        log_format="json",
        memory_cache_size=5000
    )
    
    # 3. 验证覆盖生效
    assert s2.default_api_key == "sk-test-key"
    assert s2.log_format == "json"
    assert s2.memory_cache_size == 5000
    
    # 4. 验证单例更新 (s1 已经是旧引用，再次 get 应该是新的)
    s3 = get_settings()
    assert s3.default_api_key == "sk-test-key"
    assert s3 is s2  # 引用相同

def test_singleton_pattern(clean_settings):
    """测试单例模式"""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
    
    # 强制重载应生成新实例
    s3 = get_settings(force_reload=True)
    assert s3 is not s1

def test_reset_settings(clean_settings):
    """测试重置功能"""
    s1 = get_settings()
    configure_settings(default_model="test-model")
    
    reset_settings()
    
    # 重置后再次获取应恢复默认值
    s2 = get_settings()
    assert s2.default_model == "gpt-4o"
    assert s2 is not s1

# ==================== 校验器测试 (Validators & Constraints) ====================

def test_log_level_validation(clean_settings):
    """测试日志级别校验"""
    # 1. 大小写不敏感自动转换
    s = configure_settings(log_level="debug")
    assert s.log_level == "DEBUG"
    
    # 2. 无效值抛出异常
    with pytest.raises(ValidationError) as exc:
        configure_settings(log_level="UNKNOWN_LEVEL")
    assert "log_level must be one of" in str(exc.value)

def test_numeric_constraints(clean_settings):
    """测试数值范围约束"""
    # default_temperature (0.0 - 2.0)
    with pytest.raises(ValidationError):
        configure_settings(default_temperature=3.0)
    
    with pytest.raises(ValidationError):
        configure_settings(default_temperature=-1.0)
        
    # default_model_timeout (>= 5.0)
    with pytest.raises(ValidationError):
        configure_settings(default_model_timeout=1.0)
        
    # max_turns (1 - 100)
    with pytest.raises(ValidationError):
        configure_settings(max_turns=0)
        
    # storage_pool_size (>= 1)
    with pytest.raises(ValidationError):
        configure_settings(storage_pool_size=0)
        
    # memory_summary_interval (>= 5.0)
    with pytest.raises(ValidationError):
        configure_settings(memory_summary_interval=1.0)

def test_literal_constraints(clean_settings):
    """测试枚举值 (Literal) 约束"""
    # log_format: text / json
    with pytest.raises(ValidationError):
        configure_settings(log_format="xml") 
        
    # workflow_checkpoint_strategy: always / final / manual
    with pytest.raises(ValidationError):
        configure_settings(workflow_checkpoint_strategy="random")

def test_ignore_extra_env_vars(clean_settings, monkeypatch):
    """测试忽略未定义的配置项 (extra='ignore')"""
    monkeypatch.setenv("GECKO_UNKNOWN_VAR", "some_value")
    # 不应报错
    settings = get_settings(force_reload=True)
    assert settings.default_model == "gpt-4o"