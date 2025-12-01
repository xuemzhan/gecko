import pytest
import os
from unittest.mock import patch, MagicMock
from gecko.plugins.storage.mixins import AtomicWriteMixin
from gecko.core.exceptions import ConfigurationError

class MockStorage(AtomicWriteMixin):
    pass

def test_atomic_mixin_production_strict_check():
    """验证在生产环境下缺失 filelock 会抛出异常"""
    
    # 模拟 filelock 未安装
    with patch("gecko.plugins.storage.mixins.FILELOCK_AVAILABLE", False):
        
        storage = MockStorage()
        
        # 场景 1: 开发环境 (默认) -> 仅警告
        with patch.dict(os.environ, {"GECKO_ENV": "development"}):
            with patch("gecko.plugins.storage.mixins.logger") as mock_logger:
                storage.setup_multiprocess_lock("./test.lock")
                mock_logger.warning.assert_called()
                
        # 场景 2: 生产环境 -> 抛出异常
        with patch.dict(os.environ, {"GECKO_ENV": "production"}):
            with pytest.raises(ConfigurationError, match="Running in PRODUCTION mode without filelock"):
                storage.setup_multiprocess_lock("./test.lock")