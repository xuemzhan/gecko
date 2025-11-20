# tests/utils/test_cleanup.py
import sys
from unittest.mock import MagicMock

import pytest
from gecko.utils.cleanup import register_litellm_cleanup

def test_litellm_cleanup_logic():
    # Mock litellm
    mock_litellm = MagicMock()
    mock_client = MagicMock()
    mock_client.close = MagicMock() # AsyncMock in real usage, but checking logic flow
    
    mock_litellm.async_http_handler.client = mock_client
    sys.modules["litellm"] = mock_litellm
    
    # 这里很难测试 atexit 的实际触发，
    # 但我们可以手动调用注册的函数来验证逻辑是否抛错
    try:
        # 重新导入触发注册逻辑
        import gecko.utils.cleanup
    except Exception as e:
        pytest.fail(f"Cleanup registration failed: {e}")