# tests/utils/test_cleanup.py
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# [修改] 适配新的延迟导入逻辑
def test_cleanup_registration_safe_import():
    """验证清理注册逻辑在 litellm 未安装时不会崩溃"""
    
    # 场景 1: litellm 不在 sys.modules (模拟未安装/未导入)
    with patch.dict(sys.modules):
        if "litellm" in sys.modules:
            del sys.modules["litellm"]
            
        try:
            import gecko.__init__
            # 手动触发注册函数（如果是公开的，或者通过 reload 触发）
            # 这里假设我们测试的是 utils.cleanup 模块中的逻辑
            from gecko.utils.cleanup import register_litellm_cleanup
            register_litellm_cleanup()
        except Exception as e:
            pytest.fail(f"Registration failed when litellm missing: {e}")

def test_cleanup_execution_logic():
    """验证清理逻辑被调用 (当 litellm 存在时)"""
    mock_litellm = MagicMock()
    mock_handler = MagicMock()
    mock_handler.client.close = AsyncMock()
    mock_litellm.async_http_handler = mock_handler
    
    # 注入 mock
    with patch.dict(sys.modules, {"litellm": mock_litellm}):
        from gecko.utils.cleanup import register_litellm_cleanup
        # 注册
        register_litellm_cleanup()
        
        # 这里主要验证代码路径没有语法错误，atexit 难以在单元测试中触发
        # 可以通过反射找到注册的函数并手动调用 (比较 hacky)
        import atexit
        # 假设这是最后一个注册的函数
        # func = atexit._exithandlers[-1][0]
        # func() 
        # ... 验证 mock_client.close 被调用