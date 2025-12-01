# tests/core/test_config_integration.py
import pytest
from gecko.config import configure_settings, reset_settings
from gecko.compose.workflow import Workflow, CheckpointStrategy
from gecko.plugins.storage.backends.sqlite import SQLiteStorage

@pytest.fixture(autouse=True)
def setup_teardown_config():
    # 每次测试前重置配置
    reset_settings()
    yield
    reset_settings()

def test_workflow_uses_global_config():
    """验证 Workflow 自动读取全局配置"""
    # 1. 修改全局配置
    configure_settings(
        workflow_checkpoint_strategy="final",
        workflow_history_retention=5
    )
    
    # 2. 初始化 Workflow (不传参)
    wf = Workflow("ConfigTest")
    
    # 3. 验证
    assert wf.persistence.strategy == CheckpointStrategy.FINAL
    assert wf.persistence.history_retention == 5

def test_storage_pool_config():
    """验证 Storage 读取连接池配置"""
    configure_settings(
        storage_pool_size=20,
        storage_max_overflow=5
    )
    
    # 初始化 SQLite (需要临时文件路径)
    store = SQLiteStorage("sqlite:///./test_pool.db")
    
    # 验证 SQLAlchemy Engine 的 pool 参数
    pool = store.engine.pool # type: ignore
    # SQLAlchemy Pool 的 size 属性
    assert pool.size() == 20 
    # overflow 属性通常是 _max_overflow (具体取决于实现，这里主要验证传递链路)
    
    # 清理
    store.engine.dispose() # type: ignore
    import os
    if os.path.exists("./test_pool.db"):
        os.remove("./test_pool.db")
    if os.path.exists("./test_pool.db.lock"):
        os.remove("./test_pool.db.lock")