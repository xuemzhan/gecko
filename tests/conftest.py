# tests/conftest.py
import gc
import os
import warnings
import pytest
import asyncio
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

from gecko.core.events import EventBus
from gecko.core.memory import TokenMemory
from gecko.core.protocols import ModelProtocol
from gecko.core.toolbox import ToolBox

from gecko.plugins.storage.interfaces import SessionInterface
from gecko.plugins.tools.registry import ToolRegistry

def pytest_configure(config):
    """
    Pytest 配置钩子
    """
    # 1. 屏蔽 Pydantic 序列化警告
    # 原因: LiteLLM 内部处理智谱/Ollama 等非标准 OpenAI 响应时，会触发 Pydantic v2 的序列化警告。
    # 既然 Gecko 使用 Adapter 手动提取数据，这些上游警告是无关噪音，可以安全忽略。
    warnings.filterwarnings(
        "ignore", 
        category=UserWarning, 
        message="Pydantic serializer warnings"
    )
    
    # 2. 屏蔽 LiteLLM 的一些 verbose 输出
    os.environ["LITELLM_LOG"] = "ERROR"

@pytest.fixture(autouse=True)
def clean_tool_registry():
    """
    [新增] 自动清理工具注册表
    确保每个测试用例都在一个干净的注册表状态下运行，
    避免 Test A 注册的工具影响 Test B。
    """
    # 备份当前状态
    original_registry = ToolRegistry._registry.copy()
    yield
    # 还原状态
    ToolRegistry._registry = original_registry

@pytest.fixture(autouse=True)
def env_setup():
    """自动设置测试环境"""
    # 确保测试期间不会意外读取 .env 造成依赖
    # 可以在这里 mock 环境变量
    pass

@pytest.fixture
def event_loop():
    """创建测试用的 EventLoop"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_toolbox():
    tb = MagicMock(spec=ToolBox)
    # [Fix] 添加 description 字段以通过 System Prompt 的 Jinja2 渲染
    tb.to_openai_schema.return_value = [{
        "type": "function", 
        "function": {
            "name": "t1", 
            "description": "mock tool description" 
        }
    }]
    tb.execute_many = AsyncMock(return_value=[])
    return tb

@pytest.fixture
def mock_llm():
    """
    Mock LLM 对象
    必须返回对象，且实现 count_tokens 以通过 ModelProtocol 检查
    适配 v0.2.1 ModelProtocol，包含同步的 count_tokens 方法
    """
    llm = MagicMock(spec=ModelProtocol)
    
    # 1. 模拟 acompletion (异步推理)
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Test Response", tool_calls=None))
    ]
    # 适配 Pydantic/Dict 访问
    mock_response.choices[0].message.model_dump.return_value = {
        "role": "assistant", 
        "content": "Test Response"
    }
    # 让 message 既像对象也像字典
    mock_response.choices[0].message.__getitem__ = lambda s, k: {
        "role": "assistant", "content": "Test Response"
    }.get(k)
    
    llm.acompletion = AsyncMock(return_value=mock_response)
    
    # 2. 模拟 astream (异步流式)
    async def async_gen(*args, **kwargs):
        chunk = MagicMock()
        chunk.choices = [{"delta": {"content": "stream"}, "index": 0}]
        chunk.content = "stream"
        yield chunk
    llm.astream = MagicMock(side_effect=async_gen)

    # 3. [关键] 模拟 count_tokens (同步计数)
    # 必须接受 text_or_messages 参数
    llm.count_tokens = MagicMock(return_value=10)
    
    return llm


@pytest.fixture
def model(mock_llm):
    """
    model fixture 是 mock_llm 的别名，用于某些特定测试
    """
    return mock_llm


@pytest.fixture
def mock_storage():
    """[New] Mock 存储后端"""
    storage = MagicMock(spec=SessionInterface)
    storage.get = AsyncMock(return_value=None)
    storage.set = AsyncMock()
    storage.delete = AsyncMock()
    return storage

@pytest.fixture
def memory(mock_llm, mock_storage):
    """
    [Updated] Memory Fixture
    注入 storage 和 model_driver
    """
    return TokenMemory(
        session_id="test_session", 
        max_tokens=4000, 
        model_name="gpt-3.5-turbo",
        model_driver=mock_llm,  # 注入驱动
        storage=mock_storage,   # 注入存储
        enable_async_counting=False # 测试中默认同步，便于调试，特定测试可覆盖
    )


# @pytest.fixture
# def memory():
#     return TokenMemory(session_id="test_session", model_name="gpt-3.5-turbo")

@pytest.fixture
def toolbox():
    return ToolBox()

@pytest.fixture
def event_bus():
    return EventBus()

@pytest.fixture(autouse=True)
async def cleanup_litellm_resources():
    """
    [自动执行] 每个测试结束后，强制关闭 LiteLLM 的全局异步客户端。
    防止 ResourceWarning: unclosed socket 导致测试失败。
    """
    yield  # 等待测试执行完成
    
    # 尝试导入 litellm，如果未安装则跳过
    try:
        import litellm
    except ImportError:
        return

    # 清理 LiteLLM 可能持有的各种全局 Client
    # 注意：LiteLLM 不同版本的内部变量名可能不同，这里做防御性处理
    
    clients_to_close = []

    # 1. 处理 async_http_handler (新版)
    if hasattr(litellm, "async_http_handler") and litellm.async_http_handler: # type: ignore
        if hasattr(litellm.async_http_handler, "client"): # type: ignore
            clients_to_close.append(litellm.async_http_handler.client) # type: ignore
        # 清空引用，迫使下个测试重建
        litellm.async_http_handler = None # type: ignore

    # 2. 处理 module_level_aclient (某些版本)
    if hasattr(litellm, "module_level_aclient") and litellm.module_level_aclient:
        clients_to_close.append(litellm.module_level_aclient)
        litellm.module_level_aclient = None

    # 3. 处理 http_client (旧版)
    if hasattr(litellm, "http_client") and litellm.http_client: # type: ignore
        clients_to_close.append(litellm.http_client) # type: ignore
        litellm.http_client = None # type: ignore

    # 执行关闭
    for client in clients_to_close:
        try:
            if hasattr(client, "aclose"):
                await client.aclose()
            elif hasattr(client, "close"):
                if asyncio.iscoroutinefunction(client.close):
                    await client.close()
                else:
                    client.close()
        except Exception:
            pass
            
    # 强制垃圾回收，确保 socket 立即释放
    gc.collect()