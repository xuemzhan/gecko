# tests/conftest.py
"""
Pytest 全局配置与测试夹具（fixtures）。

主要职责：
1. 统一屏蔽无关警告 / 第三方库冗余日志；
2. 提供常用的测试对象（Mock LLM、ToolBox、Memory、EventBus 等）；
3. 为每个测试用例清理全局状态（ToolRegistry / LiteLLM 全局客户端）。
"""

from __future__ import annotations

import asyncio
import gc
import os
import warnings
from unittest.mock import AsyncMock, MagicMock

import pytest

from gecko.core.events import EventBus
from gecko.core.memory import TokenMemory
from gecko.core.protocols import ModelProtocol
from gecko.core.toolbox import ToolBox
from gecko.plugins.storage.interfaces import SessionInterface
from gecko.plugins.tools.registry import ToolRegistry


# ----------------------------------------------------------------------
# Pytest 全局配置钩子
# ----------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """
    Pytest 配置钩子，在测试进程启动时执行一次。

    这里主要做两件事：
    1. 屏蔽 Pydantic v2 的序列化警告（与 LiteLLM 内部实现相关，对 Gecko 测试无意义）；
    2. 将 LiteLLM 的日志级别设置为 ERROR，避免测试输出被刷屏。
    """
    # 1) 屏蔽 Pydantic 序列化警告
    # 原因：LiteLLM 在处理部分第三方接口响应（如智谱、Ollama 等）时，
    #       会触发 Pydantic v2 的 "Pydantic serializer warnings"。
    #       Gecko 自己会通过适配层手动提取数据，因此这些警告是无关噪音。
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="Pydantic serializer warnings",
    )

    # 2) 屏蔽 LiteLLM 的部分 verbose 输出
    os.environ["LITELLM_LOG"] = "ERROR"


# ----------------------------------------------------------------------
# 全局状态清理相关 Fixture
# ----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_tool_registry():
    """
    [自动执行] 清理工具注册表（ToolRegistry）。

    目的：
    - 防止上一个测试注册的工具污染下一个测试；
    - 每个测试都在“干净的工具注册表”状态下运行。

    实现方式：
    - 在用例开始前，拷贝当前注册表（浅拷贝）；
    - 在用例结束后，将 ToolRegistry._registry 还原为原始拷贝。
    """
    original_registry = ToolRegistry._registry.copy()
    yield
    ToolRegistry._registry = original_registry


@pytest.fixture(autouse=True)
def env_setup() -> None:
    """
    [自动执行] 测试环境变量预处理钩子。

    当前实现为占位符（不做任何操作）：
    - 目的是保留一个统一入口，后续如需：
      - 屏蔽 .env 读取
      - 统一设置 GECKO_* 环境变量
      - Mock 某些敏感配置
      可集中在此 fixture 中实现。
    """
    # 目前不做任何操作，仅占位
    pass


@pytest.fixture(autouse=True)
async def cleanup_litellm_resources() -> None: # type: ignore
    """
    [自动执行] 每个测试结束后清理 LiteLLM 持有的全局 HTTP 客户端。

    背景：
    - LiteLLM 在内部会维护若干“模块级别”的 HTTP 客户端（AsyncClient/Client），
      若不显式关闭，Python 在退出时可能抛出 ResourceWarning: unclosed socket，
      甚至导致某些测试框架认为测试失败。

    策略：
    - 测试执行完成后尝试导入 litellm；
    - 若存在（即项目安装了 litellm），则扫描常见几个全局变量：
        - async_http_handler  (新版本：通常持有 .client)
        - module_level_aclient
        - http_client         (旧版本)
      将其中的 client 实例收集起来，并将模块级引用清空；
    - 对收集到的 client 调用 aclose/close，忽略所有异常；
    - 最后强制做一次 gc.collect()，加速 socket 释放。
    """
    # 先运行测试用例
    yield # type: ignore

    try:
        import litellm  # type: ignore[import]
    except ImportError:
        # 项目没安装 LiteLLM，则无需清理
        return

    clients_to_close: list[object] = []

    # 1) 处理 async_http_handler（新版本 LiteLLM 常用入口）
    #    其上通常有一个 .client 属性，为真正的 HTTP 客户端。
    if getattr(litellm, "async_http_handler", None):
        handler = litellm.async_http_handler # type: ignore
        client = getattr(handler, "client", None)
        if client is not None:
            clients_to_close.append(client)
        # 清空模块级引用，迫使下个测试重新创建
        litellm.async_http_handler = None # type: ignore

    # 2) 处理 module_level_aclient（部分版本中存在的 async client）
    if getattr(litellm, "module_level_aclient", None):
        clients_to_close.append(litellm.module_level_aclient)
        litellm.module_level_aclient = None

    # 3) 处理 http_client（旧版本 LiteLLM 可能仍在使用）
    if getattr(litellm, "http_client", None):
        clients_to_close.append(litellm.http_client) # type: ignore
        litellm.http_client = None # type: ignore

    # 定义一个通用的“关闭 client”帮助函数
    async def _close_client_like(client: object) -> None:
        """
        尝试以“异步优先”的方式关闭一个 HTTP client 对象：
        - 若存在 async aclose() 方法，则优先使用；
        - 否则若存在 close()，则调用之；若是协程则 await；
        - 所有异常都被忽略（因为关闭本来就是“尽力而为”）。
        """
        try:
            aclose = getattr(client, "aclose", None)
            if callable(aclose):
                await aclose() # type: ignore
                return

            close = getattr(client, "close", None)
            if callable(close):
                if asyncio.iscoroutinefunction(close):
                    await close()
                else:
                    close()
        except Exception:
            # 清理过程中的错误不应影响测试结果
            return

    # 执行关闭
    for client in clients_to_close:
        await _close_client_like(client)

    # 强制垃圾回收，加速 socket 资源回收
    gc.collect()


# ----------------------------------------------------------------------
# 事件循环 / 基础对象 Fixture
# ----------------------------------------------------------------------


@pytest.fixture
def event_loop():
    """
    为 pytest-asyncio 提供独立的事件循环。

    默认行为：
    - 每个测试用例使用一个新的 event loop；
    - 测试结束后关闭该 loop，防止跨用例的状态污染。
    """
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def event_bus() -> EventBus:
    """
    提供一个新的 EventBus 实例。

    说明：
    - 不同测试之间互不共享同一个事件总线，避免订阅/发布的副作用互相影响。
    """
    return EventBus()


# ----------------------------------------------------------------------
# ToolBox / ToolRegistry 相关 Fixture
# ----------------------------------------------------------------------


@pytest.fixture
def toolbox() -> ToolBox: # type: ignore
    """
    提供一个真实的 ToolBox 实例。

    适用场景：
    - 需要测试真实工具注册/调用逻辑的用例。
    """
    return ToolBox() # type: ignore


@pytest.fixture
def mock_toolbox():
    """
    提供一个 Mock 版本的 ToolBox，用于只关心「模型推理」而不关心真实工具执行的场景。

    特性：
    - to_openai_schema() 返回一个包含 description 的伪函数描述，
      以满足 System Prompt 中的 Jinja2 渲染需求；
    - execute_many() 被 Mock 为异步空列表返回，避免真实执行工具。
    """
    tb = MagicMock(spec=ToolBox)

    # Mock 工具 schema（兼容 OpenAI tools / function-calling schema）
    tb.to_openai_schema.return_value = [
        {
            "type": "function",
            "function": {
                "name": "t1",
                "description": "mock tool description",
            },
        }
    ]

    # 所有工具调用都直接返回空列表
    tb.execute_many = AsyncMock(return_value=[])

    return tb


@pytest.fixture(autouse=True)
def clean_tool_registry_alias(clean_tool_registry: None):
    """
    仅为语义清晰保留的别名 fixture（如果不需要可去掉）。

    说明：
    - clean_tool_registry 已经以 autouse=True 运行；
    - 这里通过依赖关系确保其在需要 ToolRegistry 的测试前被执行；
    - 不额外添加逻辑，只是让 conftest 结构更加清晰。
    """
    # 直接依赖 clean_tool_registry 即可，不需要再写逻辑
    yield


# ----------------------------------------------------------------------
# 模型 / Memory / 存储 Fixture
# ----------------------------------------------------------------------


@pytest.fixture
def mock_llm():
    """
    Mock LLM 对象，用于替代真实模型调用。

    要求：
    - 必须满足 ModelProtocol 接口约束（至少包含 acompletion、astream、count_tokens）；
    - acompletion 返回一个带有 .choices[0].message 的对象；
    - message 同时支持:
        - 属性访问: message.content / message.tool_calls
        - Pydantic 风格: message.model_dump()
        - 字典访问: message["content"]
    - astream 返回一个异步生成器，用于流式测试；
    - count_tokens 为同步方法，便于 TokenMemory 进行 token 估算。
    """
    llm = MagicMock(spec=ModelProtocol)

    # ---------------- 1) 模拟 acompletion (异步推理) ----------------
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(content="Test Response", tool_calls=None),
        )
    ]

    # Pydantic 风格的 model_dump()
    mock_response.choices[0].message.model_dump.return_value = {
        "role": "assistant",
        "content": "Test Response",
    }

    # 让 message 同时支持下标访问：message["content"]
    mock_response.choices[0].message.__getitem__ = lambda s, k: {
        "role": "assistant",
        "content": "Test Response",
    }.get(k)

    llm.acompletion = AsyncMock(return_value=mock_response)

    # ---------------- 2) 模拟 astream (异步流式) ----------------
    async def async_gen(*args, **kwargs):
        """
        简单的异步生成器：
        - 仅产生单个 chunk；
        - chunk.choices[0]["delta"]["content"] = "stream"；
        - chunk.content = "stream"。
        """
        chunk = MagicMock()
        chunk.choices = [{"delta": {"content": "stream"}, "index": 0}]
        chunk.content = "stream"
        yield chunk

    # 注意：这里用 MagicMock + side_effect(async_gen)，
    #       保持接口签名兼容 ModelProtocol.astream
    llm.astream = MagicMock(side_effect=async_gen)

    # ---------------- 3) 模拟 count_tokens (同步计数) ----------------
    # 接口要求：count_tokens(text_or_messages) -> int
    llm.count_tokens = MagicMock(return_value=10)

    return llm


@pytest.fixture
def model(mock_llm: MagicMock):
    """
    model fixture 是 mock_llm 的语义别名，用于部分历史测试用例。

    说明：
    - 某些测试文件可能期望 fixture 名为 model，而不是 mock_llm；
    - 这里简单返回同一个对象，避免在测试中到处重命名。
    """
    return mock_llm


@pytest.fixture
def mock_storage():
    """
    Mock 存储后端，实现 SessionInterface 所需的异步接口。

    用途：
    - 为 TokenMemory 注入一个“假的”存储实现，
      以避免测试时连接真实 Redis / 数据库。
    """
    storage = MagicMock(spec=SessionInterface)
    storage.get = AsyncMock(return_value=None)
    storage.set = AsyncMock()
    storage.delete = AsyncMock()
    return storage


@pytest.fixture
def memory(mock_llm: MagicMock, mock_storage: MagicMock) -> TokenMemory:
    """
    提供一个配置好的 TokenMemory 实例。

    特点：
    - 使用 mock_llm 作为 model_driver，用于 token 估算；
    - 使用 mock_storage 作为底层存储，避免真实 IO；
    - enable_async_counting=False，统一使用同步 token 统计，便于测试调试。
    """
    return TokenMemory(
        session_id="test_session",
        max_tokens=4000,
        model_name="gpt-3.5-turbo",
        model_driver=mock_llm,  # 注入模型驱动
        storage=mock_storage,  # 注入存储后端
        enable_async_counting=False,  # 测试中默认同步 token 统计
    )
