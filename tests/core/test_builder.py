# tests/core/test_builder.py
import pytest
from unittest.mock import MagicMock

from gecko.core.builder import AgentBuilder
from gecko.core.engine.base import CognitiveEngine
from gecko.core.exceptions import ConfigurationError
from gecko.core.toolbox import ToolBox
from gecko.plugins.storage.interfaces import SessionInterface
from gecko.plugins.tools.base import BaseTool


# ----------------------------------------------------------------------
# Test helpers
# ----------------------------------------------------------------------

class MockEngine(CognitiveEngine):
    """
    测试专用 Engine：
    - 显式接收 Agent/Builder 会传入的核心依赖（model/toolbox/memory/event_bus）
    - 接收并记录额外 kwargs，便于断言「Builder 的 engine_kwargs 是否正确透传/过滤」
    """
    def __init__(self, model, toolbox, memory, event_bus=None, **kwargs):
        # 不依赖 CognitiveEngine.__init__ 的签名，避免因版本差异导致 TypeError
        self.model = model
        self.toolbox = toolbox
        self.memory = memory
        self.event_bus = event_bus
        self._seen_kwargs = dict(kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    async def step(self, *args, **kwargs):  # type: ignore[override]
        return None


class BareModel:
    """一个“普通对象模型”，用于严格控制 hasattr 行为。"""
    pass


class BareModelWithCompletion:
    """只提供 acompletion，不提供 astream / count_tokens。"""
    def acompletion(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


class _TestTool(BaseTool):
    """
    最小可实例化工具实现，用于测试 Builder 的工具去重逻辑。

    关键点：
    - BaseTool 是抽象类，至少要求实现抽象方法 `_run`（从报错可知）。
    - 我们提供 `_run` 的最小实现以使其可实例化。
    """
    def __init__(self, name: str, description: str | None = None):
        self.name = name
        self.description = description or f"tool {name}"

    # BaseTool 抽象方法：必须实现（从失败栈可知）
    def _run(self, *args, **kwargs):  # type: ignore[override]
        return None

    # 如果 BaseTool 还定义了 run/arun，这里提供兼容实现（不一定是抽象要求，但无害）
    def run(self, *args, **kwargs):  # type: ignore[override]
        return self._run(*args, **kwargs)

    async def arun(self, *args, **kwargs):  # type: ignore[override]
        return self._run(*args, **kwargs)


def make_tool(name: str) -> BaseTool:
    return _TestTool(name)


class StubSettings:
    """
    供 with_settings 注入使用的 settings stub。
    Builder 会读取：
    - memory_max_tokens（推荐）
    - max_context_tokens（兼容）
    - default_session_id（推荐）
    """
    def __init__(self, *, memory_max_tokens=None, max_context_tokens=None, default_session_id=None):
        self.memory_max_tokens = memory_max_tokens
        self.max_context_tokens = max_context_tokens
        self.default_session_id = default_session_id


# ----------------------------------------------------------------------
# Original tests (kept, adapted for newest builder.py)
# ----------------------------------------------------------------------

def test_builder_storage_validation(mock_llm):
    """测试 Storage 接口类型检查"""
    builder = AgentBuilder().with_model(mock_llm)

    class NotStorage:
        pass

    with pytest.raises(TypeError, match="SessionInterface"):
        builder.with_storage(NotStorage())  # type: ignore[arg-type]

    valid_storage = MagicMock(spec=SessionInterface)
    builder.with_storage(valid_storage)
    assert builder._storage is valid_storage


def test_builder_engine_kwargs_passthrough(mock_llm):
    """测试 Engine 参数透传"""
    builder = AgentBuilder().with_model(mock_llm)

    builder.with_engine(
        MockEngine,
        custom_param="value",
        max_iterations=99,
    )

    agent = builder.build()

    assert isinstance(agent.engine, MockEngine)
    assert agent.engine.max_iterations == 99
    assert agent.engine._seen_kwargs["custom_param"] == "value"


def test_builder_system_prompt_handling(mock_llm):
    """测试 system_prompt 写入 engine_kwargs 并透传给 Engine"""
    builder = AgentBuilder().with_model(mock_llm).with_engine(MockEngine)
    builder.with_system_prompt("You are a bot")

    assert builder._engine_kwargs["system_prompt"] == "You are a bot"

    agent = builder.build()
    assert agent.engine._seen_kwargs["system_prompt"] == "You are a bot"


# ----------------------------------------------------------------------
# New tests to reach 100% coverage for newest builder.py
# ----------------------------------------------------------------------

def test_builder_validate_requires_model():
    """validate/build 必须先 with_model"""
    b = AgentBuilder()
    with pytest.raises(ConfigurationError):
        b.validate()
    with pytest.raises(ConfigurationError):
        b.build()


def test_builder_with_model_minimum_method_check():
    """with_model 最低要求：必须有 acompletion（注意避免 MagicMock 的 hasattr 误判）"""
    # 1) 普通对象：确实没有 acompletion，应抛 ConfigurationError
    model = BareModel()
    with pytest.raises(ConfigurationError):
        AgentBuilder().with_model(model)  # type: ignore[arg-type]

    # 2) 具备 acompletion：应通过
    model2 = BareModelWithCompletion()
    b = AgentBuilder().with_model(model2).with_engine(MockEngine)
    b.validate()


def test_builder_require_streaming_and_token_counting_success(mock_llm):
    """覆盖 require_streaming / require_token_counting 正常路径（mock_llm 具备 astream/count_tokens）"""
    b = (
        AgentBuilder()
        .with_model(mock_llm)
        .with_engine(MockEngine)
        .require_streaming(True)
        .require_token_counting(True)
    )
    b.validate()
    agent = b.build()
    assert isinstance(agent.engine, MockEngine)


def test_builder_require_streaming_missing_method():
    """require_streaming=True 时，model 缺少 astream 应报错（避免 MagicMock 的 hasattr 误判）"""
    model = BareModelWithCompletion()
    b = AgentBuilder().with_model(model).with_engine(MockEngine).require_streaming(True)
    with pytest.raises(ConfigurationError, match="astream"):
        b.validate()


def test_builder_require_token_counting_missing_method():
    """require_token_counting=True 时，model 缺少 count_tokens 应报错（避免 MagicMock 的 hasattr 误判）"""
    model = BareModelWithCompletion()
    b = AgentBuilder().with_model(model).with_engine(MockEngine).require_token_counting(True)
    with pytest.raises(ConfigurationError, match="count_tokens"):
        b.validate()


def test_builder_settings_injection_and_defaults_precedence(mock_llm):
    """with_settings 注入 + 默认值优先级：memory_max_tokens > max_context_tokens"""
    s = StubSettings(memory_max_tokens=1234, max_context_tokens=9999, default_session_id="sid-from-settings")
    b = AgentBuilder().with_settings(s).with_model(mock_llm).with_engine(MockEngine)

    agent = b.build()
    assert agent.memory.session_id == "sid-from-settings"
    assert agent.memory.max_tokens == 1234


def test_builder_settings_defaults_fallback_to_max_context_tokens(mock_llm):
    """settings 未提供 memory_max_tokens 时回退到 max_context_tokens"""
    s = StubSettings(memory_max_tokens=None, max_context_tokens=7777, default_session_id="sid2")
    b = AgentBuilder().with_settings(s).with_model(mock_llm).with_engine(MockEngine)

    agent = b.build()
    assert agent.memory.max_tokens == 7777
    assert agent.memory.session_id == "sid2"


def test_builder_settings_defaults_fallback_to_constant(mock_llm):
    """settings 的值非法时回退到 Builder 常量"""
    s = StubSettings(memory_max_tokens=0, max_context_tokens=-1, default_session_id="")
    b = AgentBuilder().with_settings(s).with_model(mock_llm).with_engine(MockEngine)

    agent = b.build()
    assert agent.memory.max_tokens == AgentBuilder._FALLBACK_MAX_TOKENS
    assert agent.memory.session_id == AgentBuilder._FALLBACK_SESSION_ID


def test_builder_disable_settings_defaults_uses_fallback(mock_llm):
    """with_settings_defaults(False) 分支：不读 settings，直接 fallback"""
    s = StubSettings(memory_max_tokens=8888, default_session_id="sid3")
    b = (
        AgentBuilder()
        .with_settings(s)
        .with_settings_defaults(False)
        .with_model(mock_llm)
        .with_engine(MockEngine)
    )
    agent = b.build()
    assert agent.memory.max_tokens == AgentBuilder._FALLBACK_MAX_TOKENS
    assert agent.memory.session_id == AgentBuilder._FALLBACK_SESSION_ID


def test_builder_freeze_defaults_from_settings(mock_llm):
    """with_defaults_from_settings 冻结默认值：settings 变化不影响 build"""
    s = StubSettings(memory_max_tokens=2222, default_session_id="sid-freeze")
    b = AgentBuilder().with_settings(s).with_model(mock_llm).with_engine(MockEngine)
    b.with_defaults_from_settings()

    s.memory_max_tokens = 9999
    s.default_session_id = "changed"

    agent = b.build()
    assert agent.memory.max_tokens == 2222
    assert agent.memory.session_id == "sid-freeze"


def test_builder_build_spec_contains_expected_fields(mock_llm):
    """build_spec 输出结构应稳定且可用于审计/调试"""
    b = (
        AgentBuilder()
        .with_model(mock_llm)
        .with_engine(MockEngine, foo="bar")
        .with_event_bus(object())
        .with_session_id("sid")
        .with_max_tokens(111)
    )

    spec = b.build_spec()
    assert spec["model"]["type"]
    assert spec["engine_cls"] == "MockEngine"
    assert spec["session_id"] == "sid"
    assert spec["max_tokens"] == 111
    assert "event_bus" in spec["agent_kwargs_keys"]
    assert "event_bus" not in spec["engine_kwargs_keys"]


def test_builder_engine_kwargs_filter_strict_raises(mock_llm):
    """engine_kwargs_filter strict=True：validate 阶段直接报错"""
    b = (
        AgentBuilder()
        .with_model(mock_llm)
        .with_engine(MockEngine, a=1, b=2)
        .with_engine_kwargs_filter(allow=["a"], strict=True)
    )
    with pytest.raises(ConfigurationError):
        b.validate()


def test_builder_engine_kwargs_filter_non_strict_filters(mock_llm):
    """engine_kwargs_filter strict=False：build 时静默过滤"""
    b = (
        AgentBuilder()
        .with_model(mock_llm)
        .with_engine(MockEngine, a=1, b=2)
        .with_engine_kwargs_filter(allow=["a"], strict=False)
    )
    agent = b.build()
    assert agent.engine._seen_kwargs.get("a") == 1
    assert "b" not in agent.engine._seen_kwargs


def test_builder_engine_kwargs_filter_deny_filters(mock_llm):
    """deny 黑名单分支：deny 优先生效"""
    b = (
        AgentBuilder()
        .with_model(mock_llm)
        .with_engine(MockEngine, a=1, secret=999)
        .with_engine_kwargs_filter(deny=["secret"], strict=False)
    )
    agent = b.build()
    assert agent.engine._seen_kwargs.get("a") == 1
    assert "secret" not in agent.engine._seen_kwargs


def test_builder_tool_dedup_last_wins(mock_llm):
    """tool_dedup_strategy=last：后者覆盖前者"""
    t1 = make_tool("same")
    t2 = make_tool("same")

    b = (
        AgentBuilder()
        .with_model(mock_llm)
        .with_engine(MockEngine)
        .with_tool_dedup_strategy("last")
        .with_tools([t1, t2])
    )
    comps = b.build_components()
    assert len(comps.toolbox.tools) == 1
    assert comps.toolbox.tools[0] is t2


def test_builder_tool_dedup_first_wins(mock_llm):
    """tool_dedup_strategy=first：保留最先注册"""
    t1 = make_tool("same")
    t2 = make_tool("same")

    b = (
        AgentBuilder()
        .with_model(mock_llm)
        .with_engine(MockEngine)
        .with_tool_dedup_strategy("first")
        .with_tools([t1, t2])
    )
    comps = b.build_components()
    assert len(comps.toolbox.tools) == 1
    assert comps.toolbox.tools[0] is t1


def test_builder_tool_dedup_error_raises(mock_llm):
    """tool_dedup_strategy=error：同名直接报错"""
    t1 = make_tool("same")
    t2 = make_tool("same")

    b = (
        AgentBuilder()
        .with_model(mock_llm)
        .with_engine(MockEngine)
        .with_tool_dedup_strategy("error")
        .with_tools([t1, t2])
    )
    with pytest.raises(ConfigurationError):
        b.validate()


def test_builder_toolbox_factory_is_used(mock_llm):
    """toolbox_factory 分支：应调用注入的工厂"""
    seen = {}

    def toolbox_factory(tools, cfg):
        seen["tools"] = tools
        seen["cfg"] = cfg
        return ToolBox(tools=tools, **cfg)

    b = (
        AgentBuilder()
        .with_model(mock_llm)
        .with_engine(MockEngine)
        .with_toolbox_factory(toolbox_factory)
        .with_toolbox_config(timeout=1)
        .with_tools([make_tool("t1")])
    )

    comps = b.build_components()
    assert "tools" in seen
    assert len(seen["tools"]) == 1
    assert seen["cfg"]["timeout"] == 1
    assert len(comps.toolbox.tools) == 1


def test_builder_memory_factory_is_used(mock_llm):
    """
    memory_factory 分支：应调用注入的工厂

    为避免依赖 TokenMemory 构造签名（不同版本可能不同），
    这里直接返回一个 MagicMock 作为 memory 实例，并断言 agent.memory 就是它。
    """
    seen = {}
    memory_obj = MagicMock()

    def memory_factory(session_id, max_tokens, model_driver, storage):
        seen["session_id"] = session_id
        seen["max_tokens"] = max_tokens
        seen["model_driver"] = model_driver
        seen["storage"] = storage
        return memory_obj  # type: ignore[return-value]

    b = (
        AgentBuilder()
        .with_model(mock_llm)
        .with_engine(MockEngine)
        .with_memory_factory(memory_factory)
        .with_session_id("sid-mf")
        .with_max_tokens(333)
    )

    agent = b.build()
    assert seen["session_id"] == "sid-mf"
    assert seen["max_tokens"] == 333
    assert seen["model_driver"] is mock_llm
    assert agent.memory is memory_obj


def test_builder_event_bus_not_passed_to_engine(mock_llm, event_bus):
    """event_bus 只应传给 Agent，不应出现在 Engine kwargs 中"""
    b = (
        AgentBuilder()
        .with_model(mock_llm)
        .with_engine(MockEngine, x=1)
        .with_event_bus(event_bus)
    )

    agent = b.build()
    assert agent.event_bus is event_bus
    assert "event_bus" not in agent.engine._seen_kwargs
    assert agent.engine._seen_kwargs["x"] == 1


def test_builder_on_build_hook_called(mock_llm):
    """on_build hook 应在 before/after 两个阶段被调用"""
    calls = []

    def hook(payload):
        calls.append(payload)

    b = (
        AgentBuilder()
        .with_model(mock_llm)
        .with_engine(MockEngine)
        .with_on_build(hook)
    )
    agent = b.build()

    assert len(calls) == 2
    assert calls[0]["phase"] == "before"
    assert calls[1]["phase"] == "after"
    assert "spec" in calls[0]
    assert calls[1]["agent"] is agent
    assert "components" in calls[1]


def test_builder_clone_and_reset_and_clear_tools(mock_llm):
    """覆盖 clone/reset/clear_tools 分支"""
    b = (
        AgentBuilder()
        .with_model(mock_llm)
        .with_engine(MockEngine, a=1)
        .with_tools([make_tool("t1"), make_tool("t2")])
    )

    b2 = b.clone()
    assert b2 is not b
    assert b2._engine_kwargs["a"] == 1
    assert len(b2._tools) == 2

    b2.clear_tools()
    assert len(b2._tools) == 0

    b2.reset()
    with pytest.raises(ConfigurationError):
        b2.validate()