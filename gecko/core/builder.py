# gecko/core/builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, Literal

from gecko.config import get_settings
from gecko.core.agent import Agent
from gecko.core.engine.base import CognitiveEngine
from gecko.core.engine.react import ReActEngine
from gecko.core.exceptions import ConfigurationError
from gecko.core.memory import TokenMemory
from gecko.core.toolbox import ToolBox
from gecko.plugins.storage.interfaces import SessionInterface
from gecko.plugins.tools.base import BaseTool


# ----------------------------------------------------------------------
# 类型别名：工厂注入点 / Hook
# ----------------------------------------------------------------------

# ToolBox 工厂：输入 “工具列表 + toolbox_config”，返回一个 ToolBox 实例
ToolBoxFactory = Callable[[list[BaseTool], dict[str, Any]], ToolBox]

# TokenMemory 工厂：输入 “session_id / max_tokens / model_driver / storage”，返回一个 TokenMemory 实例
MemoryFactory = Callable[[str, int, Any, SessionInterface | None], TokenMemory]

# build hook：用于观测 build 前后发生了什么（同步回调，不引入 async 复杂性）
OnBuildHook = Callable[[dict[str, Any]], None]

# 工具去重策略：
# - last : 后者覆盖前者（默认，兼容旧行为）
# - first: 保留最先注册的工具
# - error: 同名直接报错（更严格，适合生产）
ToolDedupStrategy = Literal["last", "first", "error"]


@dataclass(frozen=True)
class AgentComponents:
    """
    build_components() 的返回值：
    - toolbox: 工具箱（已按策略去重）
    - memory : 记忆模块（已注入 model_driver/storage）
    """
    toolbox: ToolBox
    memory: TokenMemory


@dataclass(frozen=True)
class EngineKwargsFilter:
    """
    Engine 参数过滤策略（可选）。

    目的：
    - 避免把不该透传给 Engine 的参数传进去导致难排错；
    - 可配合 strict=True 在 CI/生产中做“防呆”。

    字段：
    - allow: 白名单；如果不为 None，则只允许 allow 中的 key
    - deny : 黑名单；deny 中的 key 无论如何都禁止（优先级更高）
    - strict:
        - True  : validate/build 阶段遇到不允许的 key 直接抛 ConfigurationError
        - False : build 阶段静默过滤（丢弃）不允许的 key
    """
    allow: set[str] | None = None
    deny: set[str] | None = None
    strict: bool = False

    def apply(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        应用过滤策略并返回一个新 dict（不修改入参）。
        """
        filtered = dict(kwargs)

        # 1) 先应用 deny（黑名单优先）
        if self.deny:
            for k in list(filtered.keys()):
                if k in self.deny:
                    filtered.pop(k, None)

        # 2) 再应用 allow（白名单）
        if self.allow is not None:
            for k in list(filtered.keys()):
                if k not in self.allow:
                    filtered.pop(k, None)

        return filtered

    def validate(self, kwargs: Mapping[str, Any]) -> None:
        """
        严格模式校验：
        - 若 strict=False，不做任何校验；
        - 若 strict=True，发现不符合 allow/deny 的 key 直接报错。
        """
        if not self.strict:
            return

        keys = set(kwargs.keys())

        if self.deny:
            forbidden = keys & self.deny
            if forbidden:
                raise ConfigurationError(
                    "engine_kwargs 包含被禁止的参数",
                    context={"forbidden": sorted(forbidden)},
                )

        if self.allow is not None:
            not_allowed = keys - self.allow
            if not_allowed:
                raise ConfigurationError(
                    "engine_kwargs 包含未被允许的参数",
                    context={"not_allowed": sorted(not_allowed)},
                )


class AgentBuilder:
    """
    Agent 构建器（工程化增强版）

    主要用途：
    - 用链式 API 配置 Agent 的依赖（model/tools/storage/memory/engine）；
    - build 时统一做校验、去重、依赖注入；
    - 对外隐藏 ToolBox/TokenMemory/Engine/Agent 的装配细节；
    - 支持注入 settings、可选 build hook、可选参数过滤策略与工厂注入。

    默认行为（兼容旧版）：
    - 工具同名：后者覆盖前者（last-wins）
    - engine_kwargs：不做过滤，直接透传给 Engine
    - memory 默认值：
        - 若用户未显式设置 max_tokens/session_id，且允许使用 settings 默认值，
          则从 settings 读取；否则使用 fallback 常量。
    """

    # fallback：当没有显式设置、settings 也取不到时使用
    _FALLBACK_SESSION_ID = "default"
    _FALLBACK_MAX_TOKENS = 4000

    def __init__(self) -> None:
        # settings 注入点：
        # - 若注入了 settings，则 Builder 不再依赖全局 get_settings()
        # - 若不注入，则回退到 get_settings()（兼容原逻辑）
        self._settings: Any | None = None

        # model 是必须项（build/validate 会检查）
        self._model: Any | None = None

        # tools/toolbox 配置
        self._tools: list[BaseTool] = []
        self._toolbox_config: dict[str, Any] = {}
        self._tool_dedup_strategy: ToolDedupStrategy = "last"

        # memory 配置
        self._storage: SessionInterface | None = None
        self._session_id: str | None = None
        self._max_tokens: int | None = None

        # engine 配置
        self._engine_cls: type[CognitiveEngine] = ReActEngine
        # 注意：engine_kwargs 中可能混入 event_bus（属于 Agent 参数），build 时会拆出去
        self._engine_kwargs: dict[str, Any] = {}
        self._engine_kwargs_filter: EngineKwargsFilter | None = None

        # 默认值策略开关
        self._use_settings_defaults: bool = True

        # 工厂注入点：便于测试/替换实现/注入监控 wrapper
        self._toolbox_factory: ToolBoxFactory | None = None
        self._memory_factory: MemoryFactory | None = None

        # 可选能力要求：
        # - 默认 False，保持兼容（只要求 acompletion）
        self._require_streaming: bool = False
        self._require_token_counting: bool = False

        # build hook：可选观测 build 前后
        self._on_build: OnBuildHook | None = None

    # ----------------------------------------------------------------------
    # settings 注入
    # ----------------------------------------------------------------------
    def with_settings(self, settings_obj: Any) -> "AgentBuilder":
        """
        注入 settings 实例（推荐）。

        settings_obj 可以是：
        - GeckoSettings 实例
        - 测试 stub（只要有需要字段即可）

        Builder 读取的字段包括：
        - memory_max_tokens（推荐）
        - max_context_tokens（兼容）
        - default_session_id（推荐）
        """
        self._settings = settings_obj
        return self

    def _get_settings(self) -> Any:
        """
        内部统一获取 settings：
        - 若注入了 settings，优先使用；
        - 否则使用全局 get_settings()（兼容旧逻辑）。
        """
        return self._settings or get_settings()

    # ----------------------------------------------------------------------
    # Builder 复用能力
    # ----------------------------------------------------------------------
    def reset(self) -> "AgentBuilder":
        """重置 Builder 状态（便于复用）。"""
        self.__init__()
        return self

    def clone(self) -> "AgentBuilder":
        """
        克隆 Builder（配置隔离）：
        - 新 Builder 与旧 Builder 共享同一个 settings 引用（如果你希望隔离，可注入不同 settings）
        - 其余 dict/list 做拷贝，避免互相污染。
        """
        b = AgentBuilder()
        b._settings = self._settings

        b._model = self._model
        b._tools = list(self._tools)
        b._toolbox_config = dict(self._toolbox_config)
        b._tool_dedup_strategy = self._tool_dedup_strategy

        b._storage = self._storage
        b._session_id = self._session_id
        b._max_tokens = self._max_tokens

        b._engine_cls = self._engine_cls
        b._engine_kwargs = dict(self._engine_kwargs)
        b._engine_kwargs_filter = self._engine_kwargs_filter

        b._use_settings_defaults = self._use_settings_defaults

        b._toolbox_factory = self._toolbox_factory
        b._memory_factory = self._memory_factory

        b._require_streaming = self._require_streaming
        b._require_token_counting = self._require_token_counting

        b._on_build = self._on_build
        return b

    # ----------------------------------------------------------------------
    # 可观测性 hook
    # ----------------------------------------------------------------------
    def with_on_build(self, hook: OnBuildHook | None) -> "AgentBuilder":
        """
        注册 build hook（同步回调）。

        hook 接收到的 dict（示例）：
        - {"phase": "before", "spec": ...}
        - {"phase": "after", "spec": ..., "components": ..., "agent": ...}

        说明：
        - 这里刻意设计为同步函数，避免把 build 变为 async；
        - 如 hook 需要做 IO，可自行把数据扔到队列或后台任务。
        """
        self._on_build = hook
        return self

    # ----------------------------------------------------------------------
    # 默认值策略
    # ----------------------------------------------------------------------
    def with_settings_defaults(self, enabled: bool = True) -> "AgentBuilder":
        """控制是否从 settings 读取默认值。"""
        self._use_settings_defaults = bool(enabled)
        return self

    def with_defaults_from_settings(self) -> "AgentBuilder":
        """
        冻结当前 settings 的默认值到 Builder。

        适合：
        - 希望构建行为可复现（避免运行中 settings 改变导致 build 行为漂移）。
        """
        s = self._get_settings()
        if self._max_tokens is None:
            self._max_tokens = self._settings_memory_max_tokens(s)
        if self._session_id is None:
            self._session_id = self._settings_default_session_id(s)
        return self

    # ----------------------------------------------------------------------
    # 工厂注入
    # ----------------------------------------------------------------------
    def with_toolbox_factory(self, factory: ToolBoxFactory) -> "AgentBuilder":
        """注入 ToolBox 工厂。"""
        if not callable(factory):
            raise TypeError("toolbox_factory 必须可调用")
        self._toolbox_factory = factory
        return self

    def with_memory_factory(self, factory: MemoryFactory) -> "AgentBuilder":
        """注入 Memory 工厂。"""
        if not callable(factory):
            raise TypeError("memory_factory 必须可调用")
        self._memory_factory = factory
        return self

    # ----------------------------------------------------------------------
    # 可选能力要求
    # ----------------------------------------------------------------------
    def require_streaming(self, enabled: bool = True) -> "AgentBuilder":
        """若启用，则 model 必须提供 astream。"""
        self._require_streaming = bool(enabled)
        return self

    def require_token_counting(self, enabled: bool = True) -> "AgentBuilder":
        """若启用，则 model 必须提供 count_tokens。"""
        self._require_token_counting = bool(enabled)
        return self

    # ----------------------------------------------------------------------
    # 引擎参数过滤策略
    # ----------------------------------------------------------------------
    def with_engine_kwargs_filter(
        self,
        *,
        allow: Sequence[str] | None = None,
        deny: Sequence[str] | None = None,
        strict: bool = False,
    ) -> "AgentBuilder":
        """
        设置 engine_kwargs 的过滤策略：
        - 默认不设置 => 不过滤（兼容旧行为）
        """
        allow_set = set(allow) if allow is not None else None
        deny_set = set(deny) if deny is not None else None
        self._engine_kwargs_filter = EngineKwargsFilter(
            allow=allow_set,
            deny=deny_set,
            strict=bool(strict),
        )
        return self

    # ----------------------------------------------------------------------
    # 工具去重策略
    # ----------------------------------------------------------------------
    def with_tool_dedup_strategy(self, strategy: ToolDedupStrategy) -> "AgentBuilder":
        """设置工具同名策略：last/first/error。"""
        if strategy not in ("last", "first", "error"):
            raise ValueError(f"Unknown tool dedup strategy: {strategy}")
        self._tool_dedup_strategy = strategy
        return self

    # ----------------------------------------------------------------------
    # 基础配置
    # ----------------------------------------------------------------------
    def with_model(self, model: Any) -> "AgentBuilder":
        """
        指定模型驱动（必选）。

        最低要求：
        - model 必须至少有 acompletion 方法（异步补全）
        """
        missing = [m for m in ("acompletion",) if not hasattr(model, m)]
        if missing:
            raise ConfigurationError(
                f"Model 缺少必要方法: {', '.join(missing)}",
                context={"model": repr(model)},
            )
        self._model = model
        return self

    def with_tool(self, tool: BaseTool) -> "AgentBuilder":
        """添加单个工具。"""
        if not isinstance(tool, BaseTool):
            raise TypeError(f"Tool 必须继承 BaseTool，收到 {type(tool)}")
        self._tools.append(tool)
        return self

    def with_tools(self, tools: Sequence[BaseTool]) -> "AgentBuilder":
        """批量添加工具。"""
        for tool in tools:
            self.with_tool(tool)
        return self

    def clear_tools(self) -> "AgentBuilder":
        """清空工具列表。"""
        self._tools.clear()
        return self

    def with_storage(self, storage: SessionInterface | None) -> "AgentBuilder":
        """
        注入存储后端（可选），用于 TokenMemory 持久化。

        注意：
        - 这里使用 isinstance(storage, SessionInterface) 进行校验。
          若 SessionInterface 是 typing.Protocol，需要保证其是 @runtime_checkable，
          否则 isinstance 会抛 TypeError。当前项目应确保这一点成立。
        """
        if storage is not None and not isinstance(storage, SessionInterface):
            raise TypeError("storage 必须实现 SessionInterface，用于 TokenMemory 持久化")
        self._storage = storage
        return self

    def with_session_id(self, session_id: str) -> "AgentBuilder":
        """设置 session_id（增强：非空校验）。"""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id 必须是非空字符串")
        self._session_id = session_id
        return self

    def with_max_tokens(self, max_tokens: int) -> "AgentBuilder":
        """设置 max_tokens（增强：正整数校验）。"""
        if not isinstance(max_tokens, int):
            raise TypeError("max_tokens 必须是 int")
        if max_tokens <= 0:
            raise ValueError("max_tokens 必须为正数")
        self._max_tokens = max_tokens
        return self

    def with_engine(self, engine_cls: type[CognitiveEngine], **engine_kwargs: Any) -> "AgentBuilder":
        """
        指定 Engine 类与其参数（透传给 Engine）。

        engine_kwargs 注意点：
        - 可以包含 system_prompt/name/max_turns 等；
        - 也可以暂存 event_bus（但它会在 build 时拆出来传给 Agent，而不是 Engine）。
        """
        if not issubclass(engine_cls, CognitiveEngine):
            raise TypeError("engine_cls 必须继承 CognitiveEngine")
        self._engine_cls = engine_cls
        self._engine_kwargs.update(dict(engine_kwargs))
        return self

    def with_system_prompt(self, prompt: str) -> "AgentBuilder":
        """设置 system_prompt（统一放入 engine_kwargs）。"""
        if not isinstance(prompt, str):
            raise TypeError("system_prompt 必须是 str")
        self._engine_kwargs["system_prompt"] = prompt
        return self

    def with_event_bus(self, event_bus: Any) -> "AgentBuilder":
        """
        设置 EventBus。

        注意：
        - event_bus 并不是 Engine 的参数，而是 Agent 的参数；
        - 这里暂存在 engine_kwargs 中，build 时会拆分出来。
        """
        self._engine_kwargs["event_bus"] = event_bus
        return self

    def with_toolbox_config(self, **config: Any) -> "AgentBuilder":
        """设置 ToolBox 构造参数（并发、超时等）。"""
        self._toolbox_config.update(dict(config))
        return self

    def with_toolbox_config_mapping(self, config: Mapping[str, Any]) -> "AgentBuilder":
        """以 Mapping 的形式设置 ToolBox config。"""
        self._toolbox_config.update(dict(config))
        return self

    # ----------------------------------------------------------------------
    # 调试/审计：构建规格
    # ----------------------------------------------------------------------
    def build_spec(self) -> dict[str, Any]:
        """
        返回一个尽量可 JSON 序列化的构建规格。

        设计原则：
        - 不直接输出 engine_kwargs 的值（可能包含不可序列化对象）；
        - 只输出 key 列表 + 关键的配置摘要；
        - model 只输出 type 与 repr，不输出真实对象。
        """
        model_info = None
        if self._model is not None:
            model_info = {
                "type": type(self._model).__name__,
                "repr": repr(self._model),
                "has_astream": hasattr(self._model, "astream"),
                "has_count_tokens": hasattr(self._model, "count_tokens"),
            }

        # 解析 memory 的最终默认值（包含 settings/fallback）
        session_id, max_tokens = self._resolve_memory_defaults()

        tool_names = [getattr(t, "name", "<unknown>") for t in self._tools]

        # 将 event_bus 这类“Agent 参数”从 engine_kwargs 中拆出来
        agent_kwargs, engine_kwargs = self._split_agent_kwargs(self._engine_kwargs)

        return {
            "model": model_info,
            "engine_cls": self._engine_cls.__name__,
            "engine_kwargs_keys": sorted(engine_kwargs.keys()),
            "agent_kwargs_keys": sorted(agent_kwargs.keys()),
            "tool_count": len(self._tools),
            "tool_names": tool_names,
            "tool_dedup_strategy": self._tool_dedup_strategy,
            "session_id": session_id,
            "max_tokens": max_tokens,
            "use_settings_defaults": self._use_settings_defaults,
            "has_storage": self._storage is not None,
            "require_streaming": self._require_streaming,
            "require_token_counting": self._require_token_counting,
        }

    # ----------------------------------------------------------------------
    # 校验与构建
    # ----------------------------------------------------------------------
    def validate(self) -> None:
        """
        只做校验，不构建实例。

        作用：
        - 提前发现配置错误；
        - 便于 CI / 启动时预检。
        """
        if self._model is None:
            raise ConfigurationError("构建 Agent 前必须调用 with_model 指定模型")

        # 可选：强制模型支持流式 / token 统计
        if self._require_streaming and not hasattr(self._model, "astream"):
            raise ConfigurationError("require_streaming=True 但 model 缺少 astream 方法")
        if self._require_token_counting and not hasattr(self._model, "count_tokens"):
            raise ConfigurationError("require_token_counting=True 但 model 缺少 count_tokens 方法")

        # engine_cls 必须是 CognitiveEngine 子类
        if not issubclass(self._engine_cls, CognitiveEngine):
            raise ConfigurationError(
                "engine_cls 必须继承 CognitiveEngine",
                context={"engine_cls": repr(self._engine_cls)},
            )

        # toolbox_config / engine_kwargs 的 key 类型必须为 str（避免序列化/渲染异常）
        for k in self._toolbox_config.keys():
            if not isinstance(k, str):
                raise ConfigurationError("toolbox_config 的 key 必须为 str", context={"key": repr(k)})
        for k in self._engine_kwargs.keys():
            if not isinstance(k, str):
                raise ConfigurationError("engine_kwargs 的 key 必须为 str", context={"key": repr(k)})

        # tools 校验：类型 + name + 同名策略
        seen: set[str] = set()
        for t in self._tools:
            if not isinstance(t, BaseTool):
                raise ConfigurationError("Tool 必须继承 BaseTool", context={"tool": repr(t)})

            name = getattr(t, "name", None)
            if not isinstance(name, str) or not name.strip():
                raise ConfigurationError("Tool.name 必须是非空字符串", context={"tool": repr(t)})

            if self._tool_dedup_strategy == "error":
                if name in seen:
                    raise ConfigurationError("发现同名 Tool 且策略为 error", context={"tool_name": name})
                seen.add(name)

        # memory 参数校验（包含默认值解析）
        session_id, max_tokens = self._resolve_memory_defaults()
        if not isinstance(session_id, str) or not session_id.strip():
            raise ConfigurationError("session_id 必须是非空字符串", context={"session_id": session_id})
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ConfigurationError("max_tokens 必须为正整数", context={"max_tokens": max_tokens})

        # storage 校验
        if self._storage is not None and not isinstance(self._storage, SessionInterface):
            raise ConfigurationError("storage 必须实现 SessionInterface", context={"storage": repr(self._storage)})

        # engine_kwargs 过滤策略（严格模式校验）
        if self._engine_kwargs_filter is not None:
            _, engine_kwargs = self._split_agent_kwargs(self._engine_kwargs)
            self._engine_kwargs_filter.validate(engine_kwargs)

    def build(self) -> Agent:
        """
        构建 Agent 实例。

        构建流程：
        1) validate()
        2) 可选 before hook（带 build_spec）
        3) build toolbox/memory
        4) 拆分 agent_kwargs / engine_kwargs
        5) 可选过滤 engine_kwargs
        6) 创建 Agent
        7) 可选 after hook（带 spec/components/agent）
        """
        self.validate()

        spec = self.build_spec()
        if self._on_build is not None:
            self._on_build({"phase": "before", "spec": spec})

        toolbox = self._build_toolbox()
        memory = self._build_memory()

        # 拆分出 Agent 参数（如 event_bus），剩余为 Engine 参数
        agent_kwargs, engine_kwargs = self._split_agent_kwargs(self._engine_kwargs)

        # 引擎参数过滤：默认不启用
        if self._engine_kwargs_filter is not None:
            engine_kwargs = self._engine_kwargs_filter.apply(engine_kwargs)

        agent = Agent(
            model=self._model,  # validate 已保证非 None
            toolbox=toolbox,
            memory=memory,
            engine_cls=self._engine_cls,
            event_bus=agent_kwargs.get("event_bus"),
            **engine_kwargs,
        )

        if self._on_build is not None:
            self._on_build(
                {
                    "phase": "after",
                    "spec": spec,
                    "components": AgentComponents(toolbox=toolbox, memory=memory),
                    "agent": agent,
                }
            )

        return agent

    def build_components(self) -> AgentComponents:
        """
        只构建 ToolBox + TokenMemory，不创建 Agent。

        适用场景：
        - 单元测试只想验证 toolbox/memory 构建结果；
        - 或者上层容器想自己接管 Agent 创建流程。
        """
        self.validate()
        return AgentComponents(toolbox=self._build_toolbox(), memory=self._build_memory())

    # ----------------------------------------------------------------------
    # 内部：默认值解析 / 参数拆分 / 组件构建
    # ----------------------------------------------------------------------
    @staticmethod
    def _split_agent_kwargs(all_kwargs: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        将 all_kwargs（可能混杂 Agent 参数）拆分为两部分：
        - agent_kwargs : 仅包含 Agent.__init__ 需要的参数（当前只有 event_bus）
        - engine_kwargs: 剩余参数，将透传给 Engine

        这样做的原因：
        - Engine 不一定支持 event_bus 参数；
        - Agent 支持 event_bus，并且需要把它传进 engine 内部。
        """
        engine_kwargs = dict(all_kwargs)
        agent_kwargs: dict[str, Any] = {}

        if "event_bus" in engine_kwargs:
            agent_kwargs["event_bus"] = engine_kwargs.pop("event_bus")

        return agent_kwargs, engine_kwargs

    @staticmethod
    def _settings_default_session_id(settings_obj: Any) -> str:
        """
        从 settings 读取默认 session_id（推荐字段：default_session_id）。

        若字段不存在或非法，回退到 fallback。
        """
        v = getattr(settings_obj, "default_session_id", None)
        if isinstance(v, str) and v.strip():
            return v
        return AgentBuilder._FALLBACK_SESSION_ID

    @staticmethod
    def _settings_memory_max_tokens(settings_obj: Any) -> int:
        """
        从 settings 读取 Memory 的 max_tokens 默认值。

        优先级（语义清晰）：
        1) settings.memory_max_tokens（推荐字段）
        2) settings.max_context_tokens（兼容字段）
        3) fallback 常量
        """
        v = getattr(settings_obj, "memory_max_tokens", None)
        if isinstance(v, int) and v > 0:
            return v

        v2 = getattr(settings_obj, "max_context_tokens", None)
        if isinstance(v2, int) and v2 > 0:
            return v2

        return AgentBuilder._FALLBACK_MAX_TOKENS

    def _resolve_memory_defaults(self) -> tuple[str, int]:
        """
        解析最终的 session_id/max_tokens。

        规则：
        - 若用户显式设置，则优先使用；
        - 若 use_settings_defaults=True，则从 settings 中补齐缺省；
        - 最终兜底到 fallback 常量。
        """
        session_id = self._session_id
        max_tokens = self._max_tokens

        if self._use_settings_defaults:
            s = self._get_settings()
            if max_tokens is None:
                max_tokens = self._settings_memory_max_tokens(s)
            if session_id is None:
                session_id = self._settings_default_session_id(s)

        if session_id is None:
            session_id = self._FALLBACK_SESSION_ID
        if max_tokens is None:
            max_tokens = self._FALLBACK_MAX_TOKENS

        return session_id, max_tokens

    def _build_toolbox(self) -> ToolBox:
        """
        构建 ToolBox，并按 name 去重。

        - last : 后者覆盖前者
        - first: 保留第一个
        - error: validate 已确保无重复，这里只负责组装
        """
        deduped: dict[str, BaseTool] = {}

        if self._tool_dedup_strategy == "last":
            for t in self._tools:
                deduped[t.name] = t
        elif self._tool_dedup_strategy == "first":
            for t in self._tools:
                deduped.setdefault(t.name, t)
        else:
            for t in self._tools:
                deduped[t.name] = t

        tools = list(deduped.values())
        cfg = dict(self._toolbox_config)

        # 若注入了 factory，则由 factory 决定如何构建 ToolBox
        if self._toolbox_factory is not None:
            return self._toolbox_factory(tools, cfg)

        return ToolBox(tools=tools, **cfg) # type: ignore

    def _build_memory(self) -> TokenMemory:
        """
        构建 TokenMemory（注入 model_driver/storage）。

        关键点：
        - model_driver=self._model：让 Memory 可使用模型本身的 token 统计能力；
        - storage=self._storage：决定是否启用持久化。
        """
        session_id, max_tokens = self._resolve_memory_defaults()

        if self._memory_factory is not None:
            return self._memory_factory(session_id, max_tokens, self._model, self._storage)

        return TokenMemory(
            session_id=session_id,
            storage=self._storage,
            max_tokens=max_tokens,
            model_driver=self._model,
        )