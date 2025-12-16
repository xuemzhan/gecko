# gecko/core/engine/base.py
"""
认知引擎基类模块（Production Grade / 生产级）

本模块定义了 Agent 推理引擎的抽象基类和相关工具类。
所有具体引擎实现（ReAct、Chain、Tree 等）都应继承 CognitiveEngine 基类。

✅ 本次版本在“不破坏原有对外接口”的前提下，修复并增强了以下工业级问题：
------------------------------------------------------------
1) [P0] 移除 import 时的磁盘 IO 副作用：模型定价表改为“懒加载 + 缓存”
   - 避免库被 import 就读 ~/.gecko/pricing.json 或环境变量文件
   - 支持运行时热更新（可手动 clear cache）

2) [P0] 统计模块不再“锁抢不到就丢数据”
   - ExecutionStats 改为“短临界区 + 必达记录”
   - 增加 add_tokens / add_error 等原子化更新方法，禁止裸写 stats 字段

3) [P1] 事件发布的 awaitable 处理更稳健
   - 不仅判断 coroutinefunction，也支持“同步函数返回 awaitable”的情况

4) [P1] _safe_execute 不再隐式重复统计
   - 默认不自动记步，避免子类 record_step() 与 _safe_execute() 双重统计
   - 仍保留计时能力（由调用者决定是否记录统计）

设计目标：
- 高效：避免 O(n) 频繁计算与 import-time IO
- 稳定：统计不丢、错误语义清晰、事件发布不影响主流程
- 可扩展：定价表可外部注入、缓存可清理、Hook 可控 fail-fast
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import time
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Type, TypeVar, MutableMapping, Iterator

from pydantic import BaseModel, PrivateAttr

from gecko.core.events.bus import EventBus
from gecko.core.events.types import AgentStreamEvent
from gecko.core.exceptions import AgentError, ModelError
from gecko.core.logging import get_logger
from gecko.core.memory import TokenMemory
from gecko.core.message import Message
from gecko.core.output import AgentOutput
from gecko.core.protocols import ModelProtocol, supports_streaming, validate_model
from gecko.core.toolbox import ToolBox

logger = get_logger(__name__)

# 泛型类型变量，用于结构化输出
T = TypeVar("T", bound=BaseModel)


# =============================================================================
# 模型定价配置（懒加载 + 可缓存）
# =============================================================================

def _load_default_pricing() -> Dict[str, Dict[str, float]]:
    """
    加载内置默认的模型定价配置（USD per 1M tokens）

    注意：
    - 这是“内置兜底配置”
    - 实际生产环境建议通过外部文件/配置中心覆盖，以便随厂商调价更新

    返回：
        Dict[model_name, {"input": float, "output": float}]
    """
    return {
        # OpenAI 系列
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4o": {"input": 5.0, "output": 15.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
        # Anthropic 系列
        "claude-3-opus": {"input": 15.0, "output": 75.0},
        "claude-3-sonnet": {"input": 3.0, "output": 15.0},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
        "claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
    }


def load_model_pricing() -> Dict[str, Dict[str, float]]:
    """
    加载模型定价配置（同步函数）

    加载优先级：
    1) 环境变量 GECKO_PRICING_FILE 指定的文件
    2) 用户目录 ~/.gecko/pricing.json
    3) 内置默认配置

    外部配置文件格式示例：
    {
      "gpt-4": {"input": 30.0, "output": 60.0},
      "custom-model": {"input": 1.0, "output": 2.0}
    }

    ⚠️ 注意：本函数不应在 import 时自动调用（避免 import-time IO）
    """
    pricing = _load_default_pricing()

    # 1) 环境变量指定路径
    custom_path = os.environ.get("GECKO_PRICING_FILE")
    if custom_path:
        config_file = Path(custom_path)
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    custom_pricing = json.load(f)
                if isinstance(custom_pricing, dict):
                    pricing.update(custom_pricing)  # 允许覆盖/新增
                    logger.debug("已加载自定义定价配置", path=str(config_file))
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("加载定价配置文件失败", error=str(e), path=str(config_file))

    # 2) 用户目录
    user_config = Path.home() / ".gecko" / "pricing.json"
    if user_config.exists():
        try:
            with open(user_config, "r", encoding="utf-8") as f:
                custom_pricing = json.load(f)
            if isinstance(custom_pricing, dict):
                pricing.update(custom_pricing)
                logger.debug("已加载用户定价配置", path=str(user_config))
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("加载用户定价配置失败", error=str(e), path=str(user_config))

    return pricing


# ---- 懒加载缓存（线程安全）----

_PRICING_CACHE: Optional[Dict[str, Dict[str, float]]] = None
_PRICING_LOCK = threading.Lock()


def get_model_pricing(force_reload: bool = False) -> Dict[str, Dict[str, float]]:
    """
    获取模型定价表（懒加载 + 缓存）

    参数：
        force_reload: True 表示强制重新加载（用于配置热更新/调试）

    返回：
        合并后的定价表 dict
    """
    global _PRICING_CACHE
    if not force_reload and _PRICING_CACHE is not None:
        return _PRICING_CACHE

    with _PRICING_LOCK:
        if not force_reload and _PRICING_CACHE is not None:
            return _PRICING_CACHE
        _PRICING_CACHE = load_model_pricing()
        return _PRICING_CACHE


class _LazyPricingDict(MutableMapping[str, Dict[str, float]]):
    """
    兼容层：保持对外仍然暴露 MODEL_PRICING 变量

    目的：
    - 不破坏外部代码：from gecko.core.engine.base import MODEL_PRICING
    - 但内部实现改为“首次访问才加载”，避免 import-time IO
    """

    def __init__(self) -> None:
        self._override: Optional[Dict[str, Dict[str, float]]] = None

    def set_override(self, pricing: Dict[str, Dict[str, float]]) -> None:
        """允许外部直接注入定价表（测试/私有部署场景常用）"""
        self._override = pricing

    def _data(self) -> Dict[str, Dict[str, float]]:
        return self._override if self._override is not None else get_model_pricing()

    # ---- MutableMapping 接口实现 ----
    def __getitem__(self, key: str) -> Dict[str, float]:
        return self._data()[key]

    def __setitem__(self, key: str, value: Dict[str, float]) -> None:
        d = self._data()
        d[key] = value

    def __delitem__(self, key: str) -> None:
        d = self._data()
        del d[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data())

    def __len__(self) -> int:
        return len(self._data())

    def __repr__(self) -> str:
        return f"LazyPricingDict(len={len(self)})"


# 对外导出的“变量名”保持不变
MODEL_PRICING: MutableMapping[str, Dict[str, float]] = _LazyPricingDict()


# =============================================================================
# 执行统计类（线程安全 + 不丢数据）
# =============================================================================

class ExecutionStats(BaseModel):
    """
    引擎执行统计（工业级）

    设计点：
    - 统计用于监控/计费/调试，不能“锁抢不到就丢”
    - 临界区尽量短：仅做简单整数/浮点累加
    - 提供原子化方法（add_tokens/add_error），禁止外部裸写字段
    """

    total_steps: int = 0
    total_time: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    errors: int = 0
    estimated_cost: float = 0.0

    _lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)

    # ---- 原子化更新方法 ----

    def add_step(
        self,
        duration: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        had_error: bool = False,
    ) -> None:
        """
        记录一次步骤执行（必达记录，不丢数据）

        参数：
            duration: 执行耗时（秒）
            input_tokens/output_tokens: 本次 token
            had_error: 本次是否发生错误
        """
        # 统计属于“轻量关键区”，锁竞争概率很低；即使竞争也不应丢数据
        with self._lock:
            self.total_steps += 1
            self.total_time += float(duration)
            if input_tokens:
                self.input_tokens += int(input_tokens)
            if output_tokens:
                self.output_tokens += int(output_tokens)
            if had_error:
                self.errors += 1

    def add_tokens(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """仅累计 token（适用于流式场景在最后一次性回填）"""
        if not input_tokens and not output_tokens:
            return
        with self._lock:
            if input_tokens:
                self.input_tokens += int(input_tokens)
            if output_tokens:
                self.output_tokens += int(output_tokens)

    def add_tool_call(self, n: int = 1) -> None:
        """记录工具调用次数（支持批量增加）"""
        if n <= 0:
            return
        with self._lock:
            self.tool_calls += int(n)

    def add_error(self, n: int = 1) -> None:
        """记录错误次数（支持批量增加）"""
        if n <= 0:
            return
        with self._lock:
            self.errors += int(n)

    def add_cost(self, cost: float) -> None:
        """累计成本估算（美元）"""
        if not cost:
            return
        with self._lock:
            self.estimated_cost += float(cost)

    # ---- 读接口（无需锁也可，但这里为了更一致，仍做快照式读取）----

    def get_avg_step_time(self) -> float:
        with self._lock:
            return self.total_time / self.total_steps if self.total_steps > 0 else 0.0

    def get_total_tokens(self) -> int:
        with self._lock:
            return self.input_tokens + self.output_tokens

    def get_error_rate(self) -> float:
        with self._lock:
            return self.errors / self.total_steps if self.total_steps > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（便于日志/监控上报）"""
        with self._lock:
            total_steps = self.total_steps
            total_time = self.total_time
            input_tokens = self.input_tokens
            output_tokens = self.output_tokens
            tool_calls = self.tool_calls
            errors = self.errors
            estimated_cost = self.estimated_cost

        avg_step_time = total_time / total_steps if total_steps > 0 else 0.0
        error_rate = errors / total_steps if total_steps > 0 else 0.0
        total_tokens = input_tokens + output_tokens

        return {
            "total_steps": total_steps,
            "total_time": round(total_time, 3),
            "avg_step_time": round(avg_step_time, 3),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "tool_calls": tool_calls,
            "errors": errors,
            "error_rate": round(error_rate, 4),
            "estimated_cost": round(estimated_cost, 6),
        }


# =============================================================================
# 认知引擎基类
# =============================================================================

class CognitiveEngine(ABC):
    """
    认知引擎抽象基类

    - step(): 必须实现
    - step_stream()/step_structured(): 按需覆盖
    - before_step/after_step/on_error: Hook 扩展点
    - initialize/cleanup: 生命周期管理
    """

    def __init__(
        self,
        model: ModelProtocol,
        toolbox: ToolBox,
        memory: TokenMemory,
        event_bus: Optional[EventBus] = None,
        max_iterations: int = 10,
        enable_stats: bool = True,
        **kwargs: Any,
    ):
        # 1) 验证模型协议（鸭子类型检查）
        validate_model(model)
        self.model = model

        self.toolbox = toolbox
        self.memory = memory
        self.event_bus = event_bus
        self.max_iterations = max_iterations
        self.enable_stats = enable_stats

        # 2) 统计对象（可选）
        self.stats: Optional[ExecutionStats] = ExecutionStats() if enable_stats else None

        # 3) Hook 槽位
        self.before_step_hook: Optional[Callable[..., Any]] = None
        self.after_step_hook: Optional[Callable[..., Any]] = None
        self.on_error_hook: Optional[Callable[..., Any]] = None

        # 4) 扩展配置
        self._config: Dict[str, Any] = dict(kwargs)

        # Hook 异常是否 fail-fast（默认 False：Hook 失败不影响主流程）
        self.hooks_fail_fast: bool = bool(kwargs.get("hooks_fail_fast", False))

        logger.debug(
            "引擎初始化完成",
            engine=self.__class__.__name__,
            model=type(model).__name__,
            max_iterations=max_iterations,
            event_bus_enabled=event_bus is not None,
            stats_enabled=enable_stats,
        )

    # ====================== 核心抽象方法 ======================

    @abstractmethod
    async def step(self, input_messages: List[Message], **kwargs: Any) -> AgentOutput:
        """
        执行推理步骤（必须实现）
        """
        raise NotImplementedError

    # ====================== 可选方法（按需覆盖）======================

    async def step_stream(
        self, input_messages: List[Message], **kwargs: Any
    ) -> AsyncIterator[AgentStreamEvent]:
        """
        流式推理（可选实现）
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 不支持流式输出。"
            f"请重写 step_stream() 方法以启用此功能。"
        )

    async def step_text_stream(
        self, input_messages: List[Message], **kwargs: Any
    ) -> AsyncIterator[str]:
        """
        纯文本 token 流（兼容层）
        从 step_stream 事件流中过滤 token。
        """
        async for event in self.step_stream(input_messages, **kwargs):  # type: ignore
            if event.type == "token" and event.content is not None:
                yield str(event.content)

    async def step_structured(
        self,
        input_messages: List[Message],
        response_model: Type[T],
        **kwargs: Any,
    ) -> T:
        """
        结构化输出（可选实现）
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 不支持结构化输出。"
            f"请重写 step_structured() 方法以启用此功能。"
        )

    # ====================== Hook 方法 ======================

    async def before_step(self, input_messages: List[Message], **kwargs: Any) -> None:
        """
        步骤执行前 Hook
        """
        await self._publish_event(
            "step_started",
            {"message_count": len(input_messages), "engine": self.__class__.__name__},
        )

        if self.before_step_hook:
            try:
                await self._maybe_await(self.before_step_hook, input_messages, **kwargs)
            except Exception as e:
                logger.warning("before_step_hook 执行失败", error=str(e), exc_info=True)
                if self.hooks_fail_fast:
                    raise

    async def after_step(
        self, input_messages: List[Message], output: AgentOutput, **kwargs: Any
    ) -> None:
        """
        步骤执行后 Hook
        """
        await self._publish_event(
            "step_completed",
            {
                "content_length": len(output.content) if output.content else 0,
                "has_tool_calls": bool(output.tool_calls),
                "engine": self.__class__.__name__,
            },
        )

        if self.after_step_hook:
            try:
                await self._maybe_await(self.after_step_hook, input_messages, output, **kwargs)
            except Exception as e:
                logger.warning("after_step_hook 执行失败", error=str(e), exc_info=True)
                if self.hooks_fail_fast:
                    raise

    async def on_error(
        self, error: Exception, input_messages: List[Message], **kwargs: Any
    ) -> None:
        """
        错误处理 Hook
        """
        await self._publish_event(
            "step_error",
            {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "engine": self.__class__.__name__,
            },
        )

        if self.on_error_hook:
            try:
                await self._maybe_await(self.on_error_hook, error, input_messages, **kwargs)
            except Exception as e:
                logger.error("on_error_hook 执行失败", error=str(e), exc_info=True)
                if self.hooks_fail_fast:
                    raise

    # ====================== 事件发布（更稳健的 awaitable 处理）======================

    async def _maybe_await(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        统一处理：
        - async def 函数
        - sync 函数
        - sync 函数但返回 awaitable（某些库会这样写）
        """
        try:
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                return await result
            return result
        except TypeError:
            # 兼容某些对象的 __call__ / 方法绑定行为
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            return func(*args, **kwargs)

    async def _publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        发布事件到 EventBus（失败不影响主流程）

        说明：
        - event_bus.publish 可能是 sync / async
        - 也可能是 sync 但返回 awaitable
        """
        if self.event_bus is None:
            return

        publish = getattr(self.event_bus, "publish", None)
        if publish is None:
            return

        try:
            await self._maybe_await(publish, event_type, data)
        except Exception as e:
            # 事件链路不应影响主业务流程：这里选择 debug 级别
            logger.debug("事件发布失败", event_type=event_type, error=str(e))

    # ====================== 统计辅助方法（统一入口，禁止裸写）======================

    def record_step(
        self,
        duration: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        had_error: bool = False,
    ) -> None:
        """记录一次执行步骤"""
        if self.stats is None:
            return
        try:
            self.stats.add_step(
                duration=float(duration),
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                had_error=bool(had_error),
            )
        except Exception as e:
            logger.debug("更新步骤统计失败", error=str(e))

    def record_tokens(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """仅记录 token 统计"""
        if self.stats is None:
            return
        try:
            self.stats.add_tokens(input_tokens=input_tokens, output_tokens=output_tokens)
        except Exception as e:
            logger.debug("更新 token 统计失败", error=str(e))

    def record_tool_call(self, n: int = 1) -> None:
        """记录工具调用次数"""
        if self.stats is None:
            return
        try:
            self.stats.add_tool_call(n=n)
        except Exception as e:
            logger.debug("更新工具调用统计失败", error=str(e))

    def record_error(self, n: int = 1) -> None:
        """记录错误次数"""
        if self.stats is None:
            return
        try:
            self.stats.add_error(n=n)
        except Exception as e:
            logger.debug("更新错误统计失败", error=str(e))

    def record_cost(self, input_tokens: int = 0, output_tokens: int = 0, model_name: str = "") -> None:
        """
        基于 token 数与模型名称记录估算成本

        工业级策略：
        - 定价表懒加载 + 缓存
        - 模型名优先精确匹配，否则做前缀匹配（兼容 gpt-4-0125-preview 这类）
        - 兜底使用 gpt-3.5-turbo
        """
        if self.stats is None:
            return

        pricing_table = get_model_pricing()

        pricing = pricing_table.get(model_name)
        if not pricing and model_name:
            # 前缀匹配：model_name.startswith(key)
            for key, val in pricing_table.items():
                if model_name.startswith(key):
                    pricing = val
                    break

        if not pricing:
            pricing = pricing_table.get("gpt-3.5-turbo", {"input": 0.5, "output": 1.5})

        try:
            cost = (
                float(input_tokens) * float(pricing["input"]) / 1_000_000.0
                + float(output_tokens) * float(pricing["output"]) / 1_000_000.0
            )
            self.stats.add_cost(cost)
        except Exception as e:
            logger.debug(
                "记录成本失败",
                error=str(e),
                model=model_name,
                tokens=(input_tokens, output_tokens),
            )

    def get_stats_summary(self) -> Dict[str, Any]:
        """获取执行统计摘要"""
        if self.stats is None:
            return {}
        return self.stats.to_dict()

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """获取执行统计（兼容旧接口）"""
        return self.stats.to_dict() if self.stats else None

    def reset_stats(self) -> None:
        """重置统计信息"""
        if self.stats is not None:
            self.stats = ExecutionStats()
            logger.debug("统计信息已重置")

    # ====================== 工具方法 ======================

    def validate_input(self, input_messages: List[Message]) -> None:
        """验证输入消息"""
        if not input_messages:
            raise ValueError("input_messages 不能为空")
        if not all(isinstance(m, Message) for m in input_messages):
            raise TypeError("所有输入必须是 Message 实例")

        logger.debug("输入验证通过", message_count=len(input_messages))

    def supports_streaming(self) -> bool:
        """
        检查引擎是否支持流式输出

        需要同时满足：
        1) 模型支持 streaming
        2) 引擎重写了 step_stream()
        """
        model_supports = supports_streaming(self.model)
        engine_supports = self.__class__.step_stream != CognitiveEngine.step_stream
        return bool(model_supports and engine_supports)

    def get_config(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        return self._config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        """设置配置项"""
        self._config[key] = value

    # ====================== 生命周期管理 ======================

    async def initialize(self) -> None:
        """初始化引擎资源（子类可重写）"""
        logger.debug("引擎初始化", engine=self.__class__.__name__)

    async def cleanup(self) -> None:
        """清理资源（子类可重写）"""
        logger.debug("引擎清理", engine=self.__class__.__name__)

    async def health_check(self) -> Dict[str, Any]:
        """健康检查接口"""
        return {
            "engine": self.__class__.__name__,
            "model": type(self.model).__name__,
            "supports_streaming": self.supports_streaming(),
            "stats": self.get_stats_summary(),
            "status": "healthy",
        }

    # ====================== 上下文管理器 ======================

    async def __aenter__(self) -> "CognitiveEngine":
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        await self.cleanup()
        return False  # 不抑制异常

    # ====================== 辅助方法 ======================

    async def _safe_execute(
        self,
        func: Callable[..., Any],
        *args: Any,
        record_stats: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        安全执行函数（带错误处理）

        ✅ 重要设计变更：
        - 默认 record_stats=False：避免与子类 record_step() 形成双重统计
        - 如果你确实想让 _safe_execute 自动计步，可以显式传 record_stats=True

        参数：
            func: 要执行的函数
            record_stats: 是否在这里自动记录一次 step（默认 False）
        """
        start_time = time.time()
        had_error = False

        try:
            # 兼容：async func / sync func / sync func 返回 awaitable
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result

        except Exception as e:
            had_error = True

            # 统计错误（只记错误，不强行记步）
            self.record_error(1)

            # 调用错误 Hook（不让 Hook 的失败吞掉原错误）
            try:
                await self.on_error(e, kwargs.get("input_messages", []))
            except Exception as hook_err:
                logger.debug("on_error hook raised", error=str(hook_err))

            logger.exception(
                "引擎执行失败",
                engine=self.__class__.__name__,
                error=str(e),
            )
            raise

        finally:
            if record_stats:
                duration = time.time() - start_time
                self.record_step(duration=duration, had_error=had_error)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={type(self.model).__name__}, "
            f"max_iterations={self.max_iterations}, "
            f"stats_enabled={self.stats is not None}"
            f")"
        )


# =============================================================================
# 工厂函数
# =============================================================================

def create_engine(
    engine_class: Type[CognitiveEngine],
    model: ModelProtocol,
    toolbox: ToolBox,
    memory: TokenMemory,
    **kwargs: Any,
) -> CognitiveEngine:
    """
    创建引擎实例（工厂函数）
    """
    if not issubclass(engine_class, CognitiveEngine):
        raise TypeError(
            f"engine_class 必须是 CognitiveEngine 的子类，收到: {engine_class.__name__}"
        )

    return engine_class(model=model, toolbox=toolbox, memory=memory, **kwargs)


# =============================================================================
# 模块导出
# =============================================================================

__all__ = [
    "CognitiveEngine",
    "ExecutionStats",
    "create_engine",
    "MODEL_PRICING",
    "load_model_pricing",
    "get_model_pricing",
]
