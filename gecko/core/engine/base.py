# gecko/core/engine/base.py

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import os
import time
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Type, TypeVar, MutableMapping, Iterator, Tuple
from collections.abc import Mapping

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

T = TypeVar("T", bound=BaseModel)

DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    "claude-3-opus": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet": {"input": 3.0, "output": 15.0},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    "claude-3.5-sonnet": {"input": 3.0, "output": 15.0},
}

FALLBACK_PRICING: Dict[str, float] = {"input": 0.5, "output": 1.5}


def _load_default_pricing() -> Dict[str, Dict[str, float]]:
    return dict(DEFAULT_PRICING)


def load_model_pricing() -> Dict[str, Dict[str, float]]:
    pricing = _load_default_pricing()

    custom_path = os.environ.get("GECKO_PRICING_FILE")
    if custom_path:
        config_file = Path(custom_path)
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    custom_pricing = json.load(f)
                if isinstance(custom_pricing, dict):
                    pricing.update(custom_pricing)
                    logger.debug("Loaded custom pricing config", path=str(config_file))
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to load pricing config", error=str(e), path=str(config_file))

    user_config = Path.home() / ".gecko" / "pricing.json"
    if user_config.exists():
        try:
            with open(user_config, "r", encoding="utf-8") as f:
                custom_pricing = json.load(f)
            if isinstance(custom_pricing, dict):
                pricing.update(custom_pricing)
                logger.debug("Loaded user pricing config", path=str(user_config))
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("Failed to load user pricing config", error=str(e), path=str(user_config))

    return pricing


_PRICING_CACHE: Optional[Dict[str, Dict[str, float]]] = None
_PRICING_LOCK = threading.Lock()
_PRICING_PREFIX_CACHE: Dict[str, Optional[Dict[str, float]]] = {}
_PRICING_PREFIX_CACHE_LOCK = threading.Lock()


def get_model_pricing(force_reload: bool = False) -> Mapping[str, Dict[str, float]]:
    global _PRICING_CACHE
    
    with _PRICING_LOCK:
        if force_reload or _PRICING_CACHE is None:
            _PRICING_CACHE = load_model_pricing()
            with _PRICING_PREFIX_CACHE_LOCK:
                _PRICING_PREFIX_CACHE.clear()
        return dict(_PRICING_CACHE)


def get_pricing_for_model(model_name: str) -> Dict[str, float]:
    if not model_name:
        return dict(FALLBACK_PRICING)
    
    with _PRICING_PREFIX_CACHE_LOCK:
        if model_name in _PRICING_PREFIX_CACHE:
            cached = _PRICING_PREFIX_CACHE[model_name]
            return dict(cached) if cached else dict(FALLBACK_PRICING)
    
    pricing_table = get_model_pricing()
    
    if model_name in pricing_table:
        result = pricing_table[model_name]
        with _PRICING_PREFIX_CACHE_LOCK:
            _PRICING_PREFIX_CACHE[model_name] = result
        return dict(result)
    
    best_match: Optional[str] = None
    best_len = 0
    for key in pricing_table:
        if model_name.startswith(key) and len(key) > best_len:
            best_match = key
            best_len = len(key)
    
    if best_match:
        result = pricing_table[best_match]
        with _PRICING_PREFIX_CACHE_LOCK:
            _PRICING_PREFIX_CACHE[model_name] = result
        return dict(result)
    
    with _PRICING_PREFIX_CACHE_LOCK:
        _PRICING_PREFIX_CACHE[model_name] = None
    return dict(FALLBACK_PRICING)


class _LazyPricingDict(MutableMapping[str, Dict[str, float]]):
    def __init__(self) -> None:
        self._override: Optional[Dict[str, Dict[str, float]]] = None
        self._lock = threading.RLock()

    def set_override(self, pricing: Dict[str, Dict[str, float]]) -> None:
        with self._lock:
            self._override = dict(pricing)

    def _data(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            if self._override is not None:
                return self._override
            return dict(get_model_pricing())

    def __getitem__(self, key: str) -> Dict[str, float]:
        with self._lock:
            return self._data()[key]

    def __setitem__(self, key: str, value: Dict[str, float]) -> None:
        with self._lock:
            if self._override is None:
                self._override = dict(get_model_pricing())
            self._override[key] = value

    def __delitem__(self, key: str) -> None:
        with self._lock:
            if self._override is None:
                self._override = dict(get_model_pricing())
            del self._override[key]

    def __iter__(self) -> Iterator[str]:
        with self._lock:
            keys = list(self._data().keys())
        return iter(keys)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data())

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._data()

    def get(self, key: str, default: Optional[Dict[str, float]] = None) -> Optional[Dict[str, float]]:
        with self._lock:
            return self._data().get(key, default)

    def keys(self):
        with self._lock:
            return list(self._data().keys())

    def values(self):
        with self._lock:
            return list(self._data().values())

    def items(self):
        with self._lock:
            return list(self._data().items())

    def __repr__(self) -> str:
        return f"LazyPricingDict(len={len(self)})"


MODEL_PRICING: MutableMapping[str, Dict[str, float]] = _LazyPricingDict()


class ExecutionStats:
    __slots__ = (
        "total_steps", "total_time", "input_tokens", "output_tokens",
        "tool_calls", "errors", "estimated_cost", "_lock"
    )
    
    def __init__(self) -> None:
        self.total_steps: int = 0
        self.total_time: float = 0.0
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.tool_calls: int = 0
        self.errors: int = 0
        self.estimated_cost: float = 0.0
        self._lock: threading.RLock = threading.RLock()

    def add_step(
        self,
        duration: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        had_error: bool = False,
    ) -> None:
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
        if not input_tokens and not output_tokens:
            return
        with self._lock:
            if input_tokens:
                self.input_tokens += int(input_tokens)
            if output_tokens:
                self.output_tokens += int(output_tokens)

    def add_tool_call(self, n: int = 1) -> None:
        if n <= 0:
            return
        with self._lock:
            self.tool_calls += int(n)

    def add_error(self, n: int = 1) -> None:
        if n <= 0:
            return
        with self._lock:
            self.errors += int(n)

    def add_cost(self, cost: float) -> None:
        if not cost:
            return
        with self._lock:
            self.estimated_cost += float(cost)

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
    
    def reset(self) -> None:
        with self._lock:
            self.total_steps = 0
            self.total_time = 0.0
            self.input_tokens = 0
            self.output_tokens = 0
            self.tool_calls = 0
            self.errors = 0
            self.estimated_cost = 0.0


class CognitiveEngine(ABC):
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
        validate_model(model)
        self.model = model

        self.toolbox = toolbox
        self.memory = memory
        self.event_bus = event_bus
        self.max_iterations = max_iterations
        self.enable_stats = enable_stats

        self.stats: Optional[ExecutionStats] = ExecutionStats() if enable_stats else None

        self.before_step_hook: Optional[Callable[..., Any]] = None
        self.after_step_hook: Optional[Callable[..., Any]] = None
        self.on_error_hook: Optional[Callable[..., Any]] = None

        self._config: Dict[str, Any] = dict(kwargs)
        self._config_lock = threading.RLock()

        self.hooks_fail_fast: bool = bool(kwargs.get("hooks_fail_fast", False))

        logger.debug(
            "Engine initialized",
            engine=self.__class__.__name__,
            model=type(model).__name__,
            max_iterations=max_iterations,
            event_bus_enabled=event_bus is not None,
            stats_enabled=enable_stats,
        )

    @abstractmethod
    async def step(self, input_messages: List[Message], **kwargs: Any) -> AgentOutput:
        raise NotImplementedError

    async def step_stream(
        self, input_messages: List[Message], **kwargs: Any
    ) -> AsyncIterator[AgentStreamEvent]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support streaming. "
            f"Override step_stream() to enable this feature."
        )

    async def step_text_stream(
        self, input_messages: List[Message], **kwargs: Any
    ) -> AsyncIterator[str]:
        async for event in self.step_stream(input_messages, **kwargs):
            if event.type == "token" and event.content is not None:
                yield str(event.content)

    async def step_structured(
        self,
        input_messages: List[Message],
        response_model: Type[T],
        **kwargs: Any,
    ) -> T:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support structured output. "
            f"Override step_structured() to enable this feature."
        )

    async def before_step(self, input_messages: List[Message], **kwargs: Any) -> None:
        await self._publish_event(
            "step_started",
            {"message_count": len(input_messages), "engine": self.__class__.__name__},
        )

        if self.before_step_hook:
            try:
                await self._maybe_await(self.before_step_hook, input_messages, **kwargs)
            except Exception as e:
                logger.warning("before_step_hook failed", error=str(e), exc_info=True)
                if self.hooks_fail_fast:
                    raise

    async def after_step(
        self, input_messages: List[Message], output: AgentOutput, **kwargs: Any
    ) -> None:
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
                logger.warning("after_step_hook failed", error=str(e), exc_info=True)
                if self.hooks_fail_fast:
                    raise

    async def on_error(
        self, error: Exception, input_messages: List[Message], **kwargs: Any
    ) -> None:
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
                logger.error("on_error_hook failed", error=str(e), exc_info=True)
                if self.hooks_fail_fast:
                    raise

    async def _maybe_await(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        if self.event_bus is None:
            return

        publish = getattr(self.event_bus, "publish", None)
        if publish is None:
            return

        try:
            await self._maybe_await(publish, event_type, data)
        except Exception as e:
            logger.debug("Event publish failed", event_type=event_type, error=str(e))

    def record_step(
        self,
        duration: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        had_error: bool = False,
    ) -> None:
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
            logger.debug("Failed to update step stats", error=str(e))

    def record_tokens(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        if self.stats is None:
            return
        try:
            self.stats.add_tokens(input_tokens=input_tokens, output_tokens=output_tokens)
        except Exception as e:
            logger.debug("Failed to update token stats", error=str(e))

    def record_tool_call(self, n: int = 1) -> None:
        if self.stats is None:
            return
        try:
            self.stats.add_tool_call(n=n)
        except Exception as e:
            logger.debug("Failed to update tool call stats", error=str(e))

    def record_error(self, n: int = 1) -> None:
        if self.stats is None:
            return
        try:
            self.stats.add_error(n=n)
        except Exception as e:
            logger.debug("Failed to update error stats", error=str(e))

    def record_cost(self, input_tokens: int = 0, output_tokens: int = 0, model_name: str = "") -> None:
        if self.stats is None:
            return

        pricing = get_pricing_for_model(model_name)

        try:
            cost = (
                float(input_tokens) * float(pricing["input"]) / 1_000_000.0
                + float(output_tokens) * float(pricing["output"]) / 1_000_000.0
            )
            self.stats.add_cost(cost)
        except Exception as e:
            logger.debug(
                "Failed to record cost",
                error=str(e),
                model=model_name,
                tokens=(input_tokens, output_tokens),
            )

    def get_stats_summary(self) -> Dict[str, Any]:
        if self.stats is None:
            return {}
        return self.stats.to_dict()

    def get_stats(self) -> Optional[Dict[str, Any]]:
        return self.stats.to_dict() if self.stats else None

    def reset_stats(self) -> None:
        if self.stats is not None:
            self.stats.reset()
            logger.debug("Stats reset")

    def validate_input(self, input_messages: List[Message]) -> None:
        if not input_messages:
            raise ValueError("input_messages cannot be empty")
        if not all(isinstance(m, Message) for m in input_messages):
            raise TypeError("All inputs must be Message instances")

        logger.debug("Input validation passed", message_count=len(input_messages))

    def supports_streaming(self) -> bool:
        model_supports = supports_streaming(self.model)
        engine_supports = self.__class__.step_stream != CognitiveEngine.step_stream
        return bool(model_supports and engine_supports)

    def get_config(self, key: str, default: Any = None) -> Any:
        with self._config_lock:
            return self._config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        with self._config_lock:
            self._config[key] = value

    async def initialize(self) -> None:
        logger.debug("Engine initializing", engine=self.__class__.__name__)

    async def cleanup(self) -> None:
        try:
            async with asyncio.timeout(30.0):
                await self._do_cleanup()
        except asyncio.TimeoutError:
            logger.error("Cleanup timeout", engine=self.__class__.__name__)
        except Exception as e:
            logger.error("Cleanup failed", error=str(e))

    async def _do_cleanup(self) -> None:
        logger.debug("Engine cleanup", engine=self.__class__.__name__)

    async def health_check(self) -> Dict[str, Any]:
        return {
            "engine": self.__class__.__name__,
            "model": type(self.model).__name__,
            "supports_streaming": self.supports_streaming(),
            "stats": self.get_stats_summary(),
            "status": "healthy",
        }

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
        return False

    async def _safe_execute(
        self,
        func: Callable[..., Any],
        *args: Any,
        record_stats: bool = False,
        **kwargs: Any,
    ) -> Any:
        start_time = time.time()
        had_error = False

        try:
            result = func(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result

        except Exception as e:
            had_error = True
            self.record_error(1)

            try:
                await self.on_error(e, kwargs.get("input_messages", []))
            except Exception as hook_err:
                logger.debug("on_error hook raised", error=str(hook_err))

            logger.exception(
                "Engine execution failed",
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


def create_engine(
    engine_class: Type[CognitiveEngine],
    model: ModelProtocol,
    toolbox: ToolBox,
    memory: TokenMemory,
    **kwargs: Any,
) -> CognitiveEngine:
    if not issubclass(engine_class, CognitiveEngine):
        raise TypeError(
            f"engine_class must be a subclass of CognitiveEngine, got: {engine_class.__name__}"
        )

    return engine_class(model=model, toolbox=toolbox, memory=memory, **kwargs)


__all__ = [
    "CognitiveEngine",
    "ExecutionStats",
    "create_engine",
    "MODEL_PRICING",
    "load_model_pricing",
    "get_model_pricing",
    "get_pricing_for_model",
]