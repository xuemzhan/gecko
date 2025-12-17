# gecko/core/engine/base.py
"""
认知引擎基础模块
================

这个模块定义了 Gecko AI 框架的核心引擎抽象类和相关工具。

主要组件：
1. ExecutionStats: 执行统计跟踪器，记录 token 使用、工具调用、错误等
2. CognitiveEngine: 抽象基类，定义了所有认知引擎必须实现的接口
3. 模型定价系统: 管理不同 AI 模型的 token 定价
4. 工具函数: 创建引擎、加载定价等辅助函数
"""

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

# ============================================================================
# 模型定价配置
# ============================================================================
# 默认的模型定价表，以美元/百万 tokens 为单位
# 分为输入 token 和输出 token 两种价格
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

# 当找不到模型定价时使用的后备价格
FALLBACK_PRICING: Dict[str, float] = {"input": 0.5, "output": 1.5}


def _load_default_pricing() -> Dict[str, Dict[str, float]]:
    """
    加载默认定价表的副本
    
    返回:
        默认定价表的深拷贝，防止外部修改
    """
    return dict(DEFAULT_PRICING)


def load_model_pricing() -> Dict[str, Dict[str, float]]:
    """
    加载模型定价配置，支持多层覆盖
    
    加载顺序（后者覆盖前者）：
    1. 内置默认定价
    2. 环境变量指定的自定义文件 (GECKO_PRICING_FILE)
    3. 用户主目录配置 (~/.gecko/pricing.json)
    
    返回:
        合并后的完整定价表
    """
    # 从默认定价开始
    pricing = _load_default_pricing()

    # 尝试加载环境变量指定的自定义配置
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

    # 尝试加载用户主目录的配置
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


# ============================================================================
# 定价缓存系统
# ============================================================================
# 全局定价缓存，避免重复加载文件
_PRICING_CACHE: Optional[Dict[str, Dict[str, float]]] = None
# 保护缓存的线程锁
_PRICING_LOCK = threading.Lock()
# 前缀匹配缓存，存储已查询过的模型名称对应的定价
_PRICING_PREFIX_CACHE: Dict[str, Optional[Dict[str, float]]] = {}
# 保护前缀缓存的线程锁
_PRICING_PREFIX_CACHE_LOCK = threading.Lock()


def get_model_pricing(force_reload: bool = False) -> Mapping[str, Dict[str, float]]:
    """
    获取模型定价表（带缓存）
    
    参数:
        force_reload: 是否强制重新加载配置文件
        
    返回:
        只读的定价映射表
    """
    global _PRICING_CACHE
    
    with _PRICING_LOCK:
        # 如果需要重新加载或缓存为空，则加载配置
        if force_reload or _PRICING_CACHE is None:
            _PRICING_CACHE = load_model_pricing()
            # 清空前缀缓存，因为定价已更新
            with _PRICING_PREFIX_CACHE_LOCK:
                _PRICING_PREFIX_CACHE.clear()
        return dict(_PRICING_CACHE)

def get_pricing_for_model(model_name: str) -> Dict[str, float]:
    """
    获取指定模型的定价信息，支持前缀匹配
    
    匹配策略：
    1. 精确匹配模型名称
    2. 前缀匹配（如 "gpt-4-0613" 匹配 "gpt-4"）
    3. 使用后备定价
    
    参数:
        model_name: 模型名称，如 "glm-4" 或 "claude-3-opus-20240229"
        
    返回:
        包含 input 和 output 价格的字典（美元/百万tokens）
    """
    # 空模型名使用后备定价
    if not model_name:
        return dict(FALLBACK_PRICING)
    
    # 检查前缀缓存
    with _PRICING_PREFIX_CACHE_LOCK:
        if model_name in _PRICING_PREFIX_CACHE:
            cached = _PRICING_PREFIX_CACHE[model_name]
            return dict(cached) if cached else dict(FALLBACK_PRICING)
    
    # 获取完整定价表
    pricing_table = get_model_pricing()
    
    # 尝试精确匹配
    if model_name in pricing_table:
        result = pricing_table[model_name]
        with _PRICING_PREFIX_CACHE_LOCK:
            _PRICING_PREFIX_CACHE[model_name] = result
        return dict(result)
    
    # 尝试前缀匹配：找到最长的匹配前缀
    best_match: Optional[str] = None
    best_len = 0
    for key in pricing_table:
        if model_name.startswith(key) and len(key) > best_len:
            best_match = key
            best_len = len(key)
    
    # 找到前缀匹配
    if best_match:
        result = pricing_table[best_match]
        with _PRICING_PREFIX_CACHE_LOCK:
            _PRICING_PREFIX_CACHE[model_name] = result
        return dict(result)
    
    # 没有匹配，使用后备定价并缓存
    with _PRICING_PREFIX_CACHE_LOCK:
        _PRICING_PREFIX_CACHE[model_name] = None
    return dict(FALLBACK_PRICING)


class _LazyPricingDict(MutableMapping[str, Dict[str, float]]):
    """
    延迟加载的定价字典
    
    这个类实现了字典接口，但延迟加载实际数据，直到真正需要时才加载。
    支持覆盖默认定价，用于测试或自定义场景。
    """
    
    def __init__(self) -> None:
        """初始化延迟字典，不加载任何数据"""
        self._override: Optional[Dict[str, Dict[str, float]]] = None
        self._lock = threading.RLock()  # 使用可重入锁

    def set_override(self, pricing: Dict[str, Dict[str, float]]) -> None:
        """
        设置覆盖定价表
        
        参数:
            pricing: 要使用的自定义定价表
        """
        with self._lock:
            self._override = dict(pricing)

    def _data(self) -> Dict[str, Dict[str, float]]:
        """
        获取当前生效的定价数据
        
        返回:
            如果设置了覆盖，返回覆盖数据；否则返回全局定价
        """
        with self._lock:
            if self._override is not None:
                return self._override
            return dict(get_model_pricing())

    # 实现 MutableMapping 接口的必需方法
    def __getitem__(self, key: str) -> Dict[str, float]:
        """通过键获取定价"""
        with self._lock:
            return self._data()[key]

    def __setitem__(self, key: str, value: Dict[str, float]) -> None:
        """设置模型定价"""
        with self._lock:
            if self._override is None:
                # 首次修改时，复制当前定价作为覆盖基础
                self._override = dict(get_model_pricing())
            self._override[key] = value

    def __delitem__(self, key: str) -> None:
        """删除模型定价"""
        with self._lock:
            if self._override is None:
                self._override = dict(get_model_pricing())
            del self._override[key]

    def __iter__(self) -> Iterator[str]:
        """迭代所有模型名称"""
        with self._lock:
            keys = list(self._data().keys())
        return iter(keys)

    def __len__(self) -> int:
        """返回定价表中的模型数量"""
        with self._lock:
            return len(self._data())

    def __contains__(self, key: object) -> bool:
        """检查模型是否在定价表中"""
        with self._lock:
            return key in self._data()

    def get(self, key: str, default: Optional[Dict[str, float]] = None) -> Optional[Dict[str, float]]: # type: ignore
        """安全获取定价，不存在时返回默认值"""
        with self._lock:
            return self._data().get(key, default)

    def keys(self): # type: ignore
        """返回所有模型名称列表"""
        with self._lock:
            return list(self._data().keys())

    def values(self): # type: ignore
        """返回所有定价值列表"""
        with self._lock:
            return list(self._data().values())

    def items(self): # type: ignore
        """返回所有 (模型名, 定价) 对列表"""
        with self._lock:
            return list(self._data().items())

    def __repr__(self) -> str:
        return f"LazyPricingDict(len={len(self)})"


# 全局模型定价字典，提供给外部使用
MODEL_PRICING: MutableMapping[str, Dict[str, float]] = _LazyPricingDict()


# ============================================================================
# 执行统计类
# ============================================================================
class ExecutionStats:
    """
    执行统计跟踪器
    
    线程安全地记录引擎执行过程中的各种指标：
    - 步骤数和总耗时
    - token 使用量（输入/输出）
    - 工具调用次数
    - 错误次数
    - 估算成本
    
    使用 __slots__ 优化内存占用
    """
    __slots__ = (
        "total_steps", "total_time", "input_tokens", "output_tokens",
        "tool_calls", "errors", "estimated_cost", "_lock"
    )
    
    def __init__(self) -> None:
        """初始化所有统计指标为零"""
        self.total_steps: int = 0          # 总步骤数
        self.total_time: float = 0.0       # 总执行时间（秒）
        self.input_tokens: int = 0         # 输入 token 总数
        self.output_tokens: int = 0        # 输出 token 总数
        self.tool_calls: int = 0           # 工具调用总数
        self.errors: int = 0               # 错误总数
        self.estimated_cost: float = 0.0   # 估算成本（美元）
        self._lock: threading.RLock = threading.RLock()  # 线程锁

    def add_step(
        self,
        duration: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        had_error: bool = False,
    ) -> None:
        """
        记录一个执行步骤
        
        参数:
            duration: 步骤耗时（秒）
            input_tokens: 输入 token 数
            output_tokens: 输出 token 数
            had_error: 是否发生错误
        """
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
        """
        添加 token 使用量
        
        参数:
            input_tokens: 输入 token 数
            output_tokens: 输出 token 数
        """
        if not input_tokens and not output_tokens:
            return
        with self._lock:
            if input_tokens:
                self.input_tokens += int(input_tokens)
            if output_tokens:
                self.output_tokens += int(output_tokens)

    def add_tool_call(self, n: int = 1) -> None:
        """
        添加工具调用次数
        
        参数:
            n: 调用次数
        """
        if n <= 0:
            return
        with self._lock:
            self.tool_calls += int(n)

    def add_error(self, n: int = 1) -> None:
        """
        添加错误次数
        
        参数:
            n: 错误次数
        """
        if n <= 0:
            return
        with self._lock:
            self.errors += int(n)

    def add_cost(self, cost: float) -> None:
        """
        添加成本
        
        参数:
            cost: 成本金额（美元）
        """
        if not cost:
            return
        with self._lock:
            self.estimated_cost += float(cost)

    def get_avg_step_time(self) -> float:
        """
        计算平均每步耗时
        
        返回:
            平均耗时（秒），如果没有步骤则返回 0
        """
        with self._lock:
            return self.total_time / self.total_steps if self.total_steps > 0 else 0.0

    def get_total_tokens(self) -> int:
        """
        获取总 token 数
        
        返回:
            输入和输出 token 之和
        """
        with self._lock:
            return self.input_tokens + self.output_tokens

    def get_error_rate(self) -> float:
        """
        计算错误率
        
        返回:
            错误数占总步骤数的比例
        """
        with self._lock:
            return self.errors / self.total_steps if self.total_steps > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """
        将统计数据转换为字典格式
        
        返回:
            包含所有统计指标的字典，数值已四舍五入
        """
        # 在锁内一次性获取所有数据，避免多次加锁
        with self._lock:
            total_steps = self.total_steps
            total_time = self.total_time
            input_tokens = self.input_tokens
            output_tokens = self.output_tokens
            tool_calls = self.tool_calls
            errors = self.errors
            estimated_cost = self.estimated_cost

        # 计算派生指标
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
        """重置所有统计数据为零"""
        with self._lock:
            self.total_steps = 0
            self.total_time = 0.0
            self.input_tokens = 0
            self.output_tokens = 0
            self.tool_calls = 0
            self.errors = 0
            self.estimated_cost = 0.0


# ============================================================================
# 认知引擎抽象基类
# ============================================================================
class CognitiveEngine(ABC):
    """
    认知引擎抽象基类
    
    这是所有 AI 引擎的基类，定义了统一的接口和通用功能。
    子类需要实现 step() 方法来定义具体的推理逻辑。
    
    核心功能：
    - 与 LLM 模型交互
    - 工具箱管理和调用
    - 记忆系统集成
    - 统计跟踪
    - 事件发布
    - 生命周期钩子
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
        """
        初始化认知引擎
        
        参数:
            model: LLM 模型实例，必须实现 ModelProtocol
            toolbox: 工具箱，包含可用的工具
            memory: 记忆系统，用于存储对话历史
            event_bus: 事件总线，用于发布事件（可选）
            max_iterations: 最大迭代次数，防止无限循环
            enable_stats: 是否启用统计跟踪
            **kwargs: 额外的配置参数
        """
        # 验证模型是否符合协议
        validate_model(model)
        self.model = model

        self.toolbox = toolbox
        self.memory = memory
        self.event_bus = event_bus
        self.max_iterations = max_iterations
        self.enable_stats = enable_stats

        # 创建统计跟踪器（如果启用）
        self.stats: Optional[ExecutionStats] = ExecutionStats() if enable_stats else None

        # 生命周期钩子函数
        self.before_step_hook: Optional[Callable[..., Any]] = None  # 步骤前钩子
        self.after_step_hook: Optional[Callable[..., Any]] = None   # 步骤后钩子
        self.on_error_hook: Optional[Callable[..., Any]] = None     # 错误处理钩子

        # 额外配置存储
        self._config: Dict[str, Any] = dict(kwargs)
        self._config_lock = threading.RLock()

        # 钩子失败策略：True = 快速失败，False = 记录警告后继续
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
        """
        执行一个推理步骤（抽象方法）
        
        子类必须实现此方法来定义具体的推理逻辑。
        
        参数:
            input_messages: 输入消息列表
            **kwargs: 额外参数
            
        返回:
            AgentOutput: 包含响应内容和工具调用的输出对象
            
        抛出:
            NotImplementedError: 子类未实现此方法
        """
        raise NotImplementedError

    async def step_stream(
        self, input_messages: List[Message], **kwargs: Any
    ) -> AsyncIterator[AgentStreamEvent]:
        """
        流式执行推理步骤
        
        默认不支持流式输出，子类可以重写此方法实现流式功能。
        
        参数:
            input_messages: 输入消息列表
            **kwargs: 额外参数
            
        生成:
            AgentStreamEvent: 流式事件
            
        抛出:
            NotImplementedError: 引擎不支持流式输出
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support streaming. "
            f"Override step_stream() to enable this feature."
        )

    async def step_text_stream(
        self, input_messages: List[Message], **kwargs: Any
    ) -> AsyncIterator[str]:
        """
        流式输出文本内容
        
        从 step_stream 中提取纯文本 token。
        
        参数:
            input_messages: 输入消息列表
            **kwargs: 额外参数
            
        生成:
            str: 文本片段
        """
        async for event in self.step_stream(input_messages, **kwargs): # type: ignore
            if event.type == "token" and event.content is not None:
                yield str(event.content)

    async def step_structured(
        self,
        input_messages: List[Message],
        response_model: Type[T],
        **kwargs: Any,
    ) -> T:
        """
        执行结构化输出推理
        
        要求模型返回符合指定 Pydantic 模型的结构化数据。
        
        参数:
            input_messages: 输入消息列表
            response_model: Pydantic 模型类
            **kwargs: 额外参数
            
        返回:
            T: 解析后的结构化对象
            
        抛出:
            NotImplementedError: 引擎不支持结构化输出
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support structured output. "
            f"Override step_structured() to enable this feature."
        )

    async def before_step(self, input_messages: List[Message], **kwargs: Any) -> None:
        """
        步骤执行前的钩子
        
        在每个推理步骤开始前调用，用于：
        - 发布事件
        - 执行自定义预处理逻辑
        
        参数:
            input_messages: 输入消息列表
            **kwargs: 额外参数
        """
        # 发布步骤开始事件
        await self._publish_event(
            "step_started",
            {"message_count": len(input_messages), "engine": self.__class__.__name__},
        )

        # 执行用户自定义钩子
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
        """
        步骤执行后的钩子
        
        在每个推理步骤完成后调用，用于：
        - 发布事件
        - 执行自定义后处理逻辑
        
        参数:
            input_messages: 输入消息列表
            output: 输出结果
            **kwargs: 额外参数
        """
        # 发布步骤完成事件
        await self._publish_event(
            "step_completed",
            {
                "content_length": len(output.content) if output.content else 0,
                "has_tool_calls": bool(output.tool_calls),
                "engine": self.__class__.__name__,
            },
        )

        # 执行用户自定义钩子
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
        """
        错误处理钩子
        
        当执行过程中发生错误时调用。
        
        参数:
            error: 异常对象
            input_messages: 输入消息列表
            **kwargs: 额外参数
        """
        # 发布错误事件
        await self._publish_event(
            "step_error",
            {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "engine": self.__class__.__name__,
            },
        )

        # 执行用户自定义钩子
        if self.on_error_hook:
            try:
                await self._maybe_await(self.on_error_hook, error, input_messages, **kwargs)
            except Exception as e:
                logger.error("on_error_hook failed", error=str(e), exc_info=True)
                if self.hooks_fail_fast:
                    raise

    async def _maybe_await(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """
        智能调用函数，自动处理同步/异步
        
        参数:
            func: 要调用的函数（可以是同步或异步）
            *args: 位置参数
            **kwargs: 关键字参数
            
        返回:
            函数执行结果
        """
        result = func(*args, **kwargs)
        if inspect.isawaitable(result):
            return await result
        return result

    async def _publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        发布事件到事件总线
        
        参数:
            event_type: 事件类型
            data: 事件数据
        """
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
        """
        记录步骤执行统计
        
        参数:
            duration: 执行耗时（秒）
            input_tokens: 输入 token 数
            output_tokens: 输出 token 数
            had_error: 是否发生错误
        """
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
        """
        记录 token 使用量
        
        参数:
            input_tokens: 输入 token 数
            output_tokens: 输出 token 数
        """
        if self.stats is None:
            return
        try:
            self.stats.add_tokens(input_tokens=input_tokens, output_tokens=output_tokens)
        except Exception as e:
            logger.debug("Failed to update token stats", error=str(e))

    def record_tool_call(self, n: int = 1) -> None:
        """
        记录工具调用次数
        
        参数:
            n: 调用次数
        """
        if self.stats is None:
            return
        try:
            self.stats.add_tool_call(n=n)
        except Exception as e:
            logger.debug("Failed to update tool call stats", error=str(e))

    def record_error(self, n: int = 1) -> None:
        """
        记录错误次数
        
        参数:
            n: 错误次数
        """
        if self.stats is None:
            return
        try:
            self.stats.add_error(n=n)
        except Exception as e:
            logger.debug("Failed to update error stats", error=str(e))

    def record_cost(self, input_tokens: int = 0, output_tokens: int = 0, model_name: str = "") -> None:
        """
        记录并计算 API 调用成本
        
        根据 token 使用量和模型定价计算成本。
        
        参数:
            input_tokens: 输入 token 数
            output_tokens: 输出 token 数
            model_name: 模型名称，用于查找定价
        """
        if self.stats is None:
            return

        # 获取模型定价信息
        pricing = get_pricing_for_model(model_name)

        try:
            # 计算成本：(tokens / 1,000,000) * 每百万 tokens 价格
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
        """
        获取统计摘要
        
        返回:
            统计数据字典，如果未启用统计则返回空字典
        """
        if self.stats is None:
            return {}
        return self.stats.to_dict()

    def get_stats(self) -> Optional[Dict[str, Any]]:
        """
        获取统计数据
        
        返回:
            统计数据字典或 None
        """
        return self.stats.to_dict() if self.stats else None

    def reset_stats(self) -> None:
        """重置所有统计数据"""
        if self.stats is not None:
            self.stats.reset()
            logger.debug("Stats reset")

    def validate_input(self, input_messages: List[Message]) -> None:
        """
        验证输入消息的有效性
        
        检查：
        - 消息列表不为空
        - 所有元素都是 Message 实例
        
        参数:
            input_messages: 要验证的消息列表
            
        抛出:
            ValueError: 消息列表为空
            TypeError: 包含非 Message 对象
        """
        if not input_messages:
            raise ValueError("input_messages cannot be empty")
        if not all(isinstance(m, Message) for m in input_messages):
            raise TypeError("All inputs must be Message instances")

        logger.debug("Input validation passed", message_count=len(input_messages))

    def supports_streaming(self) -> bool:
        """
        检查引擎是否支持流式输出
        
        同时检查模型和引擎本身是否都支持流式。
        
        返回:
            True 如果支持流式输出
        """
        model_supports = supports_streaming(self.model)
        engine_supports = self.__class__.step_stream != CognitiveEngine.step_stream
        return bool(model_supports and engine_supports)

    def get_config(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        参数:
            key: 配置键
            default: 默认值
            
        返回:
            配置值，不存在则返回默认值
        """
        with self._config_lock:
            return self._config.get(key, default)

    def set_config(self, key: str, value: Any) -> None:
        """
        设置配置值
        
        参数:
            key: 配置键
            value: 配置值
        """
        with self._config_lock:
            self._config[key] = value

    async def initialize(self) -> None:
        """
        初始化引擎
        
        在引擎开始工作前调用，子类可以重写此方法进行初始化。
        """
        logger.debug("Engine initializing", engine=self.__class__.__name__)

    async def cleanup(self) -> None:
        """
        清理引擎资源
        
        在引擎结束工作后调用，带超时保护。
        """
        try:
            async with asyncio.timeout(30.0):
                await self._do_cleanup()
        except asyncio.TimeoutError:
            logger.error("Cleanup timeout", engine=self.__class__.__name__)
        except Exception as e:
            logger.error("Cleanup failed", error=str(e))

    async def _do_cleanup(self) -> None:
        """
        实际的清理逻辑
        
        子类可以重写此方法实现具体的清理操作。
        """
        logger.debug("Engine cleanup", engine=self.__class__.__name__)

    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        返回引擎的状态信息。
        
        返回:
            包含引擎状态的字典
        """
        return {
            "engine": self.__class__.__name__,
            "model": type(self.model).__name__,
            "supports_streaming": self.supports_streaming(),
            "stats": self.get_stats_summary(),
            "status": "healthy",
        }

    async def __aenter__(self) -> "CognitiveEngine":
        """
        异步上下文管理器入口
        
        支持 `async with engine:` 语法。
        
        返回:
            引擎实例本身
        """
        await self.initialize()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Any,
    ) -> bool:
        """
        异步上下文管理器出口
        
        自动清理资源。
        
        返回:
            False（不抑制异常）
        """
        await self.cleanup()
        return False

    async def _safe_execute(
        self,
        func: Callable[..., Any],
        *args: Any,
        record_stats: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        安全执行函数，带统计和错误处理
        
        参数:
            func: 要执行的函数
            *args: 位置参数
            record_stats: 是否记录统计
            **kwargs: 关键字参数
            
        返回:
            函数执行结果
            
        抛出:
            执行过程中的异常会被记录并重新抛出
        """
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

            # 调用错误钩子
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
            # 记录统计（如果启用）
            if record_stats:
                duration = time.time() - start_time
                self.record_step(duration=duration, had_error=had_error)

    def __repr__(self) -> str:
        """
        字符串表示
        
        返回:
            引擎的可读字符串表示
        """
        return (
            f"{self.__class__.__name__}("
            f"model={type(self.model).__name__}, "
            f"max_iterations={self.max_iterations}, "
            f"stats_enabled={self.stats is not None}"
            f")"
        )


# ============================================================================
# 工具函数
# ============================================================================
def create_engine(
    engine_class: Type[CognitiveEngine],
    model: ModelProtocol,
    toolbox: ToolBox,
    memory: TokenMemory,
    **kwargs: Any,
) -> CognitiveEngine:
    """
    工厂函数：创建认知引擎实例
    
    参数:
        engine_class: 引擎类（必须继承自 CognitiveEngine）
        model: LLM 模型实例
        toolbox: 工具箱
        memory: 记忆系统
        **kwargs: 传递给引擎构造函数的额外参数
        
    返回:
        创建的引擎实例
        
    抛出:
        TypeError: engine_class 不是 CognitiveEngine 的子类
    """
    if not issubclass(engine_class, CognitiveEngine):
        raise TypeError(
            f"engine_class must be a subclass of CognitiveEngine, got: {engine_class.__name__}"
        )

    return engine_class(model=model, toolbox=toolbox, memory=memory, **kwargs)


# ============================================================================
# 模块导出
# ============================================================================
__all__ = [
    "CognitiveEngine",        # 认知引擎抽象基类
    "ExecutionStats",         # 执行统计类
    "create_engine",          # 引擎工厂函数
    "MODEL_PRICING",          # 全局定价字典
    "load_model_pricing",     # 加载定价配置
    "get_model_pricing",      # 获取定价表（带缓存）
    "get_pricing_for_model",  # 获取单个模型定价
]