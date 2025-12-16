# gecko/core/engine/base.py
"""
认知引擎基类模块

本模块定义了 Agent 推理引擎的抽象基类和相关工具类。
所有具体引擎实现（ReAct、Chain、Tree 等）都应继承 CognitiveEngine 基类。

核心概念：
- CognitiveEngine: 抽象基类，定义引擎统一接口
- ExecutionStats: 执行统计类，用于性能监控
- 支持普通推理、流式推理和结构化输出
- 提供 Hook 机制用于扩展
- 统一的错误处理和资源管理

修复记录：
- [P3] 将硬编码的模型定价配置改为可外部化加载
- [P2] 优化统计锁的使用方式
- [P1] 完善 Hook 机制的异常处理
"""
from __future__ import annotations

import asyncio
import json
import os
import time
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Type, TypeVar

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


# ====================== 模型定价配置 ======================

def _load_default_pricing() -> Dict[str, Dict[str, float]]:
    """
    加载默认的模型定价配置
    
    定价单位：USD per 1M tokens
    数据来源：各厂商官方定价（2024年）
    
    返回:
        Dict[str, Dict[str, float]]: 模型名称 -> {"input": 输入价格, "output": 输出价格}
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
    加载模型定价配置，支持外部文件覆盖
    
    加载优先级：
    1. 环境变量 GECKO_PRICING_FILE 指定的文件
    2. 用户目录下的 ~/.gecko/pricing.json
    3. 内置默认配置
    
    外部配置文件格式示例：
    {
        "gpt-4": {"input": 30.0, "output": 60.0},
        "custom-model": {"input": 1.0, "output": 2.0}
    }
    
    返回:
        Dict[str, Dict[str, float]]: 合并后的定价配置
    """
    # 加载默认配置
    pricing = _load_default_pricing()
    
    # 尝试从环境变量指定的文件加载
    custom_path = os.environ.get("GECKO_PRICING_FILE")
    if custom_path:
        config_file = Path(custom_path)
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    custom_pricing = json.load(f)
                    pricing.update(custom_pricing)
                    logger.debug(f"已加载自定义定价配置: {config_file}")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"加载定价配置文件失败: {e}")
    
    # 尝试从用户目录加载
    user_config = Path.home() / ".gecko" / "pricing.json"
    if user_config.exists():
        try:
            with open(user_config, "r", encoding="utf-8") as f:
                custom_pricing = json.load(f)
                pricing.update(custom_pricing)
                logger.debug(f"已加载用户定价配置: {user_config}")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"加载用户定价配置失败: {e}")
    
    return pricing


# 全局定价配置（模块加载时初始化）
MODEL_PRICING = load_model_pricing()


# ====================== 执行统计类 ======================

class ExecutionStats(BaseModel):
    """
    引擎执行统计
    
    用于性能监控、调试和成本跟踪。支持线程安全的统计更新。
    
    属性:
        total_steps: 总执行步数
        total_time: 总执行时间（秒）
        input_tokens: 输入 token 总数
        output_tokens: 输出 token 总数
        tool_calls: 工具调用次数
        errors: 错误次数
        estimated_cost: 估算成本（美元）
    
    使用示例:
        ```python
        stats = ExecutionStats()
        stats.add_step(duration=1.5, input_tokens=100, output_tokens=50)
        print(stats.get_avg_step_time())  # 1.5
        print(stats.get_total_tokens())   # 150
        ```
    """
    
    # 统计字段
    total_steps: int = 0
    total_time: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    errors: int = 0
    estimated_cost: float = 0.0
    
    # 私有属性：可重入锁，保证线程安全
    _lock: threading.RLock = PrivateAttr(default_factory=threading.RLock)

    def add_step(
        self, 
        duration: float, 
        input_tokens: int = 0, 
        output_tokens: int = 0, 
        had_error: bool = False
    ) -> None:
        """
        记录一次步骤执行
        
        线程安全：使用非阻塞方式获取锁，如果无法立即获取则跳过本次统计，
        优先保证业务流程不被阻塞。
        
        参数:
            duration: 执行耗时（秒）
            input_tokens: 本次输入 token 数
            output_tokens: 本次输出 token 数
            had_error: 是否发生错误
        """
        # 尝试非阻塞获取锁
        acquired = self._lock.acquire(blocking=False)
        if not acquired:
            # 无法立即获取锁时跳过，避免阻塞业务
            logger.debug("统计锁竞争，跳过本次记录")
            return
        
        try:
            self.total_steps += 1
            self.total_time += duration
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            if had_error:
                self.errors += 1
        finally:
            self._lock.release()

    def add_tool_call(self) -> None:
        """记录一次工具调用"""
        acquired = self._lock.acquire(blocking=False)
        if acquired:
            try:
                self.tool_calls += 1
            finally:
                self._lock.release()

    def add_cost(self, cost: float) -> None:
        """
        累加成本估算
        
        参数:
            cost: 本次成本（美元）
        """
        acquired = self._lock.acquire(blocking=False)
        if acquired:
            try:
                self.estimated_cost += cost
            finally:
                self._lock.release()

    def get_avg_step_time(self) -> float:
        """获取平均步骤执行时间（秒）"""
        return self.total_time / self.total_steps if self.total_steps > 0 else 0.0
    
    def get_total_tokens(self) -> int:
        """获取总 token 数"""
        return self.input_tokens + self.output_tokens
    
    def get_error_rate(self) -> float:
        """获取错误率（0.0 - 1.0）"""
        return self.errors / self.total_steps if self.total_steps > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        返回:
            包含所有统计信息的字典
        """
        return {
            "total_steps": self.total_steps,
            "total_time": round(self.total_time, 3),
            "avg_step_time": round(self.get_avg_step_time(), 3),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.get_total_tokens(),
            "tool_calls": self.tool_calls,
            "errors": self.errors,
            "error_rate": round(self.get_error_rate(), 4),
            "estimated_cost": round(self.estimated_cost, 6),
        }


# ====================== 认知引擎基类 ======================

class CognitiveEngine(ABC):
    """
    认知引擎抽象基类
    
    定义 Agent 的核心推理流程。所有具体引擎实现（ReAct、Chain、Tree 等）
    都应该继承此类并实现抽象方法。
    
    核心方法（必须实现）:
        step(): 执行推理步骤，返回 AgentOutput
    
    可选方法（按需覆盖）:
        step_stream(): 流式推理，返回事件流
        step_structured(): 结构化输出推理
    
    Hook 方法（扩展点）:
        before_step(): 步骤执行前调用
        after_step(): 步骤执行后调用
        on_error(): 错误发生时调用
    
    生命周期方法:
        initialize(): 初始化引擎资源
        cleanup(): 清理引擎资源
    
    使用示例:
        ```python
        class MyEngine(CognitiveEngine):
            async def step(self, input_messages: List[Message]) -> AgentOutput:
                response = await self.model.acompletion(
                    messages=[m.to_openai_format() for m in input_messages]
                )
                return AgentOutput(content=response.choices[0].message["content"])
        
        # 使用上下文管理器确保资源清理
        async with MyEngine(model, toolbox, memory) as engine:
            output = await engine.step([Message.user("你好")])
        ```
    
    属性:
        model: 语言模型实例（必须实现 ModelProtocol）
        toolbox: 工具箱，包含可调用的工具
        memory: 记忆管理器
        event_bus: 事件总线（可选），用于发布引擎事件
        max_iterations: 最大迭代次数，防止死循环
        stats: 执行统计对象（如果启用）
    """
    
    def __init__(
        self,
        model: ModelProtocol,
        toolbox: ToolBox,
        memory: TokenMemory,
        event_bus: Optional[EventBus] = None,
        max_iterations: int = 10,
        enable_stats: bool = True,
        **kwargs: Any
    ):
        """
        初始化认知引擎
        
        参数:
            model: 语言模型实例，必须实现 ModelProtocol 协议
            toolbox: 工具箱实例
            memory: 记忆管理器实例
            event_bus: 事件总线（可选），用于发布引擎运行事件
            max_iterations: 最大迭代次数，默认 10，防止无限循环
            enable_stats: 是否启用执行统计，默认 True
            **kwargs: 额外配置参数，存储在 _config 中供子类使用
        
        异常:
            TypeError: 如果 model 不符合 ModelProtocol 协议
        
        配置项（通过 kwargs 传入）:
            hooks_fail_fast: bool - Hook 出错时是否立即终止，默认 False
        """
        # 验证模型是否符合协议（鸭子类型检查）
        validate_model(model)
        self.model = model
        
        self.toolbox = toolbox
        self.memory = memory
        self.event_bus = event_bus
        self.max_iterations = max_iterations
        self.enable_stats = enable_stats
        
        # 初始化统计对象
        self.stats = ExecutionStats() if enable_stats else None
        
        # Hook 函数槽位（可由子类或外部设置）
        self.before_step_hook: Optional[Callable] = None
        self.after_step_hook: Optional[Callable] = None
        self.on_error_hook: Optional[Callable] = None
        
        # 存储额外配置
        self._config: Dict[str, Any] = kwargs
        
        # Hook 失败时是否快速失败（默认 False，静默处理）
        self.hooks_fail_fast: bool = bool(kwargs.get("hooks_fail_fast", False))
        
        logger.debug(
            "引擎初始化完成",
            engine=self.__class__.__name__,
            model=type(model).__name__,
            max_iterations=max_iterations,
            event_bus_enabled=event_bus is not None
        )
    
    # ====================== 核心抽象方法 ======================
    
    @abstractmethod
    async def step(
        self, 
        input_messages: List[Message],
        **kwargs: Any
    ) -> AgentOutput:
        """
        执行推理步骤（必须实现）
        
        这是引擎的核心方法，定义了如何处理输入消息并生成输出。
        子类必须实现此方法。
        
        参数:
            input_messages: 输入消息列表，至少包含一条消息
            **kwargs: 额外参数，如 temperature, max_tokens 等
        
        返回:
            AgentOutput: 执行结果，包含内容、工具调用等信息
        
        异常:
            AgentError: 执行失败
            ModelError: 模型调用失败
            ValueError: 输入无效
        
        实现指南:
            1. 调用 validate_input() 验证输入
            2. 调用 before_step() Hook
            3. 执行推理逻辑（调用模型、处理工具等）
            4. 调用 after_step() Hook
            5. 返回 AgentOutput
        """
        pass
    
    # ====================== 可选方法（按需覆盖）======================
    
    async def step_stream(
        self, 
        input_messages: List[Message],
        **kwargs: Any
    ) -> AsyncIterator[AgentStreamEvent]:
        """
        流式推理（可选实现）
        
        统一契约：输出 AgentStreamEvent 事件流，而非纯文本 token。
        
        设计原因：
        - ReAct 等复杂引擎需要输出结构化事件（tool_input/tool_output/result/error）
        - 工业级系统（WebSocket/SSE/调试面板）需要事件流而非仅 token
        - 与 Agent.stream / Agent.stream_events 保持一致
        
        参数:
            input_messages: 输入消息列表
            **kwargs: 额外参数
        
        返回:
            AsyncIterator[AgentStreamEvent]: 事件流
        
        异常:
            NotImplementedError: 引擎不支持流式输出时抛出
        
        事件类型:
            - token: 实时生成的文本片段
            - tool_input: 工具调用意图和参数
            - tool_output: 工具执行结果
            - result: 最终生成的回复
            - error: 执行过程中的错误
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 不支持流式输出。"
            f"请重写 step_stream() 方法以启用此功能。"
        )
    
    async def step_text_stream(
        self,
        input_messages: List[Message],
        **kwargs: Any
    ) -> AsyncIterator[str]:
        """
        纯文本 token 流（兼容层）
        
        从 step_stream 事件流中过滤出 token 事件并 yield 文本内容。
        这提供了一个简化的接口，仅返回文本流。
        
        参数:
            input_messages: 输入消息列表
            **kwargs: 额外参数
        
        返回:
            AsyncIterator[str]: 文本 token 流
        
        使用场景:
            - 简单的流式文本展示
            - 不需要工具调用事件的场景
        """
        async for event in self.step_stream(input_messages, **kwargs): # type: ignore
            if event.type == "token" and event.content is not None:
                yield str(event.content)
    
    async def step_structured(
        self,
        input_messages: List[Message],
        response_model: Type[T],
        **kwargs: Any
    ) -> T:
        """
        结构化输出推理（可选实现）
        
        执行推理并将输出解析为指定的 Pydantic 模型实例。
        
        参数:
            input_messages: 输入消息列表
            response_model: 目标 Pydantic 模型类
            **kwargs: 额外参数
        
        返回:
            T: 解析后的模型实例
        
        异常:
            NotImplementedError: 引擎不支持结构化输出时抛出
            AgentError: 解析失败
        
        使用示例:
            ```python
            from pydantic import BaseModel
            
            class Answer(BaseModel):
                question: str
                answer: str
                confidence: float
            
            result = await engine.step_structured(
                input_messages=[Message.user("什么是人工智能？")],
                response_model=Answer
            )
            print(result.answer)
            ```
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 不支持结构化输出。"
            f"请重写 step_structured() 方法以启用此功能。"
        )
    
    # ====================== Hook 方法 ======================
    
    async def before_step(
        self, 
        input_messages: List[Message],
        **kwargs: Any
    ) -> None:
        """
        步骤执行前的 Hook
        
        在推理开始前调用，可用于：
        - 日志记录
        - 输入验证和预处理
        - 状态初始化
        - 发送开始事件
        
        参数:
            input_messages: 输入消息列表
            **kwargs: 额外参数
        
        注意:
            - 此方法不应修改输入消息
            - Hook 异常默认被捕获并记录，不影响主流程
            - 设置 hooks_fail_fast=True 可让 Hook 异常终止执行
        """
        # 发布事件到 EventBus（如果配置了）
        await self._publish_event("step_started", {
            "message_count": len(input_messages),
            "engine": self.__class__.__name__
        })
        
        # 执行用户自定义 Hook
        if self.before_step_hook:
            try:
                if asyncio.iscoroutinefunction(self.before_step_hook):
                    await self.before_step_hook(input_messages, **kwargs)
                else:
                    self.before_step_hook(input_messages, **kwargs)
            except Exception as e:
                logger.warning("before_step_hook 执行失败", error=str(e), exc_info=True)
                if self.hooks_fail_fast:
                    raise
    
    async def after_step(
        self,
        input_messages: List[Message],
        output: AgentOutput,
        **kwargs: Any
    ) -> None:
        """
        步骤执行后的 Hook
        
        在推理完成后调用，可用于：
        - 日志记录
        - 结果验证和后处理
        - 统计更新
        - 发送完成事件
        
        参数:
            input_messages: 输入消息列表
            output: 执行结果
            **kwargs: 额外参数
        """
        # 发布事件到 EventBus
        await self._publish_event("step_completed", {
            "content_length": len(output.content) if output.content else 0,
            "has_tool_calls": bool(output.tool_calls),
            "engine": self.__class__.__name__
        })
        
        # 执行用户自定义 Hook
        if self.after_step_hook:
            try:
                if asyncio.iscoroutinefunction(self.after_step_hook):
                    await self.after_step_hook(input_messages, output, **kwargs)
                else:
                    self.after_step_hook(input_messages, output, **kwargs)
            except Exception as e:
                logger.warning("after_step_hook 执行失败", error=str(e), exc_info=True)
                if self.hooks_fail_fast:
                    raise
    
    async def on_error(
        self,
        error: Exception,
        input_messages: List[Message],
        **kwargs: Any
    ) -> None:
        """
        错误处理 Hook
        
        在推理过程中发生错误时调用，可用于：
        - 错误日志记录
        - 错误恢复尝试
        - 降级处理
        - 发送告警通知
        
        参数:
            error: 异常对象
            input_messages: 输入消息列表
            **kwargs: 额外参数
        """
        # 发布错误事件到 EventBus
        await self._publish_event("step_error", {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "engine": self.__class__.__name__
        })
        
        # 执行用户自定义 Hook
        if self.on_error_hook:
            try:
                if asyncio.iscoroutinefunction(self.on_error_hook):
                    await self.on_error_hook(error, input_messages, **kwargs)
                else:
                    self.on_error_hook(error, input_messages, **kwargs)
            except Exception as e:
                logger.error("on_error_hook 执行失败", error=str(e), exc_info=True)
                if self.hooks_fail_fast:
                    raise

    # ====================== 事件发布 ======================

    async def _publish_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        发布事件到 EventBus
        
        如果配置了 event_bus，则发布事件；否则静默忽略。
        事件发布失败不会影响主流程。
        
        参数:
            event_type: 事件类型（如 "step_started", "step_completed"）
            data: 事件数据
        """
        if self.event_bus is None:
            return
        
        try:
            # EventBus 可能是同步或异步的
            if hasattr(self.event_bus, 'publish'):
                publish_method = self.event_bus.publish
                if asyncio.iscoroutinefunction(publish_method):
                    await publish_method(event_type, data) # type: ignore
                else:
                    publish_method(event_type, data) # type: ignore
        except Exception as e:
            # 事件发布失败不应影响主流程
            logger.debug(f"事件发布失败: {event_type}", error=str(e))

    # ====================== 统计辅助方法 ======================
    
    def record_step(
        self, 
        duration: float, 
        input_tokens: int = 0, 
        output_tokens: int = 0, 
        had_error: bool = False
    ) -> None:
        """
        记录一次执行步骤的统计信息
        
        供子类在合适的位置调用，统一更新统计数据。
        
        参数:
            duration: 执行耗时（秒）
            input_tokens: 输入 token 数
            output_tokens: 输出 token 数
            had_error: 是否发生错误
        """
        if self.stats is not None:
            try:
                self.stats.add_step(
                    duration, 
                    input_tokens=input_tokens, 
                    output_tokens=output_tokens, 
                    had_error=had_error
                )
            except Exception:
                logger.debug("更新步骤统计失败")

    def record_tool_call(self) -> None:
        """记录一次工具调用"""
        if self.stats is not None:
            try:
                self.stats.add_tool_call()
            except Exception:
                logger.debug("更新工具调用统计失败")

    def record_cost(
        self, 
        input_tokens: int = 0, 
        output_tokens: int = 0, 
        model_name: str = ""
    ) -> None:
        """
        基于 token 数和模型名称记录估算成本
        
        参数:
            input_tokens: 输入 token 数
            output_tokens: 输出 token 数
            model_name: 模型名称，用于查询定价表
        """
        if self.stats is None:
            return
        
        # 获取模型定价（默认使用 gpt-3.5-turbo 的价格）
        pricing = MODEL_PRICING.get(model_name)
        if not pricing:
            # 尝试模糊匹配（如 "gpt-4-0125-preview" 匹配 "gpt-4"）
            for key in MODEL_PRICING:
                if model_name.startswith(key):
                    pricing = MODEL_PRICING[key]
                    break
            if not pricing:
                pricing = MODEL_PRICING.get("gpt-3.5-turbo", {"input": 0.5, "output": 1.5})
        
        # 计算成本：tokens * (price_per_million / 1_000_000)
        cost = (
            input_tokens * pricing["input"] / 1_000_000 +
            output_tokens * pricing["output"] / 1_000_000
        )
        
        try:
            self.stats.add_cost(cost)
        except Exception:
            logger.debug("记录成本失败", tokens=(input_tokens, output_tokens))
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """
        获取执行统计摘要
        
        返回:
            统计信息字典，如果未启用统计则返回空字典
        """
        if self.stats is None:
            return {}
        return self.stats.to_dict()
    
    def get_stats(self) -> Optional[Dict[str, Any]]:
        """
        获取执行统计（兼容旧接口）
        
        返回:
            统计信息字典，如果未启用统计则返回 None
        """
        return self.stats.to_dict() if self.stats else None
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        if self.stats:
            self.stats = ExecutionStats()
            logger.debug("统计信息已重置")
    
    # ====================== 工具方法 ======================
    
    def validate_input(self, input_messages: List[Message]) -> None:
        """
        验证输入消息
        
        参数:
            input_messages: 输入消息列表
        
        异常:
            ValueError: 输入为空
            TypeError: 输入类型错误
        """
        if not input_messages:
            raise ValueError("input_messages 不能为空")
        
        if not all(isinstance(m, Message) for m in input_messages):
            raise TypeError("所有输入必须是 Message 实例")
        
        logger.debug("输入验证通过", message_count=len(input_messages))
    
    def supports_streaming(self) -> bool:
        """
        检查引擎是否支持流式输出
        
        返回:
            bool: 是否支持流式输出
        """
        # 检查模型是否支持流式
        model_supports = supports_streaming(self.model)
        
        # 检查引擎是否重写了 step_stream 方法
        engine_supports = (
            self.__class__.step_stream != CognitiveEngine.step_stream
        )
        
        return model_supports and engine_supports
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """
        获取配置项
        
        参数:
            key: 配置键
            default: 默认值
        
        返回:
            配置值
        """
        return self._config.get(key, default)
    
    def set_config(self, key: str, value: Any) -> None:
        """
        设置配置项
        
        参数:
            key: 配置键
            value: 配置值
        """
        self._config[key] = value
    
    # ====================== 生命周期管理 ======================
    
    async def initialize(self) -> None:
        """
        初始化引擎
        
        在首次使用前调用，可用于：
        - 加载外部资源
        - 预热模型连接
        - 初始化缓存
        
        子类可以重写此方法添加自定义初始化逻辑。
        """
        logger.debug("引擎初始化", engine=self.__class__.__name__)
    
    async def cleanup(self) -> None:
        """
        清理资源
        
        在引擎不再使用时调用，可用于：
        - 关闭连接
        - 释放资源
        - 保存状态
        
        子类可以重写此方法添加自定义清理逻辑。
        """
        logger.debug("引擎清理", engine=self.__class__.__name__)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查接口
        
        用于监控系统检测引擎状态。
        
        返回:
            包含健康状态信息的字典
        """
        return {
            "engine": self.__class__.__name__,
            "model": type(self.model).__name__,
            "supports_streaming": self.supports_streaming(),
            "stats": self.get_stats_summary(),
            "status": "healthy"
        }
    
    # ====================== 上下文管理器 ======================
    
    async def __aenter__(self) -> "CognitiveEngine":
        """异步上下文管理器入口"""
        await self.initialize()
        return self
    
    async def __aexit__(
        self, 
        exc_type: Optional[type], 
        exc_val: Optional[BaseException], 
        exc_tb: Any
    ) -> bool:
        """异步上下文管理器出口"""
        await self.cleanup()
        return False  # 不抑制异常
    
    # ====================== 辅助方法 ======================
    
    async def _safe_execute(
        self,
        func: Callable,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """
        安全执行函数（带错误处理和统计）
        
        包装函数执行，自动处理：
        - 执行时间统计
        - 异常捕获和记录
        - Hook 调用
        
        参数:
            func: 要执行的函数
            *args: 位置参数
            **kwargs: 关键字参数
        
        返回:
            函数执行结果
        
        异常:
            原始异常（已记录日志和统计）
        """
        start_time = time.time()
        had_error = False
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            return result
        
        except Exception as e:
            had_error = True
            
            # 记录错误统计
            if self.stats:
                self.stats.errors += 1
            
            # 调用错误 Hook
            await self.on_error(e, kwargs.get("input_messages", []))
            
            # 记录日志
            logger.exception(
                "引擎执行失败",
                engine=self.__class__.__name__,
                error=str(e)
            )
            
            raise
        
        finally:
            # 记录执行时间
            duration = time.time() - start_time
            if self.stats:
                self.stats.add_step(duration, had_error=had_error)
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"model={type(self.model).__name__}, "
            f"max_iterations={self.max_iterations}, "
            f"stats_enabled={self.stats is not None}"
            f")"
        )


# ====================== 工厂函数 ======================

def create_engine(
    engine_class: Type[CognitiveEngine],
    model: ModelProtocol,
    toolbox: ToolBox,
    memory: TokenMemory,
    **kwargs: Any
) -> CognitiveEngine:
    """
    创建引擎实例（工厂函数）
    
    提供统一的引擎创建接口，进行类型检查。
    
    参数:
        engine_class: 引擎类（必须是 CognitiveEngine 的子类）
        model: 模型实例
        toolbox: 工具箱实例
        memory: 记忆管理器实例
        **kwargs: 传递给引擎构造函数的额外参数
    
    返回:
        CognitiveEngine: 引擎实例
    
    异常:
        TypeError: engine_class 不是 CognitiveEngine 的子类
    
    使用示例:
        ```python
        engine = create_engine(
            ReActEngine,
            model=openai_model,
            toolbox=toolbox,
            memory=memory,
            max_iterations=5
        )
        ```
    """
    if not issubclass(engine_class, CognitiveEngine):
        raise TypeError(
            f"engine_class 必须是 CognitiveEngine 的子类，"
            f"收到: {engine_class.__name__}"
        )
    
    return engine_class(
        model=model,
        toolbox=toolbox,
        memory=memory,
        **kwargs
    )


# ====================== 模块导出 ======================

__all__ = [
    "CognitiveEngine",
    "ExecutionStats",
    "create_engine",
    "MODEL_PRICING",
    "load_model_pricing",
]