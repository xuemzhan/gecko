# gecko/core/engine/base.py
"""
认知引擎基类

定义 Agent 的推理和执行流程，所有引擎实现（ReAct、Chain、Tree 等）
都应继承此基类。

核心概念：
- CognitiveEngine: 抽象基类，定义引擎接口
- 支持普通推理和流式推理
- 支持结构化输出
- 提供 Hook 机制
- 统一的错误处理

优化点：
1. 强化类型注解（使用 ModelProtocol）
2. 完善抽象方法（step, step_stream, step_structured）
3. 添加 Hook 机制（before_step, after_step）
4. 提供工具方法（validate_input, log_execution）
5. 支持上下文管理器（资源管理）
"""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from gecko.core.events.bus import EventBus
from gecko.core.exceptions import AgentError, ModelError
from gecko.core.logging import get_logger
from gecko.core.memory import TokenMemory
from gecko.core.message import Message
from gecko.core.output import AgentOutput
from gecko.core.protocols import ModelProtocol, supports_streaming, validate_model
from gecko.core.toolbox import ToolBox

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

# [P3 增强] 模型定价配置（单位：USD per token）
MODEL_PRICING = {
    # OpenAI pricing (USD per 1M tokens, as of 2024)
    "gpt-4": {"input": 30.0 / 1_000_000, "output": 60.0 / 1_000_000},
    "gpt-4-turbo": {"input": 10.0 / 1_000_000, "output": 30.0 / 1_000_000},
    "gpt-3.5-turbo": {"input": 0.5 / 1_000_000, "output": 1.5 / 1_000_000},
    "claude-3-opus": {"input": 15.0 / 1_000_000, "output": 75.0 / 1_000_000},
    "claude-3-sonnet": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
    "claude-3-haiku": {"input": 0.25 / 1_000_000, "output": 1.25 / 1_000_000},
}


# ====================== 执行统计 ======================

class ExecutionStats(BaseModel):
    """
    引擎执行统计
    
    用于性能监控和调试。支持 token 成本跟踪和模型定价。
    """
    total_steps: int = 0
    total_time: float = 0.0  # 总执行时间（秒）
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    errors: int = 0
    
    # [P3 增强] 成本跟踪
    estimated_cost: float = 0.0  # 估算成本（单位：美元）
    
    def add_step(self, duration: float, input_tokens: int = 0, output_tokens: int = 0, had_error: bool = False):
        """记录一次步骤执行"""
        self.total_steps += 1
        self.total_time += duration
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if had_error:
            self.errors += 1
    
    def add_tool_call(self):
        """记录一次工具调用"""
        self.tool_calls += 1
    
    def add_cost(self, cost: float):
        """累加成本估算（单位：美元）"""
        self.estimated_cost += cost
    
    def get_avg_step_time(self) -> float:
        """获取平均步骤时间（秒）"""
        return self.total_time / self.total_steps if self.total_steps > 0 else 0.0
    
    def get_total_tokens(self) -> int:
        """获取总 token 数"""
        return self.input_tokens + self.output_tokens
    
    def get_error_rate(self) -> float:
        """获取错误率（0.0 - 1.0）"""
        return self.errors / self.total_steps if self.total_steps > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total_steps": self.total_steps,
            "total_time": self.total_time,
            "avg_step_time": self.get_avg_step_time(),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.get_total_tokens(),
            "tool_calls": self.tool_calls,
            "errors": self.errors,
            "error_rate": self.get_error_rate(),
            "estimated_cost": self.estimated_cost,
        }


# ====================== 认知引擎基类 ======================

class CognitiveEngine(ABC):
    """
    认知引擎抽象基类
    
    定义 Agent 的核心推理流程，所有具体引擎实现（ReAct、Chain、Tree 等）
    都应该继承此类。
    
    核心方法：
    - step(): 单次/多轮推理（必需）
    - step_stream(): 流式推理（可选）
    - step_structured(): 结构化输出（可选）
    
    Hook 方法：
    - before_step(): 步骤执行前
    - after_step(): 步骤执行后
    - on_error(): 错误处理
    
    生命周期：
    - initialize(): 初始化
    - cleanup(): 清理资源
    
    示例:
        ```python
        class MyEngine(CognitiveEngine):
            async def step(self, input_messages: List[Message]) -> AgentOutput:
                # 实现推理逻辑
                response = await self.model.acompletion(
                    messages=[m.to_openai_format() for m in input_messages]
                )
                return AgentOutput(content=response.choices[0].message["content"])
        
        # 使用
        engine = MyEngine(model=model, toolbox=toolbox, memory=memory)
        output = await engine.step([Message.user("Hello")])
        ```
    """
    
    def __init__(
        self,
        model: ModelProtocol,
        toolbox: ToolBox,
        memory: TokenMemory,
        event_bus: Optional[EventBus] = None,
        max_iterations: int = 10,
        enable_stats: bool = True,
        **kwargs
    ):
        """
        初始化认知引擎
        
        参数:
            model: 语言模型（必须实现 ModelProtocol）
            toolbox: 工具箱
            memory: 记忆管理器
            max_iterations: 最大迭代次数（防止死循环）
            enable_stats: 是否启用统计
            **kwargs: 子类的额外参数
        
        异常:
            TypeError: model 不符合 ModelProtocol
        """
        # 验证模型（鸭子类型检查）
        # 如果缺少必要方法，会由 validate_model 抛出带有 Missing methods 提示的 TypeError
        validate_model(model)
        self.model = model
        
        self.toolbox = toolbox
        self.event_bus = event_bus
        self.memory = memory
        self.max_iterations = max_iterations
        self.enable_stats = enable_stats
        
        # 统计信息
        self.stats = ExecutionStats() if enable_stats else None
        
        # Hook 函数（可由子类或外部设置）
        self.before_step_hook: Optional[Callable] = None
        self.after_step_hook: Optional[Callable] = None
        self.on_error_hook: Optional[Callable] = None
        
        # 存储额外的配置
        self._config = kwargs
        # hooks 出错时是否立即 fail-fast（默认 False，作为 P3 优化可开启）
        self.hooks_fail_fast: bool = bool(kwargs.get("hooks_fail_fast", False))
        
        logger.debug(
            "Engine initialized",
            engine=self.__class__.__name__,
            model=type(model).__name__,
            max_iterations=max_iterations
        )
    
    # ====================== 核心抽象方法 ======================
    
    @abstractmethod
    async def step(
        self, 
        input_messages: List[Message],
        **kwargs
    ) -> AgentOutput:
        """
        执行推理步骤（必需实现）
        
        这是引擎的核心方法，定义了如何处理输入并生成输出。
        
        参数:
            input_messages: 输入消息列表
            **kwargs: 额外参数（如 temperature, max_tokens 等）
        
        返回:
            AgentOutput: 执行结果
        
        异常:
            AgentError: 执行失败
            ModelError: 模型调用失败
        
        实现指南:
            1. 验证输入
            2. 调用 before_step_hook（如果有）
            3. 执行推理逻辑
            4. 调用 after_step_hook（如果有）
            5. 返回结果
        
        示例:
            ```python
            async def step(self, input_messages: List[Message]) -> AgentOutput:
                # 转换为 OpenAI 格式
                messages = [m.to_openai_format() for m in input_messages]
                
                # 调用模型
                response = await self.model.acompletion(messages=messages)
                
                # 构建输出
                return AgentOutput(
                    content=response.choices[0].message["content"],
                    usage=response.usage
                )
            ```
        """
        pass
    
    # ====================== 可选方法 ======================
    
    async def step_stream(
        self, 
        input_messages: List[Message],
        **kwargs
    ) -> AsyncIterator[str]:
        """
        流式推理（可选实现）
        
        如果引擎支持流式输出，应该重写此方法。
        
        参数:
            input_messages: 输入消息列表
            **kwargs: 额外参数
        
        返回:
            AsyncIterator[str]: 文本流
        
        异常:
            NotImplementedError: 引擎不支持流式输出
        
        示例:
            ```python
            async def step_stream(self, input_messages: List[Message]):
                if not supports_streaming(self.model):
                    raise NotImplementedError("Model does not support streaming")
                
                messages = [m.to_openai_format() for m in input_messages]
                
                async for chunk in self.model.astream(messages=messages):
                    if chunk.content:
                        yield chunk.content
            ```
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support streaming. "
            f"Override step_stream() to enable this feature."
        )
    
    async def step_structured(
        self,
        input_messages: List[Message],
        response_model: Type[T],
        **kwargs
    ) -> T:
        """
        结构化输出推理（可选实现）
        
        执行推理并将输出解析为 Pydantic 模型。
        
        参数:
            input_messages: 输入消息列表
            response_model: 目标 Pydantic 模型类
            **kwargs: 额外参数
        
        返回:
            T: 解析后的模型实例
        
        异常:
            NotImplementedError: 引擎不支持结构化输出
        
        示例:
            ```python
            from pydantic import BaseModel
            
            class Answer(BaseModel):
                question: str
                answer: str
                confidence: float
            
            result = await engine.step_structured(
                input_messages=[Message.user("What is AI?")],
                response_model=Answer
            )
            print(result.answer)
            ```
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support structured output. "
            f"Override step_structured() to enable this feature."
        )
    
    # ====================== Hook 方法 ======================
    
    async def before_step(
        self, 
        input_messages: List[Message],
        **kwargs
    ) -> None:
        """
        步骤执行前的 Hook
        
        在推理开始前调用，可用于：
        - 日志记录
        - 输入验证
        - 状态初始化
        - 发送事件
        
        参数:
            input_messages: 输入消息列表
            **kwargs: 额外参数
        
        注意:
            此方法不应修改输入，如需修改请在子类中重写
        """
        if self.before_step_hook:
            try:
                if asyncio.iscoroutinefunction(self.before_step_hook):
                    await self.before_step_hook(input_messages, **kwargs)
                else:
                    self.before_step_hook(input_messages, **kwargs)
            except Exception as e:
                logger.warning("before_step_hook failed", error=str(e))
                if self.hooks_fail_fast:
                    raise
    
    async def after_step(
        self,
        input_messages: List[Message],
        output: AgentOutput,
        **kwargs
    ) -> None:
        """
        步骤执行后的 Hook
        
        在推理完成后调用，可用于：
        - 日志记录
        - 结果验证
        - 统计更新
        - 发送事件
        
        参数:
            input_messages: 输入消息列表
            output: 执行结果
            **kwargs: 额外参数
        """
        if self.after_step_hook:
            try:
                if asyncio.iscoroutinefunction(self.after_step_hook):
                    await self.after_step_hook(input_messages, output, **kwargs)
                else:
                    self.after_step_hook(input_messages, output, **kwargs)
            except Exception as e:
                logger.warning("after_step_hook failed", error=str(e))
                if self.hooks_fail_fast:
                    raise
    
    async def on_error(
        self,
        error: Exception,
        input_messages: List[Message],
        **kwargs
    ) -> None:
        """
        错误处理 Hook
        
        在推理过程中发生错误时调用，可用于：
        - 错误日志记录
        - 错误恢复
        - 降级处理
        - 发送告警
        
        参数:
            error: 异常对象
            input_messages: 输入消息列表
            **kwargs: 额外参数
        """
        if self.on_error_hook:
            try:
                if asyncio.iscoroutinefunction(self.on_error_hook):
                    await self.on_error_hook(error, input_messages, **kwargs)
                else:
                    self.on_error_hook(error, input_messages, **kwargs)
            except Exception as e:
                logger.error("on_error_hook failed", error=str(e))
                if self.hooks_fail_fast:
                    raise

    # ===== 轻量统计/指标辅助 =====
    def record_step(self, duration: float, tokens: int = 0, had_error: bool = False) -> None:
        """记录一次执行步骤的轻量统计（供子类在合适位置调用）。"""
        if self.stats is not None:
            try:
                self.stats.add_step(duration, tokens=tokens, had_error=had_error)
            except Exception:
                logger.debug("Failed to update stats")

    def record_tool_call(self) -> None:
        """记录一次工具调用次数"""
        if self.stats is not None:
            try:
                self.stats.add_tool_call()
            except Exception:
                logger.debug("Failed to increment tool call stat")

    # ===== 成本与定价辅助 =====
    def record_cost(self, input_tokens: int = 0, output_tokens: int = 0, model_name: str = "") -> None:
        """基于 token 数和模型名称记录估算成本。"""
        if self.stats is None:
            return
        
        # 获取模型定价（默认使用 gpt-3.5-turbo 价格如果未找到）
        pricing = MODEL_PRICING.get(model_name, MODEL_PRICING.get("gpt-3.5-turbo"))
        
        if pricing:
            cost = (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])
            try:
                self.stats.add_cost(cost)
            except Exception:
                logger.debug("Failed to record cost")
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """获取执行统计摘要"""
        if self.stats is None:
            return {}
        return self.stats.to_dict()
    
    # ====================== 工具方法 ======================
    
    def validate_input(self, input_messages: List[Message]) -> None:
        """
        验证输入消息
        
        参数:
            input_messages: 输入消息列表
        
        异常:
            ValueError: 输入无效
        """
        if not input_messages:
            raise ValueError("input_messages 不能为空")
        
        if not all(isinstance(m, Message) for m in input_messages):
            raise TypeError("所有输入必须是 Message 实例")
        
        logger.debug("Input validated", message_count=len(input_messages))
    
    def supports_streaming(self) -> bool:
        """
        检查引擎是否支持流式输出
        
        返回:
            是否支持
        """
        # 检查模型能力
        model_supports = supports_streaming(self.model)
        
        # 检查引擎是否重写了 step_stream
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
    
    def get_stats(self) -> Optional[Dict[str, Any]]:
        """
        获取执行统计
        
        返回:
            统计信息字典，如果未启用统计则返回 None
        """
        return self.stats.to_dict() if self.stats else None
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        if self.stats:
            self.stats = ExecutionStats()
            logger.debug("Stats reset")
    
    # ====================== 生命周期管理 ======================
    
    async def initialize(self) -> None:
        """
        初始化引擎
        
        在首次使用前调用，可用于：
        - 加载资源
        - 预热模型
        - 初始化连接
        
        子类可以重写此方法以添加自定义初始化逻辑。
        """
        logger.debug("Engine initialized", engine=self.__class__.__name__)
    
    async def cleanup(self) -> None:
        """
        清理资源
        
        在引擎不再使用时调用，可用于：
        - 关闭连接
        - 释放资源
        - 保存状态
        
        子类可以重写此方法以添加自定义清理逻辑。
        """
        logger.debug("Engine cleanup", engine=self.__class__.__name__)
    
    # ====================== 上下文管理器 ======================
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.cleanup()
        return False
    
    # ====================== 辅助方法 ======================
    
    async def _safe_execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """
        安全执行函数（带错误处理和统计）
        
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
            
            # 记录统计
            if self.stats:
                self.stats.errors += 1
            
            # 调用错误 Hook
            await self.on_error(e, kwargs.get("input_messages", []))
            
            # 记录日志
            logger.exception(
                "Engine execution failed",
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
            f"max_iterations={self.max_iterations}"
            f")"
        )


# ====================== 工具函数 ======================

def create_engine(
    engine_class: Type[CognitiveEngine],
    model: ModelProtocol,
    toolbox: ToolBox,
    memory: TokenMemory,
    **kwargs
) -> CognitiveEngine:
    """
    创建引擎实例（工厂函数）
    
    参数:
        engine_class: 引擎类
        model: 模型
        toolbox: 工具箱
        memory: 记忆
        **kwargs: 额外参数
    
    返回:
        引擎实例
    
    示例:
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


# ====================== 导出 ======================

__all__ = [
    "CognitiveEngine",
    "ExecutionStats",
    "create_engine",
]