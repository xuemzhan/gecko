# gecko/core/engine/react.py
"""
ReAct (Reasoning + Acting) 引擎模块
====================================

这个模块实现了 ReAct 范式的认知引擎，结合了推理（Reasoning）和行动（Acting）。

ReAct 工作流程：
1. Think（思考）：调用 LLM 分析当前情况，决定下一步行动
2. Act（行动）：如果需要，执行工具调用获取信息
3. Observe（观察）：分析工具执行结果，检测错误和循环
4. 重复上述流程，直到任务完成或达到最大轮次

主要组件：
- ReActConfig: 配置类，定义反思、错误阈值等参数
- ExecutionContext: 执行上下文，管理消息历史和元数据
- ReActEngine: 主引擎类，实现完整的 ReAct 循环

特色功能：
- 智能循环检测（避免重复相同的工具调用）
- 错误恢复机制（自动反思和重试）
- 上下文管理（自动裁剪历史消息）
- 流式输出支持
- 结构化输出支持
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import random
import time
import uuid
from collections import deque
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set, Type, TypeVar, Union, cast

from pydantic import BaseModel

from gecko.config import get_settings
from gecko.core.engine.base import CognitiveEngine, get_pricing_for_model
from gecko.core.engine.buffer import StreamBuffer
from gecko.core.events.types import AgentStreamEvent
from gecko.core.exceptions import AgentError
from gecko.core.logging import get_logger
from gecko.core.memory import TokenMemory
from gecko.core.message import Message
from gecko.core.output import AgentOutput
from gecko.core.prompt import PromptTemplate
from gecko.core.structure import StructureEngine
from gecko.core.toolbox import ToolBox, ToolExecutionResult
from gecko.core.utils import ensure_awaitable

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

# ============================================================================
# 常量定义
# ============================================================================
# 结构化输出工具的名称前缀（内部使用）
STRUCTURE_TOOL_PREFIX = "__gecko_structured_output_"

# 默认的 ReAct 系统提示模板
DEFAULT_REACT_TEMPLATE = """You are a helpful AI assistant.
Current Time: {{ current_time }}

Available Tools:
{% for tool in tools %}
- {{ tool['function']['name'] }}: {{ tool['function']['description'] }}
{% endfor %}

Answer the user's request. Use tools if necessary.
If you use a tool, just output the tool call format.
"""

# 工具索引最大间隔（检测异常跳跃）
MAX_TOOL_INDEX_GAP = 500
# 重试最大延迟时间（秒）
MAX_RETRY_DELAY_SECONDS = 5.0
# 默认最大历史消息数
DEFAULT_MAX_HISTORY = 50
# 默认最大上下文字符数
DEFAULT_MAX_CONTEXT_CHARS = 100_000
# 工具哈希队列大小（用于循环检测）
TOOL_HASH_DEQUE_SIZE = 5


# ============================================================================
# 配置类
# ============================================================================
class ReActConfig(BaseModel):
    """
    ReAct 引擎配置
    
    定义引擎行为的关键参数，包括反思、错误处理和循环检测。
    """
    max_reflections: int = 2           # 最大反思次数（遇到错误时）
    tool_error_threshold: int = 3      # 连续工具错误阈值
    loop_repeat_threshold: int = 2     # 循环重复阈值（相同工具调用的次数）
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS  # 最大上下文字符数


# ============================================================================
# 执行上下文类
# ============================================================================
class ExecutionContext:
    """
    执行上下文管理器
    
    管理 ReAct 循环的状态和历史记录：
    - 消息历史（带自动裁剪）
    - 元数据存储
    - 错误计数
    - 循环检测数据
    
    使用 __slots__ 优化内存
    """
    __slots__ = (
        "messages",              # 消息列表
        "max_history",           # 最大历史消息数
        "max_chars",             # 最大上下文字符数
        "turn",                  # 当前轮次
        "metadata",              # 通用元数据字典
        "consecutive_errors",    # 连续错误计数
        "reflection_attempts",   # 反思尝试次数
        "last_tool_hash",        # 最后一次工具调用的哈希
        "last_tool_hashes",      # 最近的工具调用哈希队列
        "message_metadata",      # 消息级别的元数据
        "_msg_lengths_cache"     # 消息长度缓存（优化性能）
    )

    def __init__(
        self,
        messages: List[Message],
        max_history: int = DEFAULT_MAX_HISTORY,
        max_chars: Optional[int] = None,
    ):
        """
        初始化执行上下文
        
        参数:
            messages: 初始消息列表
            max_history: 最大保留消息数
            max_chars: 最大上下文字符数
        """
        self.messages: List[Message] = list(messages)
        self.max_history: int = max_history
        self.max_chars: int = max_chars or DEFAULT_MAX_CONTEXT_CHARS
        self.turn: int = 0
        self.metadata: Dict[str, Any] = {}

        # 错误跟踪
        self.consecutive_errors: int = 0
        self.reflection_attempts: int = 0

        # 循环检测
        self.last_tool_hash: Optional[str] = None
        self.last_tool_hashes: deque = deque(maxlen=TOOL_HASH_DEQUE_SIZE)

        # 元数据存储
        self.message_metadata: Dict[str, Dict[str, Any]] = {}
        self._msg_lengths_cache: Dict[int, int] = {}

    def add_message(self, message: Message) -> None:
        """
        添加消息到上下文
        
        自动处理：
        - 长度缓存
        - 历史裁剪（按数量）
        - 大小裁剪（按字符数）
        
        参数:
            message: 要添加的消息
        """
        self.messages.append(message)
        # 缓存消息长度，避免重复计算
        self._msg_lengths_cache[id(message)] = self._get_message_length(message)

        # 检查数量限制
        if len(self.messages) > self.max_history:
            self._trim_context()
            return

        # 检查大小限制
        current_chars = self._calculate_total_chars()

        if current_chars > self.max_chars:
            logger.warning(
                "Context size exceeded limit, trimming",
                current_chars=current_chars,
                limit=self.max_chars,
            )
            self._trim_context(target_chars=self.max_chars)

    def add_message_with_metadata(
        self,
        message: Message,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        添加消息并附加元数据
        
        为消息生成唯一 ID，并存储关联的元数据。
        
        参数:
            message: 要添加的消息
            metadata: 关联的元数据
            
        返回:
            生成的消息 ID
        """
        msg_id = str(uuid.uuid4())
        self.messages.append(message)
        self._msg_lengths_cache[id(message)] = self._get_message_length(message)

        if metadata:
            self.message_metadata[msg_id] = metadata

        # 尝试将 ID 附加到消息对象（可能失败，但不影响功能）
        try:
            object.__setattr__(message, "_gecko_msg_id", msg_id)
        except Exception:
            pass

        # 检查是否需要裁剪
        if len(self.messages) > self.max_history:
            self._trim_context()

        return msg_id

    def _calculate_total_chars(self) -> int:
        """
        计算总字符数
        
        使用缓存优化性能，避免重复计算相同消息的长度。
        
        返回:
            所有消息的总字符数
        """
        total = 0
        for m in self.messages:
            msg_id = id(m)
            if msg_id in self._msg_lengths_cache:
                total += self._msg_lengths_cache[msg_id]
            else:
                length = self._get_message_length(m)
                self._msg_lengths_cache[msg_id] = length
                total += length
        return total

    def _trim_context(self, target_chars: Optional[int] = None) -> None:
        """
        裁剪上下文
        
        策略：
        1. 保留所有系统消息（通常包含重要的指令）
        2. 裁剪对话消息，优先删除旧消息
        3. 保持工具调用链的完整性（assistant + tool 消息成对删除）
        
        参数:
            target_chars: 目标字符数，None 表示按数量裁剪
        """
        # 分离系统消息和对话消息
        system_msgs = [m for m in self.messages if m.role == "system"]
        conversation_msgs = [m for m in self.messages if m.role != "system"]

        if not conversation_msgs:
            self.messages = system_msgs
            return

        # 根据目标选择裁剪策略
        if target_chars is None:
            self._trim_by_count(system_msgs, conversation_msgs)
        else:
            self._trim_by_size(system_msgs, conversation_msgs, target_chars)

    def _trim_by_count(self, system_msgs: List[Message], conversation_msgs: List[Message]) -> None:
        """
        按数量裁剪上下文
        
        保留最近的 N 条消息，删除旧消息。
        确保工具调用链的完整性。
        
        参数:
            system_msgs: 系统消息列表
            conversation_msgs: 对话消息列表
        """
        # 计算需要保留的对话消息数量
        keep_count = max(1, self.max_history - len(system_msgs))
        
        remove_count = len(conversation_msgs) - keep_count
        if remove_count <= 0:
            self.messages = system_msgs + conversation_msgs
            return

        # 标记要删除的消息索引
        remove_indices: Set[int] = set()
        i = 0
        
        # 从前往后删除消息，直到达到目标数量
        while len(remove_indices) < remove_count and i < len(conversation_msgs):
            if i in remove_indices:
                i += 1
                continue
                
            msg = conversation_msgs[i]
            
            # 如果是带工具调用的 assistant 消息，需要找到相关的 tool 消息一起删除
            if self._is_assistant_with_tools(msg):
                tool_ids = self._extract_tool_ids(msg)
                chain_indices = self._find_tool_chain(conversation_msgs, i, tool_ids)
                for idx in chain_indices:
                    remove_indices.add(idx)
            else:
                remove_indices.add(i)
            
            i += 1

        # 保留未被标记删除的消息
        remaining = [m for idx, m in enumerate(conversation_msgs) if idx not in remove_indices]
        self.messages = system_msgs + remaining

    def _trim_by_size(
        self, 
        system_msgs: List[Message], 
        conversation_msgs: List[Message], 
        target_chars: int
    ) -> None:
        """
        按大小裁剪上下文
        
        删除旧消息，直到总字符数低于目标值。
        确保工具调用链的完整性。
        
        参数:
            system_msgs: 系统消息列表
            conversation_msgs: 对话消息列表
            target_chars: 目标字符数
        """
        # 预计算所有消息的长度
        msg_lengths = []
        for m in conversation_msgs:
            msg_id = id(m)
            if msg_id in self._msg_lengths_cache:
                msg_lengths.append(self._msg_lengths_cache[msg_id])
            else:
                length = self._get_message_length(m)
                self._msg_lengths_cache[msg_id] = length
                msg_lengths.append(length)

        # 计算当前总长度
        system_len = sum(self._get_message_length(m) for m in system_msgs)
        total_len = system_len + sum(msg_lengths)

        remove_indices: Set[int] = set()
        i = 0

        # 从前往后删除消息，直到总长度低于目标
        while total_len > target_chars and i < len(conversation_msgs):
            if i in remove_indices:
                i += 1
                continue

            msg = conversation_msgs[i]

            # 处理工具调用链
            if self._is_assistant_with_tools(msg):
                tool_ids = self._extract_tool_ids(msg)
                chain_indices = self._find_tool_chain(conversation_msgs, i, tool_ids)

                # 计算整条链的长度
                chain_len = sum(msg_lengths[idx] for idx in chain_indices if idx < len(msg_lengths))
                for idx in chain_indices:
                    remove_indices.add(idx)
                total_len -= chain_len
            else:
                remove_indices.add(i)
                total_len -= msg_lengths[i]

            i += 1

        remaining = [m for idx, m in enumerate(conversation_msgs) if idx not in remove_indices]
        self.messages = system_msgs + remaining

        logger.debug(
            "Context trimmed",
            remaining_messages=len(self.messages),
            removed_count=len(remove_indices),
        )

    def _is_assistant_with_tools(self, msg: Message) -> bool:
        """
        检查消息是否是带工具调用的 assistant 消息
        
        参数:
            msg: 要检查的消息
            
        返回:
            True 如果是带工具调用的 assistant 消息
        """
        return getattr(msg, "role", None) == "assistant" and bool(getattr(msg, "tool_calls", None))

    def _extract_tool_ids(self, msg: Message) -> Set[str]:
        """
        从消息中提取所有工具调用 ID
        
        参数:
            msg: assistant 消息
            
        返回:
            工具调用 ID 的集合
        """
        tool_calls = getattr(msg, "tool_calls", None) or []
        return {
            tc.get("id", "")
            for tc in tool_calls
            if isinstance(tc, dict) and tc.get("id")
        }

    def _find_tool_chain(
        self, 
        msgs: List[Message], 
        start_index: int, 
        tool_ids: Set[str]
    ) -> List[int]:
        """
        找到工具调用链的所有消息索引
        
        从 assistant 消息开始，找到所有相关的 tool 结果消息。
        
        参数:
            msgs: 消息列表
            start_index: assistant 消息的索引
            tool_ids: 工具调用 ID 集合
            
        返回:
            包含整条链的索引列表
        """
        indices = [start_index]
        
        # 从 start_index 之后查找相关的 tool 消息
        for j in range(start_index + 1, len(msgs)):
            check_msg = msgs[j]
            if getattr(check_msg, "role", None) == "tool":
                tool_call_id = getattr(check_msg, "tool_call_id", None)
                if tool_call_id in tool_ids:
                    indices.append(j)
                else:
                    # 遇到不相关的 tool 消息，停止
                    break
            else:
                # 遇到非 tool 消息，停止
                break
        return indices

    def _get_message_length(self, msg: Message) -> int:
        """
        计算消息的字符长度
        
        参数:
            msg: 消息对象
            
        返回:
            文本内容的字符数
        """
        try:
            return len(msg.get_text_content())
        except Exception:
            return 0

    @property
    def last_message(self) -> Message:
        """
        获取最后一条消息
        
        返回:
            上下文中的最后一条消息
            
        抛出:
            ValueError: 上下文为空
        """
        if not self.messages:
            raise ValueError("Context is empty, cannot get last message")
        return self.messages[-1]


# ============================================================================
# ReAct 引擎类
# ============================================================================
class ReActEngine(CognitiveEngine):
    """
    ReAct (Reasoning + Acting) 认知引擎
    
    实现了完整的 ReAct 循环：思考 → 行动 → 观察 → 重复。
    
    核心特性：
    1. 智能循环检测：避免重复相同的工具调用
    2. 错误恢复：自动反思和重试机制
    3. 上下文管理：智能裁剪历史消息
    4. 流式输出：支持实时流式响应
    5. 结构化输出：支持 Pydantic 模型约束
    
    工作流程：
    - Think Phase: 调用 LLM 生成响应或工具调用
    - Act Phase: 执行工具调用
    - Observe Phase: 分析结果，检测错误和循环
    """
    
    def __init__(
        self,
        model: Any,
        toolbox: ToolBox,
        memory: TokenMemory,
        max_turns: int = 10,
        max_observation_length: int = 2000,
        system_prompt: Union[str, PromptTemplate, None] = None,
        on_turn_start: Optional[Callable[[ExecutionContext], Any]] = None,
        on_turn_end: Optional[Callable[[ExecutionContext], Any]] = None,
        config: Optional[ReActConfig] = None,
        **kwargs: Any,
    ):
        """
        初始化 ReAct 引擎
        
        参数:
            model: LLM 模型实例
            toolbox: 工具箱
            memory: 记忆系统
            max_turns: 最大轮次（防止无限循环）
            max_observation_length: 工具输出最大长度
            system_prompt: 系统提示（字符串或模板）
            on_turn_start: 轮次开始回调
            on_turn_end: 轮次结束回调
            config: ReAct 配置
            **kwargs: 传递给父类的额外参数
        """
        super().__init__(model, toolbox, memory, **kwargs)

        self.max_turns = int(max_turns)
        self.max_observation_length = int(max_observation_length)
        self.on_turn_start = on_turn_start
        self.on_turn_end = on_turn_end
        self.config = config or ReActConfig()

        # 处理系统提示
        if system_prompt is None:
            self.prompt_template = PromptTemplate(template=DEFAULT_REACT_TEMPLATE)
        elif isinstance(system_prompt, str):
            self.prompt_template = PromptTemplate(template=system_prompt)
        else:
            self.prompt_template = system_prompt

        # 检测模型是否支持函数调用
        self._supports_functions: bool = bool(getattr(self.model, "_supports_function_calling", True))

        logger.debug(
            "ReActEngine initialized",
            max_turns=self.max_turns,
            supports_functions=self._supports_functions,
        )

    def _safe_deep_copy_messages(self, messages: List[Message]) -> List[Message]:
        """
        安全地深拷贝消息列表
        
        尝试使用 Pydantic 的 model_copy，失败则使用 deepcopy。
        如果都失败，使用原始引用并记录警告。
        
        参数:
            messages: 要拷贝的消息列表
            
        返回:
            拷贝后的消息列表
        """
        result: List[Message] = []
        for m in messages:
            try:
                if hasattr(m, "model_copy"):
                    result.append(m.model_copy(deep=True))
                else:
                    result.append(copy.deepcopy(m))
            except Exception as e:
                logger.warning(
                    "Message copy failed, using original reference",
                    error=str(e),
                    message_type=type(m).__name__,
                )
                result.append(m)
        return result

    async def step( # type: ignore
        self,
        input_messages: List[Message],
        response_model: Optional[Type[T]] = None,
        max_retries: int = 0,
        **kwargs: Any,
    ) -> Union[AgentOutput, T]:
        """
        执行一个完整的推理步骤
        
        如果指定了 response_model，会尝试解析结构化输出。
        支持自动重试机制处理解析错误。
        
        参数:
            input_messages: 输入消息列表
            response_model: 期望的输出模型类（Pydantic BaseModel）
            max_retries: 解析失败时的最大重试次数
            **kwargs: 额外参数
            
        返回:
            AgentOutput 或结构化对象（取决于是否指定 response_model）
            
        抛出:
            TypeError: response_model 不是 BaseModel 子类
            AgentError: 执行失败或达到最大重试次数
        """
        # 验证 response_model
        if response_model is not None:
            from inspect import isclass

            if not (isclass(response_model) and issubclass(response_model, BaseModel)):
                raise TypeError(
                    f"response_model must be a subclass of Pydantic BaseModel, got: {type(response_model).__name__}"
                )

            # 执行推理
            output = await self._execute_step(input_messages, response_model=response_model, **kwargs)

            current_messages = list(input_messages)
            attempts = 0

            # 解析和重试循环
            while True:
                try:
                    # 尝试从工具调用中解析
                    if output.tool_calls:
                        return await StructureEngine.parse(
                            content="",
                            model_class=response_model,
                            raw_tool_calls=output.tool_calls,
                        )

                    # 尝试从文本内容中解析
                    return await StructureEngine.parse(output.content, response_model)

                except Exception as e:
                    # 达到最大重试次数
                    if attempts >= max_retries:
                        raise AgentError(f"Structured parsing failed: {e}") from e

                    attempts += 1
                    logger.warning(
                        "Structure parse failed, retrying",
                        attempt=attempts,
                        max_retries=max_retries,
                        error=str(e),
                    )

                    # 添加错误反馈，重新执行
                    current_messages.append(
                        Message.assistant(content=output.content, tool_calls=output.tool_calls)
                    )
                    current_messages.append(
                        Message.user(
                            f"Error parsing response: {e}. Please try again using the correct format."
                        )
                    )

                    output = await self._execute_step(
                        current_messages, response_model=response_model, **kwargs
                    )

        # 不需要结构化输出，直接执行
        return await self._execute_step(input_messages, **kwargs)

    async def step_structured(
        self,
        input_messages: List[Message],
        response_model: Type[T],
        max_retries: int = 0,
        **kwargs: Any,
    ) -> T:
        """
        执行结构化输出推理
        
        这是 step() 的便捷方法，明确返回结构化对象。
        
        参数:
            input_messages: 输入消息列表
            response_model: Pydantic 模型类
            max_retries: 最大重试次数
            **kwargs: 额外参数
            
        返回:
            解析后的结构化对象
        """
        result = await self.step(
            input_messages, response_model=response_model, max_retries=max_retries, **kwargs
        )
        return cast(T, result)

    async def step_stream( # type: ignore
        self,
        input_messages: List[Message],
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[AgentStreamEvent]:
        """
        流式执行推理
        
        生成一系列事件，实时反映推理过程：
        - token: 文本片段
        - tool_input: 工具调用开始
        - tool_output: 工具执行结果
        - result: 最终结果
        - error: 错误信息
        
        参数:
            input_messages: 输入消息列表
            timeout: 超时时间（秒），None 使用默认值
            **kwargs: 额外参数
            
        生成:
            AgentStreamEvent: 流式事件
            
        抛出:
            asyncio.TimeoutError: 超时
            AgentError: 执行失败
        """
        if timeout is None:
            timeout = get_settings().default_model_timeout

        # 验证输入
        self.validate_input(input_messages)
        await self.before_step(input_messages, **kwargs)

        # 构建执行上下文
        context = await self._build_execution_context(input_messages)
        start_time = time.time()

        final_output: Optional[AgentOutput] = None
        saw_error_event: bool = False

        try:
            # 执行主循环（带超时）
            async for event in self._execute_lifecycle_with_timeout(context, float(timeout), **kwargs):
                # 捕获最终结果
                if event.type == "result" and event.data:
                    try:
                        final_output = cast(AgentOutput, event.data.get("output"))
                    except Exception:
                        final_output = None

                # 标记是否看到错误
                if event.type == "error":
                    saw_error_event = True

                yield event

        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            logger.error(
                "step_stream timeout",
                elapsed_seconds=elapsed,
                max_timeout=timeout,
                current_turn=context.turn,
            )

            # 保存上下文（即使超时）
            await self._save_context(context, force=True)

            # 发送超时错误事件
            yield AgentStreamEvent(
                type="error",
                content=f"Execution timeout after {timeout}s at turn {context.turn}",
            )
            raise

        except asyncio.CancelledError:
            logger.warning("step_stream was cancelled", current_turn=context.turn)
            await self._save_context(context, force=True)
            raise

        except AgentError as e:
            logger.error("step_stream caught AgentError", error=str(e))
            yield AgentStreamEvent(type="error", content=str(e))
            await self.on_error(e, input_messages, **kwargs)
            raise

        except Exception as e:
            logger.exception("Lifecycle execution failed", error=str(e))
            yield AgentStreamEvent(type="error", content=str(e))
            await self.on_error(e, input_messages, **kwargs)
            raise

        finally:
            # 保存上下文
            try:
                await self._save_context(context)
            except Exception as save_error:
                logger.warning("Final context save failed", error=str(save_error))

            # 调用 after_step 钩子
            if final_output is not None and not saw_error_event:
                try:
                    await self.after_step(
                        context.messages,
                        final_output,
                        original_input=input_messages,
                        **kwargs
                    )
                except Exception as hook_err:
                    logger.warning("after_step failed", error=str(hook_err), exc_info=True)

    async def _execute_lifecycle_with_timeout(
        self,
        context: ExecutionContext,
        timeout: float,
        **kwargs: Any,
    ) -> AsyncIterator[AgentStreamEvent]:
        """
        带超时的生命周期执行
        
        包装 _execute_lifecycle，添加超时保护。
        
        参数:
            context: 执行上下文
            timeout: 超时时间（秒）
            **kwargs: 额外参数
            
        生成:
            AgentStreamEvent: 流式事件
        """
        async with asyncio.timeout(timeout):
            async for event in self._execute_lifecycle(context, **kwargs):
                yield event

    async def _execute_step(
        self,
        input_messages: List[Message],
        response_model: Optional[Type[T]] = None,
        **kwargs: Any,
    ) -> AgentOutput:
        """
        执行单步推理（非流式）
        
        内部使用流式执行，但只返回最终结果。
        
        参数:
            input_messages: 输入消息列表
            response_model: 结构化输出模型
            **kwargs: 额外参数
            
        返回:
            最终的 AgentOutput
            
        抛出:
            AgentError: 执行失败或没有生成结果
        """
        # 安全拷贝消息（避免修改原始输入）
        current_messages = self._safe_deep_copy_messages(input_messages)

        stream_kwargs: Dict[str, Any] = dict(kwargs)
        stream_kwargs["response_model"] = response_model

        final_result: Optional[AgentOutput] = None

        try:
            # 消费流式事件，捕获最终结果
            async for event in self.step_stream(current_messages, **stream_kwargs):
                if event.type == "result" and event.data:
                    final_result = cast(AgentOutput, event.data.get("output"))

                elif event.type == "error":
                    error_msg = str(event.content)
                    logger.error("Received error event during execution", error=error_msg)
                    raise AgentError(error_msg)

        except AgentError:
            raise
        except Exception as e:
            logger.exception("Event stream processing failed", error=str(e))
            raise AgentError(f"Step execution failed: {e}") from e

        # 检查是否有结果
        if final_result is None:
            raise AgentError("Infinite loop detected: no result generated")

        return final_result

    async def _execute_lifecycle(self, context: ExecutionContext, **kwargs: Any) -> AsyncIterator[AgentStreamEvent]:
        """
        执行完整的 ReAct 生命周期
        
        这是引擎的核心方法，实现完整的 Think-Act-Observe 循环。
        
        工作流程：
        1. 进入循环（最多 max_turns 轮）
        2. Think Phase: 调用 LLM 生成响应
        3. 检测循环
        4. 如果有工具调用，进入 Act Phase
        5. Observe Phase: 分析结果，检测错误
        6. 如果任务完成或出错，退出循环
        
        参数:
            context: 执行上下文
            **kwargs: 额外参数（包括 response_model）
            
        生成:
            AgentStreamEvent: 流式事件
        """
        response_model = kwargs.get("response_model")
        structure_tool_name: Optional[str] = None

        # 如果需要结构化输出，准备特殊工具
        if response_model and self._supports_functions:
            schema = StructureEngine.to_openai_tool(response_model)
            structure_tool_name = schema["function"]["name"]

        # 主循环
        while context.turn < self.max_turns:
            context.turn += 1
            logger.debug("Starting turn", turn=context.turn)

            # 轮次开始回调
            if self.on_turn_start:
                await ensure_awaitable(self.on_turn_start, context)

            # === Think Phase ===
            async for event in self._think_phase(context, **kwargs):
                yield event

            assistant_msg = context.last_message

            # === 循环检测 ===
            if self._detect_loop(context, assistant_msg):
                error_msg = "Infinite loop detected"
                yield AgentStreamEvent(type="error", content=error_msg)
                return

            tool_calls = assistant_msg.safe_tool_calls

            # 检查是否调用了结构化输出工具
            if structure_tool_name and tool_calls:
                target_call = next(
                    (
                        tc for tc in tool_calls
                        if tc.get("function", {}).get("name") == structure_tool_name
                    ),
                    None,
                )
                if target_call:
                    # 找到结构化输出，返回结果
                    final_output = AgentOutput(
                        content="",
                        tool_calls=[target_call],
                        metadata={"is_structured": True},
                    )
                    yield AgentStreamEvent(type="result", data={"output": final_output})
                    return

            # 没有工具调用，任务完成
            if not tool_calls:
                final_output = AgentOutput(content=str(assistant_msg.content or ""), tool_calls=[])
                yield AgentStreamEvent(type="result", data={"output": final_output})
                return

            # 发送工具调用事件
            yield AgentStreamEvent(type="tool_input", data={"tools": tool_calls})

            # 记录统计
            for _ in tool_calls:
                self.record_tool_call()

            # === Act Phase ===
            async for event in self._act_phase(context, tool_calls):
                yield event

            # === Observe Phase ===
            should_continue = await self._observe_phase(context)

            # 轮次结束回调
            if self.on_turn_end:
                await ensure_awaitable(self.on_turn_end, context)

            # 保存上下文
            await self._save_context(context)

            # 检查是否应该停止
            if not should_continue:
                stop_output = AgentOutput(content="Task stopped by system monitor.")
                yield AgentStreamEvent(type="result", data={"output": stop_output})
                return

        # 达到最大轮次，视为无限循环
        logger.warning("Reached max turns limit, treating as infinite loop", max_turns=self.max_turns)
        yield AgentStreamEvent(type="error", content="Infinite loop detected")
        return

    async def _think_phase(self, context: ExecutionContext, **kwargs: Any) -> AsyncIterator[AgentStreamEvent]:
        """
        思考阶段：调用 LLM 生成响应
        
        使用流式 API 获取响应，并实时发送 token 事件。
        同时跟踪 token 使用量和成本。
        
        参数:
            context: 执行上下文
            **kwargs: 额外参数（包括 response_model）
            
        生成:
            AgentStreamEvent: token 事件
        """
        # 构建消息负载
        messages_payload = self._build_messages_payload(context)
        
        # 构建 LLM 参数（包括工具定义）
        llm_params = self._build_llm_params(kwargs.get("response_model"), strategy="auto")
        
        # 合并用户提供的参数（排除 response_model）
        safe_kwargs = {k: v for k, v in kwargs.items() if k not in ["response_model"]}
        llm_params.update(safe_kwargs)
        llm_params["stream"] = True

        # 初始化统计
        input_tokens = 0
        output_tokens = 0
        model_name = self._get_model_name()

        # 创建流式缓冲区
        buffer = StreamBuffer()
        stream_gen = self.model.astream(messages=messages_payload, **llm_params) # type: ignore

        try:
            # 处理流式响应
            async for chunk in stream_gen:
                # 验证 chunk 格式
                if not hasattr(chunk, "delta"):
                    delta = getattr(chunk, "choices", [{}])
                    if not delta:
                        logger.warning("Received chunk without delta, skipping", chunk_type=type(chunk).__name__)
                        continue

                # 提取 token 使用量
                usage = getattr(chunk, "usage", None)
                if usage:
                    input_tokens = max(input_tokens, int(getattr(usage, "prompt_tokens", 0) or 0))
                    output_tokens = max(output_tokens, int(getattr(usage, "completion_tokens", 0) or 0))

                # 添加到缓冲区，获取新文本
                text_delta = buffer.add_chunk(chunk)
                if text_delta:
                    # 发送 token 事件
                    yield AgentStreamEvent(type="token", content=text_delta)
        finally:
            # 记录统计（即使发生异常）
            if input_tokens > 0 or output_tokens > 0:
                self.record_tokens(input_tokens=input_tokens, output_tokens=output_tokens)
                self.record_cost(input_tokens, output_tokens, model_name)

        # 构建完整消息并添加到上下文
        assistant_msg = buffer.build_message()
        context.add_message(assistant_msg)

    async def _act_phase(
        self, 
        context: ExecutionContext, 
        tool_calls: List[Dict[str, Any]]
    ) -> AsyncIterator[AgentStreamEvent]:
        """
        行动阶段：执行工具调用
        
        批量执行所有工具调用，并将结果添加到上下文。
        同时发送工具输出事件。
        
        参数:
            context: 执行上下文
            tool_calls: 工具调用列表
            
        生成:
            AgentStreamEvent: tool_output 事件
        """
        # 规范化工具调用格式
        normalized_calls = [self._normalize_tool_call(tc) for tc in tool_calls]
        
        # 批量执行工具
        tool_results = await self.toolbox.execute_many(normalized_calls)

        # 处理每个结果
        for result in tool_results:
            # 截断过长的输出
            truncated_content = self._truncate_observation(result.result, result.tool_name)

            # 添加工具结果消息到上下文
            context.add_message(
                Message.tool_result(
                    result.call_id,
                    truncated_content,
                    result.tool_name,
                )
            )

            # 发送工具输出事件
            yield AgentStreamEvent(
                type="tool_output",
                content=result.result,
                data={"tool_name": result.tool_name, "is_error": result.is_error},
            )

        # 保存结果到元数据（用于 Observe Phase）
        context.metadata["last_tool_results"] = tool_results

    async def _observe_phase(self, context: ExecutionContext) -> bool:
        """
        观察阶段：分析工具执行结果
        
        检查：
        1. 工具执行是否出错
        2. 是否需要反思（错误恢复）
        3. 是否应该停止执行
        
        参数:
            context: 执行上下文
            
        返回:
            True 如果应该继续执行，False 如果应该停止
        """
        results = context.metadata.get("last_tool_results", [])
        error_count = sum(1 for r in results if r.is_error)

        # 更新错误计数
        if error_count > 0:
            context.consecutive_errors += 1
            logger.warning(
                "Tool execution errors",
                error_count=error_count,
                total_count=len(results),
                consecutive_errors=context.consecutive_errors,
            )
        else:
            context.consecutive_errors = 0

        # 检查是否达到错误阈值
        if context.consecutive_errors >= self.config.tool_error_threshold:
            # 尝试反思恢复
            if context.reflection_attempts < self.config.max_reflections:
                self._inject_reflection_message(context, results)
                context.reflection_attempts += 1
                context.consecutive_errors = 0
                return True

            # 反思次数用尽，停止执行
            logger.error(
                "Tool error threshold exceeded, stopping",
                consecutive_errors=context.consecutive_errors,
                reflection_attempts=context.reflection_attempts,
                turn=context.turn,
            )
            return False

        return True

    def _inject_reflection_message(
        self, 
        context: ExecutionContext, 
        results: List[ToolExecutionResult]
    ) -> None:
        """
        注入反思消息
        
        当工具执行出现连续错误时，向上下文注入系统反馈消息，
        帮助 LLM 分析错误并调整策略。
        
        参数:
            context: 执行上下文
            results: 工具执行结果列表
        """
        # 汇总错误详情
        error_details = "\n".join(
            f"- {r.tool_name}: {r.result}" for r in results if r.is_error
        )

        # 构建系统反馈
        system_feedback = (
            "System Notification: Multiple tool execution errors detected.\n"
            "Error Details:\n"
            f"{error_details}\n\n"
            "Please analyze these errors, adjust parameters or tool choice, and retry."
        )

        # 创建反思消息
        reflection_msg = Message.assistant(content=system_feedback, tool_calls=None)

        # 附加元数据
        meta = {
            "type": "system_reflection",
            "error_summary": error_details,
            "reflection_attempt": context.reflection_attempts + 1,
        }

        self._attach_metadata_safe(reflection_msg, meta)

        # 存储到上下文元数据
        store = context.metadata.setdefault("_gecko_msg_metadata", {})
        store[id(reflection_msg)] = meta

        # 添加到消息历史
        context.add_message(reflection_msg)

    def _detect_loop(self, context: ExecutionContext, msg: Message) -> bool:
        """
        检测无限循环
        
        通过哈希工具调用来检测：
        1. 连续重复相同的工具调用
        2. A-B-A 交替模式（振荡）
        
        参数:
            context: 执行上下文
            msg: 当前的 assistant 消息
            
        返回:
            True 如果检测到循环
        """
        if not msg.safe_tool_calls:
            return False

        try:
            # 提取工具调用的关键信息（名称和参数）
            calls_data = [
                {
                    "name": tc.get("function", {}).get("name"),
                    "args": tc.get("function", {}).get("arguments"),
                }
                for tc in msg.safe_tool_calls
            ]
            
            # 计算哈希
            calls_dump = json.dumps(calls_data, sort_keys=True, ensure_ascii=False)
            current_hash = hashlib.sha256(calls_dump.encode("utf-8")).hexdigest()

            # 检测连续重复
            repeat_run = 1
            for h in reversed(context.last_tool_hashes):
                if h == current_hash:
                    repeat_run += 1
                else:
                    break

            if repeat_run >= self.config.loop_repeat_threshold:
                logger.warning(
                    "Consecutive tool call loop detected",
                    repeat_run=repeat_run,
                    threshold=self.config.loop_repeat_threshold,
                    tool_hash=current_hash[:16],
                )
                return True

            # 检测 A-B-A 振荡模式
            if len(context.last_tool_hashes) >= 2:
                if (current_hash == context.last_tool_hashes[-2] and
                    current_hash != context.last_tool_hashes[-1]):
                    logger.warning(
                        "Oscillation pattern detected (A-B-A alternating calls)",
                        tool_hash=current_hash[:16],
                    )
                    return True

            # 记录当前哈希
            context.last_tool_hashes.append(current_hash)
            context.last_tool_hash = current_hash

            return False

        except Exception as e:
            logger.warning("Loop detection failed (fail-open)", error=str(e), exc_info=True)
            return False

    def _truncate_observation(self, content: str, tool_name: str) -> str:
        """
        截断工具输出
        
        防止过长的工具输出占用过多上下文空间。
        
        参数:
            content: 原始输出内容
            tool_name: 工具名称
            
        返回:
            可能被截断的内容
        """
        if len(content) > self.max_observation_length:
            logger.info(
                "Truncating output for tool",
                tool_name=tool_name,
                original_length=len(content),
                max_length=self.max_observation_length,
            )
            return content[: self.max_observation_length] + f"\n...(truncated, total {len(content)} chars)"
        return content

    def _normalize_tool_call(self, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        """
        规范化工具调用格式
        
        将 LLM 返回的工具调用转换为统一格式，并解析 JSON 参数。
        
        参数:
            tool_call: 原始工具调用字典
            
        返回:
            规范化的工具调用
        """
        func_block = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        name = func_block.get("name", "") if isinstance(func_block, dict) else ""
        raw_args = func_block.get("arguments", "{}") if isinstance(func_block, dict) else "{}"

        parsed_args: Dict[str, Any] = {}

        try:
            if isinstance(raw_args, str):
                parsed_args = json.loads(raw_args)
            elif isinstance(raw_args, dict):
                parsed_args = raw_args
            else:
                parsed_args = {}
        except json.JSONDecodeError as e:
            # JSON 解析失败，创建错误信息
            parsed_args = {
                "__gecko_parse_error__": (
                    f"JSON format error: {str(e)}. "
                    f"Content: {raw_args[:200] if isinstance(raw_args, str) and len(raw_args) > 200 else raw_args}"
                )
            }
            logger.warning("Tool arguments JSON parse failed", tool_name=name, error=str(e))

        return {"id": tool_call.get("id", ""), "name": name, "arguments": parsed_args}

    def _get_model_name(self) -> str:
        """
        获取模型名称
        
        尝试从模型对象中提取名称，失败则使用默认值。
        
        返回:
            模型名称字符串
        """
        model_name = getattr(self.model, "model_name", None) or getattr(self.model, "model", None) or "gpt-3.5-turbo"
        if not isinstance(model_name, str):
            logger.warning("Model name type unexpected", model_name_type=type(model_name).__name__)
            model_name = "gpt-3.5-turbo"
        return model_name

    def _build_llm_params(self, response_model: Any, strategy: str = "auto") -> Dict[str, Any]:
        """
        构建 LLM 调用参数
        
        根据是否需要结构化输出和可用工具，构建合适的参数。
        
        参数:
            response_model: 结构化输出模型（可选）
            strategy: 工具选择策略（默认 "auto"）
            
        返回:
            LLM 参数字典
        """
        params: Dict[str, Any] = {}
        tools_schema = self.toolbox.to_openai_schema()

        # 结构化输出：添加特殊工具并强制调用
        if response_model and self._supports_functions:
            structure_tool = StructureEngine.to_openai_tool(response_model)
            combined_tools = tools_schema + [structure_tool]
            params["tools"] = combined_tools
            params["tool_choice"] = {"type": "function", "function": {"name": structure_tool["function"]["name"]}}

        # 普通工具调用
        elif tools_schema and self._supports_functions:
            params["tools"] = tools_schema
            params["tool_choice"] = "auto"

        return params

    def _build_messages_payload(self, context: ExecutionContext) -> List[Dict[str, Any]]:
        """
        构建消息负载
        
        将上下文中的消息转换为 OpenAI 格式，并附加元数据。
        
        参数:
            context: 执行上下文
            
        返回:
            消息字典列表
        """
        meta_map = context.metadata.get("_gecko_msg_metadata", {}) or {}
        messages_payload: List[Dict[str, Any]] = []

        for m in context.messages:
            # 转换为 OpenAI 格式
            payload = m.to_openai_format()
            
            # 提取并附加元数据
            md = self._extract_message_metadata(m, meta_map, context)

            if md:
                payload["metadata"] = md

            messages_payload.append(payload)

        return messages_payload

    def _extract_message_metadata(
        self, 
        msg: Message, 
        meta_map: Dict[int, Dict[str, Any]], 
        context: ExecutionContext
    ) -> Optional[Dict[str, Any]]:
        """
        提取消息的元数据
        
        尝试多种方式获取消息关联的元数据：
        1. 直接从消息对象的 metadata 属性
        2. 从元数据映射表（使用消息 ID）
        3. 从上下文的消息元数据字典
        
        参数:
            msg: 消息对象
            meta_map: 元数据映射表
            context: 执行上下文
            
        返回:
            元数据字典，没有则返回 None
        """
        md = None

        # 尝试 1：从消息对象的 metadata 属性
        try:
            if hasattr(msg, "metadata"):
                v = getattr(msg, "metadata", None)
                if isinstance(v, dict) and v:
                    md = v
        except Exception:
            pass

        # 尝试 2：从元数据映射表
        if md is None:
            try:
                v2 = meta_map.get(id(msg))
                if isinstance(v2, dict) and v2:
                    md = v2
            except Exception:
                pass

        # 尝试 3：从上下文的消息元数据字典
        if md is None:
            try:
                msg_id = getattr(msg, "_gecko_msg_id", None)
                if msg_id and msg_id in context.message_metadata:
                    md = context.message_metadata[msg_id]
            except Exception:
                pass

        return md

    async def _build_execution_context(self, input_messages: List[Message]) -> ExecutionContext:
        """
        构建执行上下文
        
        流程：
        1. 加载历史消息（从记忆系统）
        2. 合并历史和输入消息
        3. 添加系统提示（如果没有）
        4. 创建执行上下文
        
        参数:
            input_messages: 当前输入消息
            
        返回:
            初始化的执行上下文
        """
        # 加载历史
        history = await self._load_history()
        all_messages = history + input_messages

        # 检查是否已有系统消息
        has_system_msg = any(m.role == "system" for m in all_messages)
        if not has_system_msg:
            # 渲染并添加系统提示
            system_content = self._render_system_prompt()
            all_messages.insert(0, Message.system(system_content))
        else:
            logger.debug("Using user-specified system message")

        # 创建上下文
        max_context_chars = self.config.max_context_chars
        return ExecutionContext(all_messages, max_chars=max_context_chars)

    def _render_system_prompt(self) -> str:
        """
        渲染系统提示
        
        使用模板引擎渲染系统提示，注入工具列表和当前时间。
        
        返回:
            渲染后的系统提示文本
        """
        template_vars = {
            "tools": self.toolbox.to_openai_schema(),
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            return self.prompt_template.format_safe(**template_vars)
        except Exception as e:
            logger.warning("System prompt formatting failed", error=str(e))
            return "You are a helpful AI assistant."

    async def _load_history(self) -> List[Message]:
        """
        从记忆系统加载历史消息
        
        返回:
            历史消息列表，失败返回空列表
        """
        if not getattr(self.memory, "storage", None):
            return []
        try:
            data = await self.memory.storage.get(self.memory.session_id) # type: ignore
            if data and "messages" in data:
                return await self.memory.get_history(data["messages"])
        except Exception as e:
            logger.warning("Failed to load history", error=str(e))
        return []

    async def _save_context(self, context: ExecutionContext, force: bool = False, max_retries: int = 3) -> None:
        """
        保存上下文到记忆系统
        
        支持重试机制，使用指数退避策略。
        
        参数:
            context: 要保存的执行上下文
            force: 是否强制保存（失败时抛出异常）
            max_retries: 最大重试次数
        """
        if not getattr(self.memory, "storage", None):
            return

        # 转换为可序列化格式
        messages_data = [m.to_openai_format() for m in context.messages]

        # 确定重试次数
        retries = max_retries if force else 1
        
        for attempt in range(retries):
            try:
                await self.memory.storage.set(self.memory.session_id, {"messages": messages_data}) # type: ignore
                return
            except Exception as e:
                if not force or attempt >= retries - 1:
                    logger.warning("Failed to save context", error=str(e), attempt=attempt + 1, force=force)
                    if force:
                        raise
                    return

                # 计算退避延迟（指数退避 + 随机抖动）
                base_delay = min(0.1 * (2 ** attempt), MAX_RETRY_DELAY_SECONDS)
                jitter = random.uniform(0, base_delay * 0.5)
                await asyncio.sleep(base_delay + jitter)

    def _attach_metadata_safe(self, msg: Message, metadata: Dict[str, Any]) -> None:
        """
        安全地附加元数据到消息
        
        尝试多种方式附加元数据，失败不抛出异常。
        
        参数:
            msg: 目标消息
            metadata: 要附加的元数据
        """
        if not metadata:
            return

        # 尝试 1：更新现有 metadata 属性
        try:
            if hasattr(msg, "metadata"):
                current = getattr(msg, "metadata", None)
                if isinstance(current, dict):
                    current.update(metadata)
                    return
                else:
                    try:
                        setattr(msg, "metadata", dict(metadata))
                        return
                    except Exception:
                        pass
        except Exception:
            pass

        # 尝试 2：使用 object.__setattr__ 绕过限制
        try:
            object.__setattr__(msg, "metadata", dict(metadata))
        except Exception:
            logger.debug("Failed to attach metadata to message (ignored)")


# ============================================================================
# 模块导出
# ============================================================================
__all__ = [
    "ReActEngine",              # ReAct 引擎主类
    "ExecutionContext",         # 执行上下文管理器
    "ReActConfig",              # 配置类
    "DEFAULT_REACT_TEMPLATE",   # 默认系统提示模板
    "STRUCTURE_TOOL_PREFIX",    # 结构化输出工具前缀
    "MAX_RETRY_DELAY_SECONDS",  # 最大重试延迟
]