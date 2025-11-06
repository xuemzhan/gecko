"""
Agent Service - 智能体业务逻辑服务

职责：
1. 核心运行逻辑（run/arun/continue_run）
2. 消息构建（系统消息、用户消息）
3. 工具管理和执行
4. 推理处理
5. 响应处理和解析
6. 钩子执行
7. 知识库检索
8. 默认工具提供
9. CLI和打印功能

这是Agent的核心业务逻辑层，协调各个组件完成AI交互
"""

from __future__ import annotations

from asyncio import CancelledError, create_task
from collections import ChainMap, deque
from inspect import iscoroutinefunction, signature
from textwrap import dedent
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    cast,
    get_args,
)
from uuid import uuid4

from pydantic import BaseModel

from gecko.domain.agent.entity import AgentEntity
from gecko.domain.agent.repository import AgentRepository
from gecko.domain.agent.value_objects import (
    RunContext,
    RunMessages,
    ReasoningState,
    ToolExecutionContext,
)
from gecko.core.context.manager import ContextManager
from gecko.core.memory.manager import (
    AgentMemoryManager,
    SessionMetricsManager,
    SessionNameManager,
)
from gecko.utils.logging import (
    log_debug,
    log_error,
    log_exception,
    log_info,
    log_warning,
)
from gecko.utils.timer import Timer

# 临时导入（后续需要迁移到gecko架构下）
try:
    from agno.exceptions import (
        InputCheckError,
        ModelProviderError,
        OutputCheckError,
        RunCancelledException,
        StopAgentRun,
    )
    from agno.knowledge.document import Document
    from agno.knowledge.types import KnowledgeFilter
    from agno.media import Audio, File, Image, Video
    from agno.models.base import Model
    from agno.models.message import Message, MessageReferences
    from agno.models.metrics import Metrics
    from agno.models.response import ModelResponse, ModelResponseEvent, ToolExecution
    from agno.reasoning.step import NextAction, ReasoningStep, ReasoningSteps
    from agno.run import RunStatus
    from agno.run.agent import RunEvent, RunInput, RunOutput, RunOutputEvent
    from agno.run.cancel import (
        cancel_run as cancel_run_global,
        cleanup_run,
        raise_if_cancelled,
        register_run,
    )
    from agno.session import AgentSession
    from agno.tools import Toolkit
    from agno.tools.function import Function
    from agno.utils.common import is_typed_dict, validate_typed_dict
    from agno.utils.events import (
        create_parser_model_response_completed_event,
        create_parser_model_response_started_event,
        create_post_hook_completed_event,
        create_post_hook_started_event,
        create_pre_hook_completed_event,
        create_pre_hook_started_event,
        create_reasoning_completed_event,
        create_reasoning_started_event,
        create_reasoning_step_event,
        create_run_cancelled_event,
        create_run_completed_event,
        create_run_content_completed_event,
        create_run_continued_event,
        create_run_error_event,
        create_run_output_content_event,
        create_run_paused_event,
        create_run_started_event,
        create_session_summary_completed_event,
        create_session_summary_started_event,
        create_tool_call_completed_event,
        create_tool_call_started_event,
        handle_event,
    )
    from agno.utils.hooks import filter_hook_args, normalize_hooks
    from agno.utils.knowledge import get_agentic_or_user_search_filters
    from agno.utils.message import filter_tool_calls, get_text_from_message
    from agno.utils.prompts import get_json_output_prompt, get_response_model_format_prompt
    from agno.utils.reasoning import (
        add_reasoning_metrics_to_metadata,
        add_reasoning_step_to_metadata,
        append_to_reasoning_content,
        update_run_output_with_reasoning,
    )
    from agno.utils.response import get_paused_content
    from agno.utils.string import parse_response_model_str
except ImportError as e:
    # 临时处理导入错误
    log_warning(f"Some imports from agno failed: {e}")


class AgentService:
    """
    Agent服务 - 核心业务逻辑
    
    整合Entity、Repository、ContextManager、MemoryManager
    提供完整的Agent运行能力
    """
    
    def __init__(
        self,
        entity: AgentEntity,
        repository: Optional[AgentRepository] = None,
        context_manager: Optional[ContextManager] = None,
        memory_manager: Optional[AgentMemoryManager] = None,
    ):
        """
        初始化Agent服务
        
        Args:
            entity: Agent实体
            repository: 仓储（可选，用于数据持久化）
            context_manager: 上下文管理器（可选）
            memory_manager: 记忆管理器（可选）
        """
        self.entity = entity
        
        # 初始化仓储
        self.repository = repository or AgentRepository(
            db=entity.db,
            team_id=entity.team_id,
            workflow_id=entity.workflow_id,
            cache_enabled=entity.cache_session,
        )
        
        # 初始化上下文管理器
        self.context_manager = context_manager or ContextManager(
            agent_id=entity.id,
            agent_name=entity.name,
            default_user_id=entity.user_id,
            default_session_id=entity.session_id,
        )
        
        # 初始化记忆管理器
        self.memory_manager = memory_manager or AgentMemoryManager(
            agent_id=entity.id,
            db=entity.db,
            model=entity.model,
            enable_user_memories=entity.enable_user_memories,
            enable_agentic_memory=entity.enable_agentic_memory,
            enable_cultural_knowledge=(
                entity.enable_agentic_culture 
                or entity.update_cultural_knowledge
            ),
            enable_session_summaries=entity.enable_session_summaries,
        )
    
    # ========================================
    # 初始化方法
    # ========================================
    
    def initialize(self, debug_mode: Optional[bool] = None) -> None:
        """初始化Agent服务
        
        Args:
            debug_mode: 调试模式
        """
        # 设置默认模型
        self._set_default_model()
        
        # 设置调试模式
        self._set_debug(debug_mode=debug_mode)
        
        # 设置Agent ID
        self.entity.set_id()
        
        # 初始化记忆管理器
        if self.entity.enable_user_memories or self.entity.enable_agentic_memory:
            self.memory_manager.initialize_memory_manager()
        
        # 初始化文化知识管理器
        if (
            self.entity.add_culture_to_context
            or self.entity.update_cultural_knowledge
            or self.entity.enable_agentic_culture
        ):
            self.memory_manager.initialize_culture_manager()
        
        # 初始化会话摘要管理器
        if self.entity.enable_session_summaries:
            self.memory_manager.initialize_session_summary_manager()
        
        # 初始化格式化器
        if self.entity._formatter is None:
            from agno.utils.safe_formatter import SafeFormatter
            self.entity._formatter = SafeFormatter()
        
        log_debug(f"Agent ID: {self.entity.id}", center=True)
    
    def _set_default_model(self) -> None:
        """设置默认模型"""
        if self.entity.model is None:
            try:
                from agno.models.openai import OpenAIChat
            except ModuleNotFoundError as e:
                log_exception(e)
                log_error(
                    "Gecko agents use `openai` as the default model provider. "
                    "Please provide a `model` or install `openai`."
                )
                exit(1)
            
            log_info("Setting default model to OpenAI Chat")
            self.entity.model = OpenAIChat(id="gpt-4o")
    
    def _set_debug(self, debug_mode: Optional[bool] = None) -> None:
        """设置调试模式
        
        Args:
            debug_mode: 调试模式开关
        """
        from os import getenv
        from gecko.utils.logging import set_log_level_to_debug, set_log_level_to_info
        
        if (
            self.entity.debug_mode 
            or debug_mode 
            or getenv("GECKO_DEBUG", "false").lower() == "true"
        ):
            set_log_level_to_debug(level=self.entity.debug_level)
        else:
            set_log_level_to_info()
    
    # ========================================
    # 输入验证
    # ========================================
    
    def validate_input(
        self,
        input: Union[str, List, Dict, Message, BaseModel],
    ) -> Union[str, List, Dict, Message, BaseModel]:
        """验证输入
        
        如果设置了input_schema，则验证输入是否符合schema
        
        Args:
            input: 输入内容
            
        Returns:
            验证后的输入
            
        Raises:
            ValueError: 输入验证失败
        """
        if self.entity.input_schema is None:
            return input
        
        # 处理Message对象 - 提取内容
        if isinstance(input, Message):
            input = input.content  # type: ignore
        
        # 如果输入是字符串，转换为字典
        if isinstance(input, str):
            import json
            try:
                input = json.loads(input)
            except Exception as e:
                raise ValueError(f"Failed to parse input. Is it a valid JSON string?: {e}")
        
        # Case 1: 输入已经是BaseModel实例
        if isinstance(input, BaseModel):
            if isinstance(input, self.entity.input_schema):
                try:
                    return input
                except Exception as e:
                    raise ValueError(f"BaseModel validation failed: {str(e)}")
            else:
                raise ValueError(
                    f"Expected {self.entity.input_schema.__name__} "
                    f"but got {type(input).__name__}"
                )
        
        # Case 2: 输入是字典
        elif isinstance(input, dict):
            try:
                # 检查schema是否为TypedDict
                if is_typed_dict(self.entity.input_schema):
                    validated_dict = validate_typed_dict(input, self.entity.input_schema)
                    return validated_dict
                else:
                    validated_model = self.entity.input_schema(**input)
                    return validated_model
            except Exception as e:
                raise ValueError(
                    f"Failed to parse dict into {self.entity.input_schema.__name__}: {str(e)}"
                )
        
        # Case 3: 其他类型不支持
        else:
            raise ValueError(
                f"Cannot validate {type(input)} against input_schema. "
                f"Expected dict or {self.entity.input_schema.__name__} instance."
            )
    
    # ========================================
    # 运行取消
    # ========================================
    
    @staticmethod
    def cancel_run(run_id: str) -> bool:
        """取消正在运行的Agent执行
        
        Args:
            run_id: 要取消的运行ID
            
        Returns:
            True如果找到并标记为取消，否则False
        """
        return cancel_run_global(run_id)
    # ========================================
    # 核心运行方法 - 同步run
    # ========================================
    
    def run(
        self,
        input: Union[str, List, Dict, Message, BaseModel, List[Message]],
        *,
        stream: Optional[bool] = None,
        stream_events: Optional[bool] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
        audio: Optional[Sequence[Audio]] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        files: Optional[Sequence[File]] = None,
        retries: Optional[int] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        yield_run_response: bool = False,
        debug_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[RunOutput, Iterator[Union[RunOutputEvent, RunOutput]]]:
        """运行Agent并返回响应
        
        Args:
            input: 用户输入
            stream: 是否流式输出
            stream_events: 是否流式输出事件
            user_id: 用户ID
            session_id: 会话ID
            session_state: 会话状态
            audio: 音频输入
            images: 图片输入
            videos: 视频输入
            files: 文件输入
            retries: 重试次数
            knowledge_filters: 知识库过滤器
            add_history_to_context: 是否添加历史到上下文
            add_dependencies_to_context: 是否添加依赖到上下文
            add_session_state_to_context: 是否添加会话状态到上下文
            dependencies: 依赖项
            metadata: 元数据
            yield_run_response: 是否yield最终响应
            debug_mode: 调试模式
            **kwargs: 其他参数
            
        Returns:
            RunOutput或Iterator[RunOutputEvent]
        """
        if self.repository._has_async_db():
            raise RuntimeError(
                "`run` method is not supported with an async database. "
                "Please use `arun` method instead."
            )
        
        if (add_history_to_context or self.entity.add_history_to_context) and not self.entity.db and not self.entity.team_id:
            log_warning(
                "add_history_to_context is True, but no database has been assigned to the agent. "
                "History will not be added to the context."
            )
        
        # 创建运行ID
        run_id = str(uuid4())
        
        # 验证输入
        validated_input = self.validate_input(input)
        
        # 标准化钩子
        if not self.entity._hooks_normalised:
            if self.entity.pre_hooks:
                self.entity.pre_hooks = normalize_hooks(self.entity.pre_hooks)  # type: ignore
            if self.entity.post_hooks:
                self.entity.post_hooks = normalize_hooks(self.entity.post_hooks)  # type: ignore
            self.entity._hooks_normalised = True
        
        # 初始化会话
        session_id, user_id = self.context_manager.initialize_session(
            session_id=session_id,
            user_id=user_id,
        )
        
        # 初始化Agent
        self.initialize(debug_mode=debug_mode)
        
        # 验证媒体对象ID
        from agno.utils.agent import validate_media_object_id
        image_artifacts, video_artifacts, audio_artifacts, file_artifacts = validate_media_object_id(
            images=images, videos=videos, audios=audio, files=files
        )
        
        # 创建RunInput
        run_input = RunInput(
            input_content=validated_input,
            images=image_artifacts,
            videos=video_artifacts,
            audios=audio_artifacts,
            files=file_artifacts,
        )
        
        # 读取或创建会话
        agent_session = self.repository.read_or_create_session(
            session_id=session_id,
            user_id=user_id,
            agent_id=self.entity.id,
            agent_data=self.context_manager.get_agent_data(
                agent_name=self.entity.name,
                agent_id=self.entity.id,
                model=self.entity.model,
            ),
            session_state=self.entity.session_state,
            metadata=self.entity.metadata,
        )
        
        # 更新元数据
        self.context_manager.update_metadata(
            session=agent_session,
            agent_metadata=self.entity.metadata,
        )
        
        # 初始化会话状态
        session_state = self.context_manager.initialize_session_state(
            session_state=session_state or {},
            user_id=user_id,
            session_id=session_id,
            run_id=run_id,
        )
        
        # 从数据库加载会话状态
        session_state = self.context_manager.load_session_state(
            session=agent_session,
            session_state=session_state,
            overwrite_db_state=self.entity.overwrite_db_session_state,
        )
        
        # 确定运行时依赖
        dependencies = dependencies if dependencies is not None else self.entity.dependencies
        
        # 创建运行上下文
        run_context = self.context_manager.create_run_context(
            run_id=run_id,
            session_id=session_id,
            user_id=user_id,
            session_state=session_state,
            dependencies=dependencies,
            knowledge_filters=knowledge_filters or self.entity.knowledge_filters,
            metadata=metadata,
        )
        
        # 解析依赖
        if run_context.dependencies is not None:
            self.context_manager.resolve_dependencies(
                run_context=run_context,
                agent=self.entity,
            )
        
        # 合并Agent元数据和运行元数据
        if self.entity.metadata is not None and metadata is not None:
            from gecko.utils.merge_dict import merge_dictionaries
            merge_dictionaries(metadata, self.entity.metadata)
        
        # 确定流式设置
        add_dependencies = (
            add_dependencies_to_context
            if add_dependencies_to_context is not None
            else self.entity.add_dependencies_to_context
        )
        add_session_state = (
            add_session_state_to_context
            if add_session_state_to_context is not None
            else self.entity.add_session_state_to_context
        )
        add_history = (
            add_history_to_context
            if add_history_to_context is not None
            else self.entity.add_history_to_context
        )
        
        if stream is None:
            stream = False if self.entity.stream is None else self.entity.stream
        
        stream_events = stream_events
        if stream is False:
            stream_events = False
        
        if stream_events is None:
            stream_events = False if self.entity.stream_events is None else self.entity.stream_events
        
        self.entity.stream = self.entity.stream or stream
        self.entity.stream_events = self.entity.stream_events or stream_events
        
        # 准备响应格式
        response_format = self._get_response_format() if self.entity.parser_model is None else None
        self.entity.model = cast(Model, self.entity.model)
        
        # 创建RunOutput
        run_response = RunOutput(
            run_id=run_id,
            session_id=session_id,
            agent_id=self.entity.id,
            user_id=user_id,
            agent_name=self.entity.name,
            metadata=run_context.metadata,
            session_state=run_context.session_state,
            input=run_input,
        )
        
        run_response.model = self.entity.model.id if self.entity.model is not None else None
        run_response.model_provider = self.entity.model.provider if self.entity.model is not None else None
        
        # 启动运行指标计时器
        run_response.metrics = Metrics()
        run_response.metrics.start_timer()
        
        # 确定重试次数
        retries = retries if retries is not None else self.entity.retries
        
        last_exception = None
        num_attempts = retries + 1
        
        for attempt in range(num_attempts):
            try:
                if stream:
                    response_iterator = self._run_stream(
                        run_response=run_response,
                        run_context=run_context,
                        session=agent_session,
                        user_id=user_id,
                        add_history_to_context=add_history,
                        add_dependencies_to_context=add_dependencies,
                        add_session_state_to_context=add_session_state,
                        response_format=response_format,
                        stream_events=stream_events,
                        yield_run_response=yield_run_response,
                        debug_mode=debug_mode,
                        **kwargs,
                    )
                    return response_iterator
                else:
                    response = self._run(
                        run_response=run_response,
                        run_context=run_context,
                        session=agent_session,
                        user_id=user_id,
                        add_history_to_context=add_history,
                        add_dependencies_to_context=add_dependencies,
                        add_session_state_to_context=add_session_state,
                        response_format=response_format,
                        debug_mode=debug_mode,
                        **kwargs,
                    )
                    return response
            
            except (InputCheckError, OutputCheckError) as e:
                log_error(f"Validation failed: {str(e)} | Check: {e.check_trigger}")
                raise e
            
            except ModelProviderError as e:
                log_warning(f"Attempt {attempt + 1}/{num_attempts} failed: {str(e)}")
                if isinstance(e, StopAgentRun):
                    raise e
                last_exception = e
                if attempt < num_attempts - 1:
                    if self.entity.exponential_backoff:
                        delay = 2**attempt * self.entity.delay_between_retries
                    else:
                        delay = self.entity.delay_between_retries
                    import time
                    time.sleep(delay)
            
            except KeyboardInterrupt:
                run_response.content = "Operation cancelled by user"
                run_response.status = RunStatus.cancelled
                
                if stream:
                    from agno.utils.response import generator_wrapper
                    return generator_wrapper(  # type: ignore
                        create_run_cancelled_event(
                            from_run_response=run_response,
                            reason="Operation cancelled by user",
                        )
                    )
                else:
                    return run_response
        
        # 所有重试失败
        if last_exception is not None:
            log_error(
                f"Failed after {num_attempts} attempts. "
                f"Last error using {last_exception.model_name}({last_exception.model_id})"
            )
            if stream:
                from agno.utils.response import generator_wrapper
                return generator_wrapper(  # type: ignore
                    create_run_error_event(run_response, error=str(last_exception))
                )
            raise last_exception
        else:
            if stream:
                from agno.utils.response import generator_wrapper
                return generator_wrapper(  # type: ignore
                    create_run_error_event(run_response, error=str(last_exception))
                )
            raise Exception(f"Failed after {num_attempts} attempts.")
    # ========================================
    # 内部运行实现 - _run (同步)
    # ========================================
    
    def _run(
        self,
        run_response: RunOutput,
        run_context: RunContext,
        session: AgentSession,
        user_id: Optional[str] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        debug_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> RunOutput:
        """运行Agent并返回RunOutput
        
        执行步骤：
        1. 执行pre-hooks
        2. 确定模型工具
        3. 准备运行消息
        4. 启动后台记忆创建
        5. 执行推理（如果启用）
        6. 生成模型响应（包括工具调用）
        7. 更新RunOutput
        8. 存储媒体
        9. 转换为结构化格式
        10. 执行post-hooks
        11. 等待后台任务完成
        12. 创建会话摘要
        13. 清理和存储
        
        Args:
            run_response: 运行响应对象
            run_context: 运行上下文
            session: Agent会话
            user_id: 用户ID
            add_history_to_context: 添加历史
            add_dependencies_to_context: 添加依赖
            add_session_state_to_context: 添加会话状态
            response_format: 响应格式
            debug_mode: 调试模式
            **kwargs: 其他参数
            
        Returns:
            RunOutput对象
        """
        # 注册运行以支持取消
        register_run(run_response.run_id)  # type: ignore
        
        try:
            # 1. 执行pre-hooks
            run_input = cast(RunInput, run_response.input)
            self.entity.model = cast(Model, self.entity.model)
            
            if self.entity.pre_hooks is not None:
                pre_hook_iterator = self._execute_pre_hooks(
                    hooks=self.entity.pre_hooks,  # type: ignore
                    run_response=run_response,
                    run_input=run_input,
                    run_context=run_context,
                    session=session,
                    user_id=user_id,
                    debug_mode=debug_mode,
                    **kwargs,
                )
                # 消费生成器
                deque(pre_hook_iterator, maxlen=0)
            
            # 2. 确定模型工具
            processed_tools = self._get_tools(
                run_response=run_response,
                run_context=run_context,
                session=session,
                user_id=user_id,
            )
            
            _tools = self._determine_tools_for_model(
                model=self.entity.model,
                processed_tools=processed_tools,
                run_response=run_response,
                session=session,
                run_context=run_context,
            )
            
            # 3. 准备运行消息
            run_messages: RunMessages = self._get_run_messages(
                run_response=run_response,
                run_context=run_context,
                input=run_input.input_content,
                session=session,
                user_id=user_id,
                audio=run_input.audios,
                images=run_input.images,
                videos=run_input.videos,
                files=run_input.files,
                add_history_to_context=add_history_to_context,
                add_dependencies_to_context=add_dependencies_to_context,
                add_session_state_to_context=add_session_state_to_context,
                tools=_tools,
                **kwargs,
            )
            
            if len(run_messages.messages) == 0:
                log_error("No messages to be sent to the model.")
            
            log_debug(f"Agent Run Start: {run_response.run_id}", center=True)
            
            # 4. 启动后台记忆创建
            memory_future = None
            if (
                run_messages.user_message is not None
                and self.memory_manager.memory_manager is not None
                and not self.entity.enable_agentic_memory
            ):
                log_debug("Starting memory creation in background thread.")
                memory_future = self.entity.background_executor.submit(
                    self.memory_manager.create_user_memories,
                    run_messages=run_messages,
                    user_id=user_id,
                )
            
            # 启动文化知识创建
            cultural_knowledge_future = None
            if (
                run_messages.user_message is not None
                and self.memory_manager.culture_manager is not None
                and self.entity.update_cultural_knowledge
            ):
                log_debug("Starting cultural knowledge creation in background thread.")
                cultural_knowledge_future = self.entity.background_executor.submit(
                    self.memory_manager.create_cultural_knowledge,
                    run_messages=run_messages,
                )
            
            # 检查取消
            raise_if_cancelled(run_response.run_id)  # type: ignore
            
            # 5. 执行推理
            self._handle_reasoning(run_response=run_response, run_messages=run_messages)
            
            # 检查取消
            raise_if_cancelled(run_response.run_id)  # type: ignore
            
            # 6. 生成模型响应
            self.entity.model = cast(Model, self.entity.model)
            model_response: ModelResponse = self.entity.model.response(
                messages=run_messages.messages,
                tools=_tools,
                tool_choice=self.entity.tool_choice,
                tool_call_limit=self.entity.tool_call_limit,
                response_format=response_format,
                run_response=run_response,
                send_media_to_model=self.entity.send_media_to_model,
            )
            
            # 检查取消
            raise_if_cancelled(run_response.run_id)  # type: ignore
            
            # 如果有输出模型，使用它生成输出
            self._generate_response_with_output_model(model_response, run_messages)
            
            # 如果有解析器模型，结构化响应
            self._parse_response_with_parser_model(model_response, run_messages)
            
            # 7. 更新RunOutput
            self._update_run_response(
                model_response=model_response,
                run_response=run_response,
                run_messages=run_messages,
            )
            
            # 检查是否暂停
            if any(tool_call.is_paused for tool_call in run_response.tools or []):
                from agno.utils.agent import wait_for_background_tasks
                wait_for_background_tasks(
                    memory_future=memory_future,
                    cultural_knowledge_future=cultural_knowledge_future,
                )
                return self._handle_agent_run_paused(
                    run_response=run_response,
                    session=session,
                    user_id=user_id,
                )
            
            # 8. 存储媒体
            if self.entity.store_media:
                from agno.utils.agent import store_media_util
                store_media_util(run_response, model_response)
            
            # 9. 转换为结构化格式
            self._convert_response_to_structured_format(run_response)
            
            # 10. 执行post-hooks
            if self.entity.post_hooks is not None:
                post_hook_iterator = self._execute_post_hooks(
                    hooks=self.entity.post_hooks,  # type: ignore
                    run_output=run_response,
                    run_context=run_context,
                    session=session,
                    user_id=user_id,
                    debug_mode=debug_mode,
                    **kwargs,
                )
                deque(post_hook_iterator, maxlen=0)
            
            # 检查取消
            raise_if_cancelled(run_response.run_id)  # type: ignore
            
            # 11. 等待后台任务
            from agno.utils.agent import wait_for_background_tasks
            wait_for_background_tasks(
                memory_future=memory_future,
                cultural_knowledge_future=cultural_knowledge_future,
            )
            
            # 12. 创建会话摘要
            if self.memory_manager.session_summary_manager is not None:
                session.upsert_run(run=run_response)
                try:
                    self.memory_manager.create_session_summary(session=session)
                except Exception as e:
                    log_warning(f"Error in session summary creation: {str(e)}")
            
            run_response.status = RunStatus.completed
            
            # 13. 清理和存储
            self._cleanup_and_store(
                run_response=run_response,
                session=session,
                run_context=run_context,
                user_id=user_id,
            )
            
            # 记录遥测
            self._log_agent_telemetry(session_id=session.session_id, run_id=run_response.run_id)
            
            log_debug(f"Agent Run End: {run_response.run_id}", center=True, symbol="*")
            
            return run_response
        
        except RunCancelledException as e:
            # 处理运行取消
            log_info(f"Run {run_response.run_id} was cancelled")
            run_response.content = str(e)
            run_response.status = RunStatus.cancelled
            
            self._cleanup_and_store(
                run_response=run_response,
                session=session,
                run_context=run_context,
                user_id=user_id,
            )
            
            return run_response
        
        finally:
            # 清理运行跟踪
            cleanup_run(run_response.run_id)  # type: ignore
    
    # ========================================
    # 内部运行实现 - _run_stream (同步流式)
    # ========================================
    
    def _run_stream(
        self,
        run_response: RunOutput,
        run_context: RunContext,
        session: AgentSession,
        user_id: Optional[str] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        stream_events: bool = False,
        yield_run_response: bool = False,
        debug_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> Iterator[Union[RunOutputEvent, RunOutput]]:
        """运行Agent并yield RunOutput事件
        
        执行步骤：
        1. 执行pre-hooks
        2. 确定模型工具
        3. 准备运行消息
        4. 启动后台记忆创建
        5. 执行推理（流式）
        6. 处理模型响应（流式）
        7. 解析响应
        8. 等待后台任务
        9. 创建会话摘要
        10. 清理和存储
        
        Args:
            与_run相同，额外参数：
            stream_events: 是否流式输出事件
            yield_run_response: 是否yield最终响应
            
        Yields:
            RunOutputEvent或RunOutput
        """
        # 注册运行
        register_run(run_response.run_id)  # type: ignore
        
        try:
            # 1. 执行pre-hooks
            run_input = cast(RunInput, run_response.input)
            self.entity.model = cast(Model, self.entity.model)
            
            if self.entity.pre_hooks is not None:
                pre_hook_iterator = self._execute_pre_hooks(
                    hooks=self.entity.pre_hooks,  # type: ignore
                    run_response=run_response,
                    run_input=run_input,
                    run_context=run_context,
                    session=session,
                    user_id=user_id,
                    debug_mode=debug_mode,
                    **kwargs,
                )
                for event in pre_hook_iterator:
                    yield event
            
            # 2. 确定模型工具
            processed_tools = self._get_tools(
                run_response=run_response,
                run_context=run_context,
                session=session,
                user_id=user_id,
            )
            
            _tools = self._determine_tools_for_model(
                model=self.entity.model,
                processed_tools=processed_tools,
                run_response=run_response,
                session=session,
                run_context=run_context,
            )
            
            # 3. 准备运行消息
            run_messages: RunMessages = self._get_run_messages(
                run_response=run_response,
                run_context=run_context,
                input=run_input.input_content,
                session=session,
                user_id=user_id,
                audio=run_input.audios,
                images=run_input.images,
                videos=run_input.videos,
                files=run_input.files,
                add_history_to_context=add_history_to_context,
                add_dependencies_to_context=add_dependencies_to_context,
                add_session_state_to_context=add_session_state_to_context,
                tools=_tools,
                **kwargs,
            )
            
            if len(run_messages.messages) == 0:
                log_error("No messages to be sent to the model.")
            
            log_debug(f"Agent Run Start: {run_response.run_id}", center=True)
            
            # 4. 启动后台记忆创建
            memory_future = None
            if (
                run_messages.user_message is not None
                and self.memory_manager.memory_manager is not None
                and not self.entity.enable_agentic_memory
            ):
                log_debug("Starting memory creation in background thread.")
                memory_future = self.entity.background_executor.submit(
                    self.memory_manager.create_user_memories,
                    run_messages=run_messages,
                    user_id=user_id,
                )
            
            cultural_knowledge_future = None
            if (
                run_messages.user_message is not None
                and self.memory_manager.culture_manager is not None
                and self.entity.update_cultural_knowledge
            ):
                log_debug("Starting cultural knowledge creation in background thread.")
                cultural_knowledge_future = self.entity.background_executor.submit(
                    self.memory_manager.create_cultural_knowledge,
                    run_messages=run_messages,
                )
            
            # Yield运行开始事件
            if stream_events:
                yield handle_event(  # type: ignore
                    create_run_started_event(run_response),
                    run_response,
                    events_to_skip=self.entity.events_to_skip,  # type: ignore
                    store_events=self.entity.store_events,
                )
            
            # 5. 执行推理（流式）
            yield from self._handle_reasoning_stream(
                run_response=run_response,
                run_messages=run_messages,
                stream_events=stream_events,
            )
            
            # 检查取消
            raise_if_cancelled(run_response.run_id)  # type: ignore
            
            # 6. 处理模型响应（流式）
            if self.entity.output_model is None:
                for event in self._handle_model_response_stream(
                    session=session,
                    run_response=run_response,
                    run_messages=run_messages,
                    tools=_tools,
                    response_format=response_format,
                    stream_events=stream_events,
                    session_state=run_context.session_state,
                ):
                    raise_if_cancelled(run_response.run_id)  # type: ignore
                    yield event
            else:
                from agno.run.agent import IntermediateRunContentEvent, RunContentEvent
                
                for event in self._handle_model_response_stream(
                    session=session,
                    run_response=run_response,
                    run_messages=run_messages,
                    tools=_tools,
                    response_format=response_format,
                    stream_events=stream_events,
                    session_state=run_context.session_state,
                ):
                    raise_if_cancelled(run_response.run_id)  # type: ignore
                    if isinstance(event, RunContentEvent):
                        if stream_events:
                            yield IntermediateRunContentEvent(
                                content=event.content,
                                content_type=event.content_type,
                            )
                    else:
                        yield event
                
                # 使用输出模型生成
                for event in self._generate_response_with_output_model_stream(
                    session=session,
                    run_response=run_response,
                    run_messages=run_messages,
                    stream_events=stream_events,
                ):
                    raise_if_cancelled(run_response.run_id)  # type: ignore
                    yield event
            
            # 检查取消
            raise_if_cancelled(run_response.run_id)  # type: ignore
            
            # 7. 解析响应
            yield from self._parse_response_with_parser_model_stream(
                session=session,
                run_response=run_response,
                stream_events=stream_events,
            )
            
            # 检查是否暂停
            if any(tool_call.is_paused for tool_call in run_response.tools or []):
                from agno.utils.agent import wait_for_background_tasks_stream
                yield from wait_for_background_tasks_stream(
                    memory_future=memory_future,
                    cultural_knowledge_future=cultural_knowledge_future,
                    stream_events=stream_events,
                    run_response=run_response,
                    events_to_skip=self.entity.events_to_skip,
                    store_events=self.entity.store_events,
                )
                
                yield from self._handle_agent_run_paused_stream(
                    run_response=run_response,
                    session=session,
                    user_id=user_id,
                )
                return
            
            # Yield内容完成事件
            if stream_events:
                yield handle_event(  # type: ignore
                    create_run_content_completed_event(from_run_response=run_response),
                    run_response,
                    events_to_skip=self.entity.events_to_skip,  # type: ignore
                    store_events=self.entity.store_events,
                )
            
            # 执行post-hooks
            if self.entity.post_hooks is not None:
                yield from self._execute_post_hooks(
                    hooks=self.entity.post_hooks,  # type: ignore
                    run_output=run_response,
                    run_context=run_context,
                    session=session,
                    user_id=user_id,
                    debug_mode=debug_mode,
                    **kwargs,
                )
            
            # 8. 等待后台任务
            from agno.utils.agent import wait_for_background_tasks_stream
            yield from wait_for_background_tasks_stream(
                memory_future=memory_future,
                cultural_knowledge_future=cultural_knowledge_future,
                stream_events=stream_events,
                run_response=run_response,
            )
            
            # 9. 创建会话摘要
            if self.memory_manager.session_summary_manager is not None:
                session.upsert_run(run=run_response)
                
                if stream_events:
                    yield handle_event(  # type: ignore
                        create_session_summary_started_event(from_run_response=run_response),
                        run_response,
                        events_to_skip=self.entity.events_to_skip,  # type: ignore
                        store_events=self.entity.store_events,
                    )
                
                try:
                    self.memory_manager.create_session_summary(session=session)
                except Exception as e:
                    log_warning(f"Error in session summary creation: {str(e)}")
                
                if stream_events:
                    yield handle_event(  # type: ignore
                        create_session_summary_completed_event(
                            from_run_response=run_response,
                            session_summary=session.summary,
                        ),
                        run_response,
                        events_to_skip=self.entity.events_to_skip,  # type: ignore
                        store_events=self.entity.store_events,
                    )
            
            # 更新会话状态
            if session.session_data is not None and "session_state" in session.session_data:
                run_response.session_state = session.session_data["session_state"]
            
            # 创建完成事件
            completed_event = handle_event(  # type: ignore
                create_run_completed_event(from_run_response=run_response),
                run_response,
                events_to_skip=self.entity.events_to_skip,  # type: ignore
                store_events=self.entity.store_events,
            )
            
            run_response.status = RunStatus.completed
            
            # 10. 清理和存储
            self._cleanup_and_store(
                run_response=run_response,
                session=session,
                run_context=run_context,
                user_id=user_id,
            )
            
            if stream_events:
                yield completed_event  # type: ignore
            
            if yield_run_response:
                yield run_response
            
            # 记录遥测
            self._log_agent_telemetry(session_id=session.session_id, run_id=run_response.run_id)
            
            log_debug(f"Agent Run End: {run_response.run_id}", center=True, symbol="*")
        
        except RunCancelledException as e:
            # 处理取消
            log_info(f"Run {run_response.run_id} was cancelled during streaming")
            run_response.status = RunStatus.cancelled
            if not run_response.content:
                run_response.content = str(e)
            
            yield handle_event(  # type: ignore
                create_run_cancelled_event(from_run_response=run_response, reason=str(e)),
                run_response,
                events_to_skip=self.entity.events_to_skip,  # type: ignore
                store_events=self.entity.store_events,
            )
            
            self._cleanup_and_store(
                run_response=run_response,
                session=session,
                run_context=run_context,
                user_id=user_id,
            )
        
        finally:
            cleanup_run(run_response.run_id)  # type: ignoree

    # ========================================
    # 核心运行方法 - 异步arun
    # ========================================
    
    def arun(
        self,
        input: Union[str, List, Dict, Message, BaseModel, List[Message]],
        *,
        stream: Optional[bool] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
        audio: Optional[Sequence[Audio]] = None,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        files: Optional[Sequence[File]] = None,
        stream_events: Optional[bool] = None,
        retries: Optional[int] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        yield_run_response: Optional[bool] = None,
        debug_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[RunOutput, AsyncIterator[Union[RunOutputEvent, RunOutput]]]:
        """异步运行Agent并返回响应
        
        Args:
            参数与run方法相同
            
        Returns:
            RunOutput或AsyncIterator[RunOutputEvent]
        """
        if (add_history_to_context or self.entity.add_history_to_context) and not self.entity.db and not self.entity.team_id:
            log_warning(
                "add_history_to_context is True, but no database has been assigned to the agent. "
                "History will not be added to the context."
            )
        
        # 创建运行ID
        run_id = str(uuid4())
        
        # 验证输入
        validated_input = self.validate_input(input)
        
        # 标准化钩子
        if not self.entity._hooks_normalised:
            if self.entity.pre_hooks:
                self.entity.pre_hooks = normalize_hooks(self.entity.pre_hooks, async_mode=True)  # type: ignore
            if self.entity.post_hooks:
                self.entity.post_hooks = normalize_hooks(self.entity.post_hooks, async_mode=True)  # type: ignore
            self.entity._hooks_normalised = True
        
        # 初始化会话
        session_id, user_id = self.context_manager.initialize_session(
            session_id=session_id,
            user_id=user_id,
        )
        
        # 初始化Agent
        self.initialize(debug_mode=debug_mode)
        
        # 验证媒体对象ID
        from agno.utils.agent import validate_media_object_id
        image_artifacts, video_artifacts, audio_artifacts, file_artifacts = validate_media_object_id(
            images=images, videos=videos, audios=audio, files=files
        )
        
        # 解析变量
        dependencies = dependencies if dependencies is not None else self.entity.dependencies
        add_dependencies = (
            add_dependencies_to_context
            if add_dependencies_to_context is not None
            else self.entity.add_dependencies_to_context
        )
        add_session_state = (
            add_session_state_to_context
            if add_session_state_to_context is not None
            else self.entity.add_session_state_to_context
        )
        add_history = (
            add_history_to_context
            if add_history_to_context is not None
            else self.entity.add_history_to_context
        )
        
        # 创建RunInput
        run_input = RunInput(
            input_content=validated_input,
            images=image_artifacts,
            videos=video_artifacts,
            audios=audio_artifacts,
            files=file_artifacts,
        )
        
        # 确定流式设置
        if stream is None:
            stream = False if self.entity.stream is None else self.entity.stream
        
        stream_events = stream_events
        if stream is False:
            stream_events = False
        
        if stream_events is None:
            stream_events = False if self.entity.stream_events is None else self.entity.stream_events
        
        self.entity.stream = self.entity.stream or stream
        self.entity.stream_events = self.entity.stream_events or stream_events
        
        # 准备响应格式
        response_format = self._get_response_format() if self.entity.parser_model is None else None
        self.entity.model = cast(Model, self.entity.model)
        
        # 获取知识库过滤器
        knowledge_filters = knowledge_filters
        if self.entity.knowledge_filters or knowledge_filters:
            knowledge_filters = self._get_effective_filters(knowledge_filters)
        
        # 合并元数据
        if self.entity.metadata is not None:
            if metadata is None:
                metadata = self.entity.metadata
            else:
                from gecko.utils.merge_dict import merge_dictionaries
                merge_dictionaries(metadata, self.entity.metadata)
        
        # 创建运行上下文
        run_context = self.context_manager.create_run_context(
            run_id=run_id,
            session_id=session_id,
            user_id=user_id,
            session_state=session_state,
            dependencies=dependencies,
            knowledge_filters=knowledge_filters,
            metadata=metadata,
        )
        
        # 确定重试次数
        retries = retries if retries is not None else self.entity.retries
        
        # 创建RunOutput
        run_response = RunOutput(
            run_id=run_id,
            session_id=session_id,
            agent_id=self.entity.id,
            user_id=user_id,
            agent_name=self.entity.name,
            metadata=run_context.metadata,
            session_state=run_context.session_state,
            input=run_input,
        )
        
        run_response.model = self.entity.model.id if self.entity.model is not None else None
        run_response.model_provider = self.entity.model.provider if self.entity.model is not None else None
        
        # 启动运行指标计时器
        run_response.metrics = Metrics()
        run_response.metrics.start_timer()
        
        last_exception = None
        num_attempts = retries + 1
        
        for attempt in range(num_attempts):
            try:
                if stream:
                    return self._arun_stream(  # type: ignore
                        run_response=run_response,
                        run_context=run_context,
                        user_id=user_id,
                        response_format=response_format,
                        stream_events=stream_events,
                        yield_run_response=yield_run_response,
                        session_id=session_id,
                        add_history_to_context=add_history,
                        add_dependencies_to_context=add_dependencies,
                        add_session_state_to_context=add_session_state,
                        debug_mode=debug_mode,
                        **kwargs,
                    )  # type: ignore
                else:
                    return self._arun(  # type: ignore
                        run_response=run_response,
                        run_context=run_context,
                        user_id=user_id,
                        response_format=response_format,
                        session_id=session_id,
                        add_history_to_context=add_history,
                        add_dependencies_to_context=add_dependencies,
                        add_session_state_to_context=add_session_state,
                        debug_mode=debug_mode,
                        **kwargs,
                    )
            
            except (InputCheckError, OutputCheckError) as e:
                log_error(f"Validation failed: {str(e)} | Check trigger: {e.check_trigger}")
                raise e
            
            except ModelProviderError as e:
                log_warning(f"Attempt {attempt + 1}/{num_attempts} failed: {str(e)}")
                if isinstance(e, StopAgentRun):
                    raise e
                last_exception = e
                if attempt < num_attempts - 1:
                    if self.entity.exponential_backoff:
                        delay = 2**attempt * self.entity.delay_between_retries
                    else:
                        delay = self.entity.delay_between_retries
                    import time
                    time.sleep(delay)
            
            except KeyboardInterrupt:
                run_response.content = "Operation cancelled by user"
                run_response.status = RunStatus.cancelled
                
                if stream:
                    from agno.utils.response import async_generator_wrapper
                    return async_generator_wrapper(  # type: ignore
                        create_run_cancelled_event(
                            from_run_response=run_response,
                            reason="Operation cancelled by user",
                        )
                    )
                else:
                    return run_response
        
        # 所有重试失败
        if last_exception is not None:
            log_error(
                f"Failed after {num_attempts} attempts. "
                f"Last error using {last_exception.model_name}({last_exception.model_id})"
            )
            
            if stream:
                from agno.utils.response import async_generator_wrapper
                return async_generator_wrapper(  # type: ignore
                    create_run_error_event(run_response, error=str(last_exception))
                )
            raise last_exception
        else:
            if stream:
                from agno.utils.response import async_generator_wrapper
                return async_generator_wrapper(  # type: ignore
                    create_run_error_event(run_response, error=str(last_exception))
                )
            raise Exception(f"Failed after {num_attempts} attempts.")
    
    # ========================================
    # 内部运行实现 - _arun (异步)
    # ========================================
    
    async def _arun(
        self,
        run_response: RunOutput,
        run_context: RunContext,
        session_id: str,
        user_id: Optional[str] = None,
        add_history_to_context: Optional[bool] = None,
        add_dependencies_to_context: Optional[bool] = None,
        add_session_state_to_context: Optional[bool] = None,
        response_format: Optional[Union[Dict, Type[BaseModel]]] = None,
        debug_mode: Optional[bool] = None,
        **kwargs: Any,
    ) -> RunOutput:
        """异步运行Agent并返回RunOutput
        
        执行步骤：
        1. 读取或创建会话
        2. 更新元数据和会话状态
        3. 解析依赖
        4. 执行pre-hooks
        5. 确定模型工具
        6. 准备运行消息
        7. 启动后台记忆创建
        8. 执行推理
        9. 生成模型响应
        10. 更新RunOutput
        11. 转换为结构化格式
        12. 存储媒体
        13. 执行post-hooks
        14. 等待后台任务
        15. 创建会话摘要
        16. 清理和存储
        
        Args:
            与_run相同
            
        Returns:
            RunOutput对象
        """
        log_debug(f"Agent Run Start: {run_response.run_id}", center=True)
        
        # 注册运行
        register_run(run_response.run_id)  # type: ignore
        
        try:
            # 1. 读取或创建会话
            agent_session = await self.repository.aread_or_create_session(
                session_id=session_id,
                user_id=user_id,
                agent_id=self.entity.id,
                agent_data=self.context_manager.get_agent_data(
                    agent_name=self.entity.name,
                    agent_id=self.entity.id,
                    model=self.entity.model,
                ),
                session_state=self.entity.session_state,
                metadata=self.entity.metadata,
            )
            
            # 2. 更新元数据和会话状态
            self.context_manager.update_metadata(
                session=agent_session,
                agent_metadata=self.entity.metadata,
            )
            
            # 初始化会话状态
            run_context.session_state = self.context_manager.initialize_session_state(
                session_state=run_context.session_state or {},
                user_id=user_id,
                session_id=session_id,
                run_id=run_response.run_id,
            )
            
            # 从数据库加载会话状态
            if run_context.session_state is not None:
                run_context.session_state = self.context_manager.load_session_state(
                    session=agent_session,
                    session_state=run_context.session_state,
                    overwrite_db_state=self.entity.overwrite_db_session_state,
                )
            
            # 3. 解析依赖
            if run_context.dependencies is not None:
                await self.context_manager.aresolve_dependencies(
                    run_context=run_context,
                    agent=self.entity,
                )
            
            # 4. 执行pre-hooks
            run_input = cast(RunInput, run_response.input)
            self.entity.model = cast(Model, self.entity.model)
            
            if self.entity.pre_hooks is not None:
                pre_hook_iterator = self._aexecute_pre_hooks(
                    hooks=self.entity.pre_hooks,  # type: ignore
                    run_response=run_response,
                    run_context=run_context,
                    run_input=run_input,
                    session=agent_session,
                    user_id=user_id,
                    debug_mode=debug_mode,
                    **kwargs,
                )
                async for _ in pre_hook_iterator:
                    pass
            
            # 5. 确定模型工具
            self.entity.model = cast(Model, self.entity.model)
            processed_tools = await self._aget_tools(
                run_response=run_response,
                run_context=run_context,
                session=agent_session,
                user_id=user_id,
            )
            
            _tools = self._determine_tools_for_model(
                model=self.entity.model,
                processed_tools=processed_tools,
                run_response=run_response,
                run_context=run_context,
                session=agent_session,
            )
            
            # 6. 准备运行消息
            run_messages: RunMessages = await self._aget_run_messages(
                run_response=run_response,
                run_context=run_context,
                input=run_input.input_content,
                session=agent_session,
                user_id=user_id,
                audio=run_input.audios,
                images=run_input.images,
                videos=run_input.videos,
                files=run_input.files,
                add_history_to_context=add_history_to_context,
                add_dependencies_to_context=add_dependencies_to_context,
                add_session_state_to_context=add_session_state_to_context,
                tools=_tools,
                **kwargs,
            )
            
            if len(run_messages.messages) == 0:
                log_error("No messages to be sent to the model.")
            
            # 7. 启动后台记忆创建
            memory_task = None
            if (
                run_messages.user_message is not None
                and self.memory_manager.memory_manager is not None
                and not self.entity.enable_agentic_memory
            ):
                log_debug("Starting memory creation in background task.")
                memory_task = create_task(
                    self.memory_manager.acreate_user_memories(
                        run_messages=run_messages,
                        user_id=user_id,
                    )
                )
            
            cultural_knowledge_task = None
            if (
                run_messages.user_message is not None
                and self.memory_manager.culture_manager is not None
                and self.entity.update_cultural_knowledge
            ):
                log_debug("Starting cultural knowledge creation in background task.")
                cultural_knowledge_task = create_task(
                    self.memory_manager.acreate_cultural_knowledge(
                        run_messages=run_messages,
                    )
                )
            
            # 检查取消
            raise_if_cancelled(run_response.run_id)  # type: ignore
            
            # 8. 执行推理
            await self._ahandle_reasoning(run_response=run_response, run_messages=run_messages)
            
            # 检查取消
            raise_if_cancelled(run_response.run_id)  # type: ignore
            
            # 9. 生成模型响应
            model_response: ModelResponse = await self.entity.model.aresponse(
                messages=run_messages.messages,
                tools=_tools,
                tool_choice=self.entity.tool_choice,
                tool_call_limit=self.entity.tool_call_limit,
                response_format=response_format,
                send_media_to_model=self.entity.send_media_to_model,
                run_response=run_response,
            )
            
            # 检查取消
            raise_if_cancelled(run_response.run_id)  # type: ignore
            
            # 如果有输出模型
            await self._agenerate_response_with_output_model(
                model_response=model_response,
                run_messages=run_messages,
            )
            
            # 如果有解析器模型
            await self._aparse_response_with_parser_model(
                model_response=model_response,
                run_messages=run_messages,
            )
            
            # 10. 更新RunOutput
            self._update_run_response(
                model_response=model_response,
                run_response=run_response,
                run_messages=run_messages,
            )
            
            # 检查是否暂停
            if any(tool_call.is_paused for tool_call in run_response.tools or []):
                from agno.utils.agent import await_for_background_tasks
                await await_for_background_tasks(
                    memory_task=memory_task,
                    cultural_knowledge_task=cultural_knowledge_task,
                )
                return await self._ahandle_agent_run_paused(
                    run_response=run_response,
                    session=agent_session,
                    user_id=user_id,
                )
            
            # 11. 转换为结构化格式
            self._convert_response_to_structured_format(run_response)
            
            # 12. 存储媒体
            if self.entity.store_media:
                from agno.utils.agent import store_media_util
                store_media_util(run_response, model_response)
            
            # 13. 执行post-hooks
            if self.entity.post_hooks is not None:
                async for _ in self._aexecute_post_hooks(
                    hooks=self.entity.post_hooks,  # type: ignore
                    run_output=run_response,
                    run_context=run_context,
                    session=agent_session,
                    user_id=user_id,
                    debug_mode=debug_mode,
                    **kwargs,
                ):
                    pass
            
            # 检查取消
            raise_if_cancelled(run_response.run_id)  # type: ignore
            
            # 14. 等待后台任务
            from agno.utils.agent import await_for_background_tasks
            await await_for_background_tasks(
                memory_task=memory_task,
                cultural_knowledge_task=cultural_knowledge_task,
            )
            
            # 15. 创建会话摘要
            if self.memory_manager.session_summary_manager is not None:
                agent_session.upsert_run(run=run_response)
                try:
                    await self.memory_manager.acreate_session_summary(session=agent_session)
                except Exception as e:
                    log_warning(f"Error in session summary creation: {str(e)}")
            
            run_response.status = RunStatus.completed
            
            # 16. 清理和存储
            await self._acleanup_and_store(
                run_response=run_response,
                session=agent_session,
                run_context=run_context,
                user_id=user_id,
            )
            
            # 记录遥测
            await self._alog_agent_telemetry(session_id=agent_session.session_id, run_id=run_response.run_id)
            
            log_debug(f"Agent Run End: {run_response.run_id}", center=True, symbol="*")
            
            return run_response
        
        except RunCancelledException as e:
            # 处理取消
            log_info(f"Run {run_response.run_id} was cancelled")
            run_response.content = str(e)
            run_response.status = RunStatus.cancelled
            
            await self._acleanup_and_store(
                run_response=run_response,
                session=agent_session,
                run_context=run_context,
                user_id=user_id,
            )
            
            return run_response
        
        finally:
            # 断开MCP工具连接
            await self._disconnect_mcp_tools()
            
            # 取消记忆任务
            if memory_task is not None and not memory_task.done():
                memory_task.cancel()
                try:
                    await memory_task
                except CancelledError:
                    pass
            
            # 取消文化知识任务
            if cultural_knowledge_task is not None and not cultural_knowledge_task.done():
                cultural_knowledge_task.cancel()
                try:
                    await cultural_knowledge_task
                except CancelledError:
                    pass
            
            # 清理运行跟踪
            cleanup_run(run_response.run_id)  # type: ignore
            
            