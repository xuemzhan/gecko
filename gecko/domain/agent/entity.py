"""
Agent Entity - 智能体实体定义

职责：
1. 定义Agent的所有属性和配置
2. 提供基本的实体操作（ID生成、工具管理等）
3. 实现深拷贝功能
4. 不包含外部依赖的业务逻辑
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
)
from uuid import uuid4

from pydantic import BaseModel

from agno.culture.manager import CultureManager
from agno.db.base import AsyncBaseDb, BaseDb
from agno.guardrails import BaseGuardrail
from agno.knowledge.knowledge import Knowledge
from agno.media import Audio, File, Image, Video
from agno.memory import MemoryManager
from agno.models.base import Model
from agno.models.message import Message
from agno.run import RunEvent
from agno.session import SessionSummaryManager
from agno.tools import Toolkit
from agno.tools.function import Function
from agno.utils.log import log_debug, log_warning
from agno.utils.string import generate_id_from_name


@dataclass(init=False)
class AgentEntity:
    """
    智能体实体 - 包含所有配置属性和基本操作
    
    按职责分组的属性：
    - Agent Settings: 基本配置
    - User Settings: 用户配置
    - Session Settings: 会话配置
    - Agent Dependencies: 依赖项
    - Agent Memory: 记忆管理
    - Database: 数据库
    - Agent History: 历史记录
    - Knowledge: 知识库
    - Agent Tools: 工具
    - Agent Hooks: 钩子
    - Agent Reasoning: 推理
    - Default Tools: 默认工具
    - System Message Settings: 系统消息
    - User Message Settings: 用户消息
    - Agent Response Settings: 响应设置
    - Agent Response Model Settings: 响应模型
    - Agent Streaming: 流式输出
    - Team/Workflow Settings: 团队/工作流
    - Metadata: 元数据
    - Experimental Features: 实验性功能
    - Debug: 调试
    - Telemetry: 遥测
    """
    
    # --- Agent Settings ---
    model: Optional[Model] = None
    name: Optional[str] = None
    id: Optional[str] = None
    introduction: Optional[str] = None
    
    # --- User Settings ---
    user_id: Optional[str] = None
    
    # --- Session Settings ---
    session_id: Optional[str] = None
    session_state: Optional[Dict[str, Any]] = None
    add_session_state_to_context: bool = False
    enable_agentic_state: bool = False
    overwrite_db_session_state: bool = False
    cache_session: bool = False
    search_session_history: Optional[bool] = False
    num_history_sessions: Optional[int] = None
    enable_session_summaries: bool = False
    add_session_summary_to_context: Optional[bool] = None
    session_summary_manager: Optional[SessionSummaryManager] = None
    
    # --- Agent Dependencies ---
    dependencies: Optional[Dict[str, Any]] = None
    add_dependencies_to_context: bool = False
    
    # --- Agent Memory ---
    memory_manager: Optional[MemoryManager] = None
    enable_agentic_memory: bool = False
    enable_user_memories: bool = False
    add_memories_to_context: Optional[bool] = None
    
    # --- Database ---
    db: Optional[Union[BaseDb, AsyncBaseDb]] = None
    
    # --- Agent History ---
    add_history_to_context: bool = False
    num_history_runs: Optional[int] = None
    num_history_messages: Optional[int] = None
    max_tool_calls_from_history: Optional[int] = None
    
    # --- Knowledge ---
    knowledge: Optional[Knowledge] = None
    knowledge_filters: Optional[Dict[str, Any]] = None
    enable_agentic_knowledge_filters: Optional[bool] = False
    add_knowledge_to_context: bool = False
    knowledge_retriever: Optional[Callable[..., Optional[List[Union[Dict, str]]]]] = None
    references_format: Literal["json", "yaml"] = "json"
    
    # --- Agent Tools ---
    tools: Optional[List[Union[Toolkit, Callable, Function, Dict]]] = None
    tool_call_limit: Optional[int] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    tool_hooks: Optional[List[Callable]] = None
    
    # --- Agent Hooks ---
    pre_hooks: Optional[List[Union[Callable[..., Any], BaseGuardrail]]] = None
    post_hooks: Optional[List[Union[Callable[..., Any], BaseGuardrail]]] = None
    
    # --- Agent Reasoning ---
    reasoning: bool = False
    reasoning_model: Optional[Model] = None
    reasoning_agent: Optional[AgentEntity] = None
    reasoning_min_steps: int = 1
    reasoning_max_steps: int = 10
    
    # --- Default Tools ---
    read_chat_history: bool = False
    search_knowledge: bool = True
    update_knowledge: bool = False
    read_tool_call_history: bool = False
    send_media_to_model: bool = True
    store_media: bool = True
    store_tool_messages: bool = True
    store_history_messages: bool = True
    
    # --- System Message Settings ---
    system_message: Optional[Union[str, Callable, Message]] = None
    system_message_role: str = "system"
    build_context: bool = True
    
    # --- Settings for Building Default System Message ---
    description: Optional[str] = None
    instructions: Optional[Union[str, List[str], Callable]] = None
    expected_output: Optional[str] = None
    additional_context: Optional[str] = None
    markdown: bool = False
    add_name_to_context: bool = False
    add_datetime_to_context: bool = False
    add_location_to_context: bool = False
    timezone_identifier: Optional[str] = None
    resolve_in_context: bool = True
    
    # --- Extra Messages ---
    additional_input: Optional[List[Union[str, Dict, BaseModel, Message]]] = None
    
    # --- User Message Settings ---
    user_message_role: str = "user"
    build_user_context: bool = True
    
    # --- Agent Response Settings ---
    retries: int = 0
    delay_between_retries: int = 1
    exponential_backoff: bool = False
    
    # --- Agent Response Model Settings ---
    input_schema: Optional[Type[BaseModel]] = None
    output_schema: Optional[Type[BaseModel]] = None
    parser_model: Optional[Model] = None
    parser_model_prompt: Optional[str] = None
    output_model: Optional[Model] = None
    output_model_prompt: Optional[str] = None
    parse_response: bool = True
    structured_outputs: Optional[bool] = None
    use_json_mode: bool = False
    save_response_to_file: Optional[str] = None
    
    # --- Agent Streaming ---
    stream: Optional[bool] = None
    stream_events: Optional[bool] = None
    stream_intermediate_steps: Optional[bool] = None
    store_events: bool = False
    events_to_skip: Optional[List[RunEvent]] = None
    
    # --- Team/Workflow Settings ---
    role: Optional[str] = None
    team_id: Optional[str] = None
    workflow_id: Optional[str] = None
    
    # --- Metadata ---
    metadata: Optional[Dict[str, Any]] = None
    
    # --- Experimental Features: Agent Culture ---
    culture_manager: Optional[CultureManager] = None
    enable_agentic_culture: bool = False
    update_cultural_knowledge: bool = False
    add_culture_to_context: Optional[bool] = None
    
    # --- Debug ---
    debug_mode: bool = False
    debug_level: Literal[1, 2] = 1
    
    # --- Telemetry ---
    telemetry: bool = True
    
    # --- Internal State (Not initialized in __init__) ---
    _cached_session: Optional[Any] = None
    _tool_instructions: Optional[List[str]] = None
    _formatter: Optional[Any] = None
    _hooks_normalised: bool = False
    _mcp_tools_initialized_on_run: List[Any] = None
    _background_executor: Optional[Any] = None
    
    def __init__(
        self,
        *,
        model: Optional[Union[Model, str]] = None,
        name: Optional[str] = None,
        id: Optional[str] = None,
        introduction: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
        add_session_state_to_context: bool = False,
        overwrite_db_session_state: bool = False,
        enable_agentic_state: bool = False,
        cache_session: bool = False,
        search_session_history: Optional[bool] = False,
        num_history_sessions: Optional[int] = None,
        dependencies: Optional[Dict[str, Any]] = None,
        add_dependencies_to_context: bool = False,
        db: Optional[Union[BaseDb, AsyncBaseDb]] = None,
        memory_manager: Optional[MemoryManager] = None,
        enable_agentic_memory: bool = False,
        enable_user_memories: bool = False,
        add_memories_to_context: Optional[bool] = None,
        enable_session_summaries: bool = False,
        add_session_summary_to_context: Optional[bool] = None,
        session_summary_manager: Optional[SessionSummaryManager] = None,
        add_history_to_context: bool = False,
        num_history_runs: Optional[int] = None,
        num_history_messages: Optional[int] = None,
        max_tool_calls_from_history: Optional[int] = None,
        store_media: bool = True,
        store_tool_messages: bool = True,
        store_history_messages: bool = True,
        knowledge: Optional[Knowledge] = None,
        knowledge_filters: Optional[Dict[str, Any]] = None,
        enable_agentic_knowledge_filters: Optional[bool] = None,
        add_knowledge_to_context: bool = False,
        knowledge_retriever: Optional[Callable[..., Optional[List[Union[Dict, str]]]]] = None,
        references_format: Literal["json", "yaml"] = "json",
        metadata: Optional[Dict[str, Any]] = None,
        tools: Optional[Sequence[Union[Toolkit, Callable, Function, Dict]]] = None,
        tool_call_limit: Optional[int] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        tool_hooks: Optional[List[Callable]] = None,
        pre_hooks: Optional[List[Union[Callable[..., Any], BaseGuardrail]]] = None,
        post_hooks: Optional[List[Union[Callable[..., Any], BaseGuardrail]]] = None,
        reasoning: bool = False,
        reasoning_model: Optional[Union[Model, str]] = None,
        reasoning_agent: Optional[AgentEntity] = None,
        reasoning_min_steps: int = 1,
        reasoning_max_steps: int = 10,
        read_chat_history: bool = False,
        search_knowledge: bool = True,
        update_knowledge: bool = False,
        read_tool_call_history: bool = False,
        send_media_to_model: bool = True,
        system_message: Optional[Union[str, Callable, Message]] = None,
        system_message_role: str = "system",
        build_context: bool = True,
        description: Optional[str] = None,
        instructions: Optional[Union[str, List[str], Callable]] = None,
        expected_output: Optional[str] = None,
        additional_context: Optional[str] = None,
        markdown: bool = False,
        add_name_to_context: bool = False,
        add_datetime_to_context: bool = False,
        add_location_to_context: bool = False,
        timezone_identifier: Optional[str] = None,
        resolve_in_context: bool = True,
        additional_input: Optional[List[Union[str, Dict, BaseModel, Message]]] = None,
        user_message_role: str = "user",
        build_user_context: bool = True,
        retries: int = 0,
        delay_between_retries: int = 1,
        exponential_backoff: bool = False,
        parser_model: Optional[Union[Model, str]] = None,
        parser_model_prompt: Optional[str] = None,
        input_schema: Optional[Type[BaseModel]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
        parse_response: bool = True,
        output_model: Optional[Union[Model, str]] = None,
        output_model_prompt: Optional[str] = None,
        structured_outputs: Optional[bool] = None,
        use_json_mode: bool = False,
        save_response_to_file: Optional[str] = None,
        stream: Optional[bool] = None,
        stream_events: Optional[bool] = None,
        stream_intermediate_steps: Optional[bool] = None,
        store_events: bool = False,
        events_to_skip: Optional[List[RunEvent]] = None,
        role: Optional[str] = None,
        culture_manager: Optional[CultureManager] = None,
        enable_agentic_culture: bool = False,
        update_cultural_knowledge: bool = False,
        add_culture_to_context: Optional[bool] = None,
        debug_mode: bool = False,
        debug_level: Literal[1, 2] = 1,
        telemetry: bool = True,
    ):
        """初始化Agent实体"""
        self.model = model  # type: ignore[assignment]
        self.name = name
        self.id = id
        self.introduction = introduction
        self.user_id = user_id
        
        self.session_id = session_id
        self.session_state = session_state
        self.overwrite_db_session_state = overwrite_db_session_state
        self.enable_agentic_state = enable_agentic_state
        self.cache_session = cache_session
        
        self.search_session_history = search_session_history
        self.num_history_sessions = num_history_sessions
        
        self.dependencies = dependencies
        self.add_dependencies_to_context = add_dependencies_to_context
        self.add_session_state_to_context = add_session_state_to_context
        
        self.db = db
        
        self.memory_manager = memory_manager
        self.enable_agentic_memory = enable_agentic_memory
        self.enable_user_memories = enable_user_memories
        self.add_memories_to_context = add_memories_to_context
        
        self.session_summary_manager = session_summary_manager
        self.enable_session_summaries = enable_session_summaries
        self.add_session_summary_to_context = add_session_summary_to_context
        
        self.add_history_to_context = add_history_to_context
        self.num_history_runs = num_history_runs
        self.num_history_messages = num_history_messages
        if self.num_history_messages is not None and self.num_history_runs is not None:
            log_warning(
                "num_history_messages and num_history_runs cannot be set at the same time. Using num_history_runs."
            )
            self.num_history_messages = None
        if self.num_history_messages is None and self.num_history_runs is None:
            self.num_history_runs = 3
            
        self.max_tool_calls_from_history = max_tool_calls_from_history
        
        self.store_media = store_media
        self.store_tool_messages = store_tool_messages
        self.store_history_messages = store_history_messages
        
        self.knowledge = knowledge
        self.knowledge_filters = knowledge_filters
        self.enable_agentic_knowledge_filters = enable_agentic_knowledge_filters
        self.add_knowledge_to_context = add_knowledge_to_context
        self.knowledge_retriever = knowledge_retriever
        self.references_format = references_format
        
        self.metadata = metadata
        
        self.tools = list(tools) if tools else []
        self.tool_call_limit = tool_call_limit
        self.tool_choice = tool_choice
        self.tool_hooks = tool_hooks
        
        self.pre_hooks = pre_hooks
        self.post_hooks = post_hooks
        
        self.reasoning = reasoning
        self.reasoning_model = reasoning_model  # type: ignore[assignment]
        self.reasoning_agent = reasoning_agent
        self.reasoning_min_steps = reasoning_min_steps
        self.reasoning_max_steps = reasoning_max_steps
        
        self.read_chat_history = read_chat_history
        self.search_knowledge = search_knowledge
        self.update_knowledge = update_knowledge
        self.read_tool_call_history = read_tool_call_history
        self.send_media_to_model = send_media_to_model
        self.system_message = system_message
        self.system_message_role = system_message_role
        self.build_context = build_context
        
        self.description = description
        self.instructions = instructions
        self.expected_output = expected_output
        self.additional_context = additional_context
        self.markdown = markdown
        self.add_name_to_context = add_name_to_context
        self.add_datetime_to_context = add_datetime_to_context
        self.add_location_to_context = add_location_to_context
        self.timezone_identifier = timezone_identifier
        self.resolve_in_context = resolve_in_context
        self.additional_input = additional_input
        self.user_message_role = user_message_role
        self.build_user_context = build_user_context
        
        self.retries = retries
        self.delay_between_retries = delay_between_retries
        self.exponential_backoff = exponential_backoff
        self.parser_model = parser_model  # type: ignore[assignment]
        self.parser_model_prompt = parser_model_prompt
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.parse_response = parse_response
        self.output_model = output_model  # type: ignore[assignment]
        self.output_model_prompt = output_model_prompt
        
        self.structured_outputs = structured_outputs
        
        self.use_json_mode = use_json_mode
        self.save_response_to_file = save_response_to_file
        
        self.stream = stream
        self.stream_events = stream_events or stream_intermediate_steps
        
        self.store_events = store_events
        self.role = role
        self.events_to_skip = events_to_skip
        if self.events_to_skip is None:
            self.events_to_skip = [RunEvent.run_content]
            
        self.culture_manager = culture_manager
        self.enable_agentic_culture = enable_agentic_culture
        self.update_cultural_knowledge = update_cultural_knowledge
        self.add_culture_to_context = add_culture_to_context
        
        self.debug_mode = debug_mode
        if debug_level not in [1, 2]:
            log_warning(f"Invalid debug level: {debug_level}. Setting to 1.")
            debug_level = 1
        self.debug_level = debug_level
        self.telemetry = telemetry
        
        # Internal state
        self._cached_session = None
        self._tool_instructions = None
        self._formatter = None
        self._hooks_normalised = False
        self._mcp_tools_initialized_on_run = []
        self._background_executor = None
    
    # ========================================
    # 基本实体操作
    # ========================================
    
    def set_id(self) -> None:
        """设置Agent ID，如果未提供则根据名称生成"""
        if self.id is None:
            self.id = generate_id_from_name(self.name)
    
    def add_tool(self, tool: Union[Toolkit, Callable, Function, Dict]) -> None:
        """添加单个工具到Agent"""
        if not self.tools:
            self.tools = []
        self.tools.append(tool)
    
    def set_tools(self, tools: Sequence[Union[Toolkit, Callable, Function, Dict]]) -> None:
        """设置Agent的工具列表"""
        self.tools = list(tools) if tools else []
    
    # ========================================
    # 深拷贝功能
    # ========================================
    
    def deep_copy(self, *, update: Optional[Dict[str, Any]] = None) -> AgentEntity:
        """创建Agent的深拷贝
        
        Args:
            update: 可选的字段更新字典
            
        Returns:
            新的Agent实例
        """
        # 提取所有字段值
        fields_for_new_agent: Dict[str, Any] = {}
        
        for f in fields(self):
            field_value = getattr(self, f.name)
            if field_value is not None:
                fields_for_new_agent[f.name] = self._deep_copy_field(f.name, field_value)
        
        # 应用更新
        if update:
            fields_for_new_agent.update(update)
        
        # 创建新实例
        new_agent = self.__class__(**fields_for_new_agent)
        log_debug(f"Created new {self.__class__.__name__}")
        return new_agent
    
    def _deep_copy_field(self, field_name: str, field_value: Any) -> Any:
        """深拷贝单个字段
        
        Args:
            field_name: 字段名称
            field_value: 字段值
            
        Returns:
            拷贝后的字段值
        """
        from copy import copy, deepcopy
        
        # 对于reasoning_agent，使用其deep_copy方法
        if field_name == "reasoning_agent":
            return field_value.deep_copy()
        
        # 对于db, model, reasoning_model，使用深拷贝
        elif field_name in ("db", "model", "reasoning_model"):
            try:
                return deepcopy(field_value)
            except Exception:
                try:
                    return copy(field_value)
                except Exception as e:
                    log_warning(f"Failed to copy field: {field_name} - {e}")
                    return field_value
        
        # 对于复合类型，尝试深拷贝
        elif isinstance(field_value, (list, dict, set)):
            try:
                return deepcopy(field_value)
            except Exception:
                try:
                    return copy(field_value)
                except Exception as e:
                    log_warning(f"Failed to copy field: {field_name} - {e}")
                    return field_value
        
        # 对于Pydantic模型，使用model_copy
        elif isinstance(field_value, BaseModel):
            try:
                return field_value.model_copy(deep=True)
            except Exception:
                try:
                    return field_value.model_copy(deep=False)
                except Exception as e:
                    log_warning(f"Failed to copy field: {field_name} - {e}")
                    return field_value
        
        # 其他类型尝试浅拷贝
        try:
            return copy(field_value)
        except Exception:
            return field_value
    
    # ========================================
    # 属性访问器
    # ========================================
    
    @property
    def cached_session(self) -> Optional[Any]:
        """获取缓存的会话"""
        return self._cached_session
    
    @property
    def should_parse_structured_output(self) -> bool:
        """判断是否需要解析结构化输出"""
        return self.output_schema is not None and self.parse_response and self.parser_model is None
    
    @property
    def background_executor(self) -> Any:
        """懒加载的后台任务执行器
        
        用于处理记忆创建和文化知识更新等后台任务
        仅在首次使用时初始化，跨运行重用
        """
        if self._background_executor is None:
            from concurrent.futures import ThreadPoolExecutor
            
            self._background_executor = ThreadPoolExecutor(
                max_workers=3, 
                thread_name_prefix="gecko-bg"
            )
        return self._background_executor