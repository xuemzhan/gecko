"""
Agent Value Objects - 智能体值对象

职责：
1. 定义Agent相关的轻量级数据结构
2. 提供不可变的值对象
3. 封装领域概念
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agno.models.message import Message


@dataclass
class RunMessages:
    """
    运行消息集合 - 封装一次运行所需的所有消息
    
    Attributes:
        messages: 发送给模型的完整消息列表
        system_message: 系统消息
        user_message: 用户消息
        extra_messages: 额外的消息（few-shot示例等）
    """
    messages: List[Message] = field(default_factory=list)
    system_message: Optional[Message] = None
    user_message: Optional[Message] = None
    extra_messages: Optional[List[Message]] = None
    
    def get_input_messages(self) -> List[Message]:
        """获取输入消息（排除系统消息）
        
        Returns:
            用户输入和历史消息的列表
        """
        if self.system_message is None:
            return self.messages
        
        # 过滤掉系统消息
        return [m for m in self.messages if m != self.system_message]
    
    def add_message(self, message: Message) -> None:
        """添加消息到列表"""
        self.messages.append(message)
    
    def add_messages(self, messages: List[Message]) -> None:
        """批量添加消息"""
        self.messages.extend(messages)


@dataclass
class RunContext:
    """
    运行上下文 - 封装一次运行的完整上下文信息
    
    Attributes:
        run_id: 运行ID
        session_id: 会话ID
        user_id: 用户ID
        session_state: 会话状态（动态数据）
        dependencies: 依赖项（注入的对象/函数）
        knowledge_filters: 知识库过滤器
        metadata: 元数据
    """
    run_id: str
    session_id: str
    user_id: Optional[str] = None
    session_state: Optional[Dict[str, Any]] = None
    dependencies: Optional[Dict[str, Any]] = None
    knowledge_filters: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """初始化默认值"""
        if self.session_state is None:
            self.session_state = {}
        if self.dependencies is None:
            self.dependencies = {}
        if self.metadata is None:
            self.metadata = {}
    
    def update_session_state(self, updates: Dict[str, Any]) -> None:
        """更新会话状态
        
        Args:
            updates: 要更新的键值对
        """
        if self.session_state is None:
            self.session_state = {}
        self.session_state.update(updates)
    
    def get_session_state_value(self, key: str, default: Any = None) -> Any:
        """获取会话状态中的值
        
        Args:
            key: 键名
            default: 默认值
            
        Returns:
            状态值或默认值
        """
        if self.session_state is None:
            return default
        return self.session_state.get(key, default)
    
    def update_metadata(self, updates: Dict[str, Any]) -> None:
        """更新元数据
        
        Args:
            updates: 要更新的键值对
        """
        if self.metadata is None:
            self.metadata = {}
        self.metadata.update(updates)


@dataclass
class AgentExecutionContext:
    """
    Agent执行上下文 - 封装Agent执行过程中的临时状态
    
    用于在执行过程中传递临时数据，不持久化
    
    Attributes:
        reasoning_started: 推理是否已开始
        reasoning_time_taken: 推理耗时（秒）
        parser_model_used: 是否使用了解析器模型
        output_model_used: 是否使用了输出模型
        tools_executed: 已执行的工具数量
        background_tasks: 后台任务列表
    """
    reasoning_started: bool = False
    reasoning_time_taken: float = 0.0
    parser_model_used: bool = False
    output_model_used: bool = False
    tools_executed: int = 0
    background_tasks: List[Any] = field(default_factory=list)
    
    def reset(self) -> None:
        """重置执行上下文"""
        self.reasoning_started = False
        self.reasoning_time_taken = 0.0
        self.parser_model_used = False
        self.output_model_used = False
        self.tools_executed = 0
        self.background_tasks.clear()


@dataclass
class SessionStateUpdate:
    """
    会话状态更新 - 封装会话状态的变更
    
    Attributes:
        key: 状态键
        value: 新值
        operation: 操作类型 (set/delete/clear)
    """
    key: str
    value: Any = None
    operation: str = "set"  # set, delete, clear
    
    def __post_init__(self):
        """验证操作类型"""
        valid_operations = {"set", "delete", "clear"}
        if self.operation not in valid_operations:
            raise ValueError(
                f"Invalid operation: {self.operation}. "
                f"Must be one of {valid_operations}"
            )


@dataclass
class KnowledgeSearchContext:
    """
    知识库搜索上下文 - 封装知识库搜索相关信息
    
    Attributes:
        query: 搜索查询
        filters: 过滤器
        num_documents: 返回文档数量
        min_relevance_score: 最小相关性分数
    """
    query: str
    filters: Optional[Dict[str, Any]] = None
    num_documents: Optional[int] = None
    min_relevance_score: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "query": self.query,
            "filters": self.filters,
            "num_documents": self.num_documents,
            "min_relevance_score": self.min_relevance_score,
        }


@dataclass
class ToolExecutionContext:
    """
    工具执行上下文 - 封装工具执行相关信息
    
    Attributes:
        tool_name: 工具名称
        tool_args: 工具参数
        requires_confirmation: 是否需要确认
        confirmed: 是否已确认
        requires_user_input: 是否需要用户输入
        external_execution_required: 是否需要外部执行
    """
    tool_name: str
    tool_args: Optional[Dict[str, Any]] = None
    requires_confirmation: bool = False
    confirmed: Optional[bool] = None
    requires_user_input: bool = False
    external_execution_required: bool = False
    
    def is_ready_to_execute(self) -> bool:
        """检查工具是否准备好执行
        
        Returns:
            True如果工具可以执行
        """
        # 需要确认但未确认
        if self.requires_confirmation and not self.confirmed:
            return False
        
        # 需要用户输入但未提供
        if self.requires_user_input:
            return False
        
        # 需要外部执行
        if self.external_execution_required:
            return False
        
        return True


@dataclass
class ModelResponseMetadata:
    """
    模型响应元数据 - 封装模型响应的额外信息
    
    Attributes:
        model_id: 模型ID
        model_provider: 模型提供商
        completion_tokens: 完成令牌数
        prompt_tokens: 提示令牌数
        total_tokens: 总令牌数
        time_to_first_token: 首个令牌时间
        total_time: 总时间
    """
    model_id: Optional[str] = None
    model_provider: Optional[str] = None
    completion_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    time_to_first_token: Optional[float] = None
    total_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model_id": self.model_id,
            "model_provider": self.model_provider,
            "completion_tokens": self.completion_tokens,
            "prompt_tokens": self.prompt_tokens,
            "total_tokens": self.total_tokens,
            "time_to_first_token": self.time_to_first_token,
            "total_time": self.total_time,
        }


@dataclass(frozen=True)
class AgentIdentity:
    """
    Agent身份 - 不可变的Agent标识信息
    
    Attributes:
        agent_id: Agent ID
        agent_name: Agent名称
        agent_role: Agent角色
        team_id: 所属团队ID
        workflow_id: 所属工作流ID
    """
    agent_id: str
    agent_name: Optional[str] = None
    agent_role: Optional[str] = None
    team_id: Optional[str] = None
    workflow_id: Optional[str] = None
    
    def is_team_member(self) -> bool:
        """检查是否为团队成员"""
        return self.team_id is not None
    
    def is_workflow_member(self) -> bool:
        """检查是否为工作流成员"""
        return self.workflow_id is not None
    
    def is_standalone(self) -> bool:
        """检查是否为独立Agent"""
        return not self.is_team_member() and not self.is_workflow_member()


@dataclass
class HookExecutionResult:
    """
    钩子执行结果 - 封装钩子函数的执行结果
    
    Attributes:
        hook_name: 钩子名称
        success: 是否成功
        error: 错误信息（如果失败）
        execution_time: 执行时间
        modified_data: 修改后的数据
    """
    hook_name: str
    success: bool = True
    error: Optional[str] = None
    execution_time: float = 0.0
    modified_data: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "hook_name": self.hook_name,
            "success": self.success,
            "error": self.error,
            "execution_time": self.execution_time,
            "has_modified_data": self.modified_data is not None,
        }


@dataclass
class ReasoningState:
    """
    推理状态 - 封装推理过程的状态
    
    Attributes:
        step_count: 当前步骤数
        reasoning_started: 是否已开始
        reasoning_completed: 是否已完成
        total_time: 总耗时
        next_action: 下一步动作
    """
    step_count: int = 0
    reasoning_started: bool = False
    reasoning_completed: bool = False
    total_time: float = 0.0
    next_action: str = "continue"
    
    def increment_step(self) -> None:
        """增加步骤计数"""
        self.step_count += 1
    
    def mark_started(self) -> None:
        """标记为已开始"""
        self.reasoning_started = True
    
    def mark_completed(self) -> None:
        """标记为已完成"""
        self.reasoning_completed = True
    
    def should_continue(self, max_steps: int) -> bool:
        """检查是否应该继续推理
        
        Args:
            max_steps: 最大步骤数
            
        Returns:
            True如果应该继续
        """
        return (
            self.next_action == "continue" 
            and self.step_count < max_steps 
            and not self.reasoning_completed
        )


# 类型别名
MessageList = List[Message]
ToolList = List[Dict[str, Any]]
FilterDict = Dict[str, Any]
StateDict = Dict[str, Any]
MetadataDict = Dict[str, Any]