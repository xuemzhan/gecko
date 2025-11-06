"""
Memory Manager - 记忆管理器

职责：
1. 用户记忆的创建、读取、更新
2. 文化知识的管理
3. 会话摘要的生成和管理
4. 会话元数据和指标管理
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

from gecko.utils.logging import log_debug, log_error, log_info, log_warning

# 临时导入，后续应移到gecko下
try:
    from agno.culture.manager import CultureManager
    from agno.db.base import AsyncBaseDb, BaseDb
    from agno.db.schemas.culture import CulturalKnowledge
    from agno.memory import MemoryManager as AgnoMemoryManager
    from agno.models.message import Message
    from agno.models.metrics import Metrics
    from agno.session import AgentSession, SessionSummaryManager
    from agno.session.summary import SessionSummary
    from agno.db.base import UserMemory
except ImportError:
    # 临时处理
    CultureManager = Any  # type: ignore
    AgnoMemoryManager = Any  # type: ignore
    SessionSummaryManager = Any  # type: ignore
    AgentSession = Any  # type: ignore
    SessionSummary = Any  # type: ignore
    CulturalKnowledge = Any  # type: ignore
    UserMemory = Any  # type: ignore
    BaseDb = Any  # type: ignore
    AsyncBaseDb = Any  # type: ignore
    Message = Any  # type: ignore
    Metrics = Any  # type: ignore


class AgentMemoryManager:
    """
    Agent记忆管理器
    
    整合用户记忆、文化知识和会话摘要的管理
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        db: Optional[Any] = None,
        model: Optional[Any] = None,
        enable_user_memories: bool = False,
        enable_agentic_memory: bool = False,
        enable_cultural_knowledge: bool = False,
        enable_session_summaries: bool = False,
    ):
        """
        初始化记忆管理器
        
        Args:
            agent_id: Agent ID
            db: 数据库实例
            model: 模型实例
            enable_user_memories: 启用用户记忆
            enable_agentic_memory: 启用代理记忆
            enable_cultural_knowledge: 启用文化知识
            enable_session_summaries: 启用会话摘要
        """
        self.agent_id = agent_id
        self.db = db
        self.model = model
        
        # 功能开关
        self.enable_user_memories = enable_user_memories
        self.enable_agentic_memory = enable_agentic_memory
        self.enable_cultural_knowledge = enable_cultural_knowledge
        self.enable_session_summaries = enable_session_summaries
        
        # 子管理器
        self.memory_manager: Optional[AgnoMemoryManager] = None
        self.culture_manager: Optional[CultureManager] = None
        self.session_summary_manager: Optional[SessionSummaryManager] = None
    
    def _has_async_db(self) -> bool:
        """检查是否为异步数据库"""
        return self.db is not None and isinstance(self.db, AsyncBaseDb)
    
    # ========================================
    # 初始化子管理器
    # ========================================
    
    def initialize_memory_manager(self) -> None:
        """初始化用户记忆管理器"""
        if self.db is None:
            log_warning("Database not provided. Memories will not be stored.")
        
        if self.memory_manager is None:
            self.memory_manager = AgnoMemoryManager(model=self.model, db=self.db)
        else:
            if self.memory_manager.model is None:
                self.memory_manager.model = self.model
            if self.memory_manager.db is None:
                self.memory_manager.db = self.db
    
    def initialize_culture_manager(self) -> None:
        """初始化文化知识管理器"""
        if self.db is None:
            log_warning("Database not provided. Cultural knowledge will not be stored.")
        
        if self.culture_manager is None:
            self.culture_manager = CultureManager(model=self.model, db=self.db)
        else:
            if self.culture_manager.model is None:
                self.culture_manager.model = self.model
            if self.culture_manager.db is None:
                self.culture_manager.db = self.db
    
    def initialize_session_summary_manager(self) -> None:
        """初始化会话摘要管理器"""
        if self.session_summary_manager is None:
            self.session_summary_manager = SessionSummaryManager(model=self.model)
        else:
            if self.session_summary_manager.model is None:
                self.session_summary_manager.model = self.model
    
    # ========================================
    # 用户记忆管理
    # ========================================
    
    def create_user_memories(
        self,
        run_messages: Any,
        user_id: Optional[str] = None,
    ) -> None:
        """创建用户记忆（同步）
        
        从运行消息中提取并创建用户记忆
        
        Args:
            run_messages: 运行消息对象
            user_id: 用户ID
        """
        if self.memory_manager is None:
            return
        
        user_message_str = (
            run_messages.user_message.get_content_string()
            if run_messages.user_message is not None
            else None
        )
        
        # 从用户消息创建记忆
        if user_message_str is not None and user_message_str.strip() != "":
            log_debug("Managing user memories")
            self.memory_manager.create_user_memories(  # type: ignore
                message=user_message_str,
                user_id=user_id,
                agent_id=self.agent_id,
            )
        
        # 从额外消息创建记忆
        if run_messages.extra_messages is not None and len(run_messages.extra_messages) > 0:
            parsed_messages = self._parse_extra_messages(run_messages.extra_messages)
            
            # 过滤空内容的消息
            non_empty_messages = [
                msg for msg in parsed_messages
                if msg.content and (not isinstance(msg.content, str) or msg.content.strip() != "")
            ]
            
            if len(non_empty_messages) > 0:
                self.memory_manager.create_user_memories(  # type: ignore
                    messages=non_empty_messages,
                    user_id=user_id,
                    agent_id=self.agent_id,
                )
            else:
                log_warning("Unable to add messages to memory")
    
    async def acreate_user_memories(
        self,
        run_messages: Any,
        user_id: Optional[str] = None,
    ) -> None:
        """创建用户记忆（异步）
        
        Args:
            run_messages: 运行消息对象
            user_id: 用户ID
        """
        if self.memory_manager is None:
            return
        
        user_message_str = (
            run_messages.user_message.get_content_string()
            if run_messages.user_message is not None
            else None
        )
        
        # 从用户消息创建记忆
        if user_message_str is not None and user_message_str.strip() != "":
            log_debug("Managing user memories")
            await self.memory_manager.acreate_user_memories(  # type: ignore
                message=user_message_str,
                user_id=user_id,
                agent_id=self.agent_id,
            )
        
        # 从额外消息创建记忆
        if run_messages.extra_messages is not None and len(run_messages.extra_messages) > 0:
            parsed_messages = self._parse_extra_messages(run_messages.extra_messages)
            
            # 过滤空内容的消息
            non_empty_messages = [
                msg for msg in parsed_messages
                if msg.content and (not isinstance(msg.content, str) or msg.content.strip() != "")
            ]
            
            if len(non_empty_messages) > 0:
                await self.memory_manager.acreate_user_memories(  # type: ignore
                    messages=non_empty_messages,
                    user_id=user_id,
                    agent_id=self.agent_id,
                )
            else:
                log_warning("Unable to add messages to memory")
    
    def _parse_extra_messages(self, extra_messages: List[Any]) -> List[Message]:
        """解析额外消息
        
        Args:
            extra_messages: 额外消息列表
            
        Returns:
            解析后的消息列表
        """
        parsed_messages = []
        
        for _im in extra_messages:
            if isinstance(_im, Message):
                parsed_messages.append(_im)
            elif isinstance(_im, dict):
                try:
                    parsed_messages.append(Message(**_im))
                except Exception as e:
                    log_warning(f"Failed to validate message during memory update: {e}")
            else:
                log_warning(f"Unsupported message type: {type(_im)}")
                continue
        
        return parsed_messages
    
    def get_user_memories(
        self,
        user_id: Optional[str] = None,
    ) -> Optional[List[UserMemory]]:
        """获取用户记忆
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户记忆列表
        """
        if self.memory_manager is None:
            return None
        
        if user_id is None:
            user_id = "default"
        
        return self.memory_manager.get_user_memories(user_id=user_id)
    
    async def aget_user_memories(
        self,
        user_id: Optional[str] = None,
    ) -> Optional[List[UserMemory]]:
        """异步获取用户记忆
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户记忆列表
        """
        if self.memory_manager is None:
            return None
        
        if user_id is None:
            user_id = "default"
        
        return await self.memory_manager.aget_user_memories(user_id=user_id)
    
    # ========================================
    # 文化知识管理
    # ========================================
    
    def create_cultural_knowledge(
        self,
        run_messages: Any,
    ) -> None:
        """创建文化知识（同步）
        
        Args:
            run_messages: 运行消息对象
        """
        if run_messages.user_message is None or self.culture_manager is None:
            return
        
        log_debug("Creating cultural knowledge.")
        self.culture_manager.create_cultural_knowledge(
            message=run_messages.user_message.get_content_string()
        )
    
    async def acreate_cultural_knowledge(
        self,
        run_messages: Any,
    ) -> None:
        """创建文化知识（异步）
        
        Args:
            run_messages: 运行消息对象
        """
        if run_messages.user_message is None or self.culture_manager is None:
            return
        
        log_debug("Creating cultural knowledge.")
        await self.culture_manager.acreate_cultural_knowledge(
            message=run_messages.user_message.get_content_string()
        )
    
    def get_cultural_knowledge(self) -> Optional[List[CulturalKnowledge]]:
        """获取文化知识
        
        Returns:
            文化知识列表
        """
        if self.culture_manager is None:
            return None
        
        return self.culture_manager.get_all_knowledge()
    
    async def aget_cultural_knowledge(self) -> Optional[List[CulturalKnowledge]]:
        """异步获取文化知识
        
        Returns:
            文化知识列表
        """
        if self.culture_manager is None:
            return None
        
        return await self.culture_manager.aget_all_knowledge()
    
    # ========================================
    # 会话摘要管理
    # ========================================
    
    def create_session_summary(
        self,
        session: AgentSession,
    ) -> None:
        """创建会话摘要（同步）
        
        Args:
            session: Agent会话对象
        """
        if self.session_summary_manager is None:
            return
        
        try:
            self.session_summary_manager.create_session_summary(session=session)
        except Exception as e:
            log_warning(f"Error in session summary creation: {str(e)}")
    
    async def acreate_session_summary(
        self,
        session: AgentSession,
    ) -> None:
        """创建会话摘要（异步）
        
        Args:
            session: Agent会话对象
        """
        if self.session_summary_manager is None:
            return
        
        try:
            await self.session_summary_manager.acreate_session_summary(session=session)
        except Exception as e:
            log_warning(f"Error in session summary creation: {str(e)}")
    
    def get_session_summary(
        self,
        session: AgentSession,
    ) -> Optional[SessionSummary]:
        """获取会话摘要
        
        Args:
            session: Agent会话对象
            
        Returns:
            会话摘要对象
        """
        return session.get_session_summary()
    
    def generate_session_name(
        self,
        session: AgentSession,
        model: Optional[Any] = None,
    ) -> str:
        """生成会话名称
        
        使用前6条消息生成会话名称
        
        Args:
            session: Agent会话对象
            model: 模型实例（用于生成名称）
            
        Returns:
            生成的会话名称
        """
        if model is None:
            model = self.model
        
        if model is None:
            raise Exception("Model not set")
        
        # 构建生成名称的提示
        gen_session_name_prompt = "Conversation\n"
        
        messages_for_generating_session_name = session.get_messages_for_session()
        
        for message in messages_for_generating_session_name:
            gen_session_name_prompt += f"{message.role.upper()}: {message.content}\n"
        
        gen_session_name_prompt += "\n\nConversation Name: "
        
        # 创建系统和用户消息
        system_message = Message(
            role="system",
            content=(
                "Please provide a suitable name for this conversation in maximum 5 words. "
                "Remember, do not exceed 5 words."
            ),
        )
        user_message = Message(role="user", content=gen_session_name_prompt)
        generate_name_messages = [system_message, user_message]
        
        # 生成名称
        generated_name = model.response(messages=generate_name_messages)
        content = generated_name.content
        
        if content is None:
            log_error("Generated name is None. Trying again.")
            return self.generate_session_name(session=session, model=model)
        
        if len(content.split()) > 5:
            log_error("Generated name is too long. It should be less than 5 words. Trying again.")
            return self.generate_session_name(session=session, model=model)
        
        return content.replace('"', "").strip()


class SessionMetricsManager:
    """
    会话指标管理器
    
    负责计算和管理会话相关的指标
    """
    
    @staticmethod
    def calculate_run_metrics(
        messages: List[Message],
        current_run_metrics: Optional[Metrics] = None,
        assistant_role: str = "assistant",
    ) -> Metrics:
        """计算运行指标
        
        汇总消息中的指标到单个Metrics对象
        
        Args:
            messages: 消息列表
            current_run_metrics: 当前运行指标
            assistant_role: 助手角色名
            
        Returns:
            计算后的指标对象
        """
        metrics = current_run_metrics or Metrics()
        
        for m in messages:
            if m.role == assistant_role and m.metrics is not None and m.from_history is False:
                metrics += m.metrics
        
        # 如果运行指标已初始化，保持时间相关指标
        if current_run_metrics is not None:
            metrics.timer = current_run_metrics.timer
            metrics.duration = current_run_metrics.duration
            metrics.time_to_first_token = current_run_metrics.time_to_first_token
        
        return metrics
    
    @staticmethod
    def get_session_metrics(
        session: AgentSession,
    ) -> Metrics:
        """从会话数据中获取会话指标
        
        Args:
            session: Agent会话对象
            
        Returns:
            会话指标对象
        """
        if session.session_data is not None and "session_metrics" in session.session_data:
            session_metrics_from_db = session.session_data.get("session_metrics")
            
            if session_metrics_from_db is not None:
                if isinstance(session_metrics_from_db, dict):
                    return Metrics(**session_metrics_from_db)
                elif isinstance(session_metrics_from_db, Metrics):
                    return session_metrics_from_db
        
        return Metrics()
    
    @staticmethod
    def update_session_metrics(
        session: AgentSession,
        run_metrics: Metrics,
    ) -> None:
        """更新会话指标
        
        将运行指标累加到会话指标
        
        Args:
            session: Agent会话对象
            run_metrics: 运行指标
        """
        session_metrics = SessionMetricsManager.get_session_metrics(session=session)
        
        # 累加当前运行的指标
        session_metrics += run_metrics
        
        # 清除time_to_first_token（这是运行级别的指标）
        session_metrics.time_to_first_token = None
        
        # 更新会话数据
        if session.session_data is not None:
            session.session_data["session_metrics"] = session_metrics
        else:
            session.session_data = {"session_metrics": session_metrics}


class SessionNameManager:
    """
    会话名称管理器
    
    管理会话的命名
    """
    
    @staticmethod
    def set_session_name(
        session: AgentSession,
        session_name: Optional[str] = None,
        autogenerate: bool = False,
        memory_manager: Optional[AgentMemoryManager] = None,
    ) -> AgentSession:
        """设置会话名称
        
        Args:
            session: Agent会话对象
            session_name: 会话名称
            autogenerate: 是否自动生成
            memory_manager: 记忆管理器（用于生成名称）
            
        Returns:
            更新后的会话对象
        """
        if autogenerate and memory_manager is not None:
            session_name = memory_manager.generate_session_name(session=session)
        
        if session_name:
            session.session_name = session_name
            log_info(f"Session name set to: {session_name}")
        
        return session
    
    @staticmethod
    def get_session_name(session: AgentSession) -> str:
        """获取会话名称
        
        Args:
            session: Agent会话对象
            
        Returns:
            会话名称
        """
        return session.session_name or "Unnamed Session"


class AgentRenameManager:
    """
    Agent重命名管理器
    
    管理Agent的重命名操作
    """
    
    @staticmethod
    def rename_agent(
        session: AgentSession,
        new_name: str,
    ) -> None:
        """重命名Agent
        
        更新会话中的Agent名称
        
        Args:
            session: Agent会话对象
            new_name: 新名称
        """
        if session.agent_data is not None:
            session.agent_data["name"] = new_name
        else:
            session.agent_data = {"name": new_name}
        
        log_info(f"Agent renamed to: {new_name}")


# ========================================
# 便捷函数（兼容旧代码）
# ========================================

def create_memory_manager(
    model: Optional[Any] = None,
    db: Optional[Any] = None,
    agent_id: Optional[str] = None,
) -> AgnoMemoryManager:
    """创建记忆管理器
    
    Args:
        model: 模型实例
        db: 数据库实例
        agent_id: Agent ID
        
    Returns:
        记忆管理器实例
    """
    return AgnoMemoryManager(model=model, db=db)


def create_culture_manager(
    model: Optional[Any] = None,
    db: Optional[Any] = None,
) -> CultureManager:
    """创建文化知识管理器
    
    Args:
        model: 模型实例
        db: 数据库实例
        
    Returns:
        文化知识管理器实例
    """
    return CultureManager(model=model, db=db)


def create_session_summary_manager(
    model: Optional[Any] = None,
) -> SessionSummaryManager:
    """创建会话摘要管理器
    
    Args:
        model: 模型实例
        
    Returns:
        会话摘要管理器实例
    """
    return SessionSummaryManager(model=model)