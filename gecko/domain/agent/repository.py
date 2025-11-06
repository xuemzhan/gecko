"""
Agent Repository - 智能体仓储接口

职责：
1. 定义Agent会话持久化的接口
2. 提供数据访问的抽象层
3. 解耦领域层和基础设施层

注意：这是接口定义，具体实现在infrastructure层
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Union, cast

from gecko.domain.agent.value_objects import RunContext
from gecko.infrastructure.storage.adapter import AsyncBaseDb, BaseDb, SessionType
from gecko.utils.logging import log_debug, log_warning

# 导入会话相关类型（这些应该在domain层定义）
try:
    from agno.session import AgentSession, TeamSession, WorkflowSession
    from agno.run.agent import RunOutput
except ImportError:
    # 临时兼容，后续这些类型应该移到gecko.domain下
    AgentSession = Any  # type: ignore
    TeamSession = Any  # type: ignore
    WorkflowSession = Any  # type: ignore
    RunOutput = Any  # type: ignore


class IAgentRepository(ABC):
    """
    Agent仓储接口
    
    定义Agent数据持久化的所有操作
    """
    
    @abstractmethod
    def read_session(
        self, 
        session_id: str, 
        session_type: SessionType = SessionType.AGENT
    ) -> Optional[Union[AgentSession, TeamSession, WorkflowSession]]:
        """读取会话"""
        pass
    
    @abstractmethod
    async def aread_session(
        self, 
        session_id: str, 
        session_type: SessionType = SessionType.AGENT
    ) -> Optional[Union[AgentSession, TeamSession, WorkflowSession]]:
        """异步读取会话"""
        pass
    
    @abstractmethod
    def upsert_session(self, session: AgentSession) -> Optional[AgentSession]:
        """插入或更新会话"""
        pass
    
    @abstractmethod
    async def aupsert_session(self, session: AgentSession) -> Optional[AgentSession]:
        """异步插入或更新会话"""
        pass
    
    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """删除会话"""
        pass
    
    @abstractmethod
    async def adelete_session(self, session_id: str) -> None:
        """异步删除会话"""
        pass
    
    @abstractmethod
    def get_run_output(self, run_id: str, session_id: str) -> Optional[RunOutput]:
        """获取运行输出"""
        pass
    
    @abstractmethod
    async def aget_run_output(self, run_id: str, session_id: str) -> Optional[RunOutput]:
        """异步获取运行输出"""
        pass


class AgentRepository(IAgentRepository):
    """
    Agent仓储实现
    
    封装了Agent会话的所有数据访问操作
    依赖注入数据库实例，支持同步和异步操作
    """
    
    def __init__(
        self,
        db: Optional[Union[BaseDb, AsyncBaseDb]] = None,
        team_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        cache_enabled: bool = False,
    ):
        """
        初始化仓储
        
        Args:
            db: 数据库实例（同步或异步）
            team_id: 团队ID（如果是团队成员）
            workflow_id: 工作流ID（如果是工作流成员）
            cache_enabled: 是否启用缓存
        """
        self.db = db
        self.team_id = team_id
        self.workflow_id = workflow_id
        self.cache_enabled = cache_enabled
        self._cached_session: Optional[AgentSession] = None
    
    def _has_async_db(self) -> bool:
        """检查是否为异步数据库"""
        return self.db is not None and isinstance(self.db, AsyncBaseDb)
    
    # ========================================
    # 会话读取
    # ========================================
    
    def read_session(
        self, 
        session_id: str, 
        session_type: SessionType = SessionType.AGENT
    ) -> Optional[Union[AgentSession, TeamSession, WorkflowSession]]:
        """从数据库读取会话
        
        Args:
            session_id: 会话ID
            session_type: 会话类型
            
        Returns:
            会话对象，如果不存在返回None
        """
        try:
            if not self.db:
                raise ValueError("Database not initialized")
            return self.db.get_session(session_id=session_id, session_type=session_type)  # type: ignore
        except Exception as e:
            log_warning(f"Error reading session from db: {e}")
            return None
    
    async def aread_session(
        self, 
        session_id: str, 
        session_type: SessionType = SessionType.AGENT
    ) -> Optional[Union[AgentSession, TeamSession, WorkflowSession]]:
        """异步从数据库读取会话
        
        Args:
            session_id: 会话ID
            session_type: 会话类型
            
        Returns:
            会话对象，如果不存在返回None
        """
        try:
            if not self.db:
                raise ValueError("Database not initialized")
            return await self.db.get_session(session_id=session_id, session_type=session_type)  # type: ignore
        except Exception as e:
            log_warning(f"Error reading session from db: {e}")
            return None
    
    # ========================================
    # 会话写入
    # ========================================
    
    def upsert_session(self, session: AgentSession) -> Optional[AgentSession]:
        """插入或更新会话到数据库
        
        Args:
            session: 会话对象
            
        Returns:
            更新后的会话对象
        """
        try:
            if not self.db:
                raise ValueError("Database not initialized")
            return self.db.upsert_session(session=session)  # type: ignore
        except Exception as e:
            log_warning(f"Error upserting session into db: {e}")
            return None
    
    async def aupsert_session(self, session: AgentSession) -> Optional[AgentSession]:
        """异步插入或更新会话到数据库
        
        Args:
            session: 会话对象
            
        Returns:
            更新后的会话对象
        """
        try:
            if not self.db:
                raise ValueError("Database not initialized")
            return await self.db.upsert_session(session=session)  # type: ignore
        except Exception as e:
            log_warning(f"Error upserting session into db: {e}")
            return None
    
    # ========================================
    # 会话删除
    # ========================================
    
    def delete_session(self, session_id: str) -> None:
        """删除会话
        
        Args:
            session_id: 会话ID
        """
        if self.db is None:
            return
        
        self.db.delete_session(session_id=session_id)
        log_debug(f"Deleted session: {session_id}")
    
    async def adelete_session(self, session_id: str) -> None:
        """异步删除会话
        
        Args:
            session_id: 会话ID
        """
        if self.db is None:
            return
        
        await self.db.delete_session(session_id=session_id)  # type: ignore
        log_debug(f"Deleted session: {session_id}")
    
    # ========================================
    # 会话创建或读取
    # ========================================
    
    def read_or_create_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_data: Optional[dict] = None,
        session_state: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> AgentSession:
        """读取或创建会话
        
        如果启用了缓存且缓存中有会话，直接返回
        否则尝试从数据库读取，不存在则创建新会话
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            agent_id: Agent ID
            agent_data: Agent数据
            session_state: 会话状态
            metadata: 元数据
            
        Returns:
            会话对象
        """
        # 返回缓存的会话
        if self.cache_enabled and self._cached_session is not None:
            if self._cached_session.session_id == session_id:
                return self._cached_session
        
        # 尝试从数据库加载
        agent_session = None
        if self.db is not None and self.team_id is None and self.workflow_id is None:
            log_debug(f"Reading AgentSession: {session_id}")
            agent_session = cast(AgentSession, self.read_session(session_id=session_id))
        
        # 创建新会话
        if agent_session is None:
            from time import time
            from copy import deepcopy
            
            log_debug(f"Creating new AgentSession: {session_id}")
            
            session_data = {}
            if session_state is not None:
                session_data["session_state"] = deepcopy(session_state)
            
            agent_session = AgentSession(
                session_id=session_id,
                agent_id=agent_id,
                user_id=user_id,
                agent_data=agent_data or {},
                session_data=session_data,
                metadata=metadata,
                created_at=int(time()),
            )
        
        # 缓存会话
        if self.cache_enabled:
            self._cached_session = agent_session
        
        return agent_session
    
    async def aread_or_create_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        agent_data: Optional[dict] = None,
        session_state: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> AgentSession:
        """异步读取或创建会话
        
        Args:
            session_id: 会话ID
            user_id: 用户ID
            agent_id: Agent ID
            agent_data: Agent数据
            session_state: 会话状态
            metadata: 元数据
            
        Returns:
            会话对象
        """
        # 返回缓存的会话
        if self.cache_enabled and self._cached_session is not None:
            if self._cached_session.session_id == session_id:
                return self._cached_session
        
        # 尝试从数据库加载
        agent_session = None
        if self.db is not None and self.team_id is None and self.workflow_id is None:
            log_debug(f"Reading AgentSession: {session_id}")
            if self._has_async_db():
                agent_session = cast(AgentSession, await self.aread_session(session_id=session_id))
            else:
                agent_session = cast(AgentSession, self.read_session(session_id=session_id))
        
        # 创建新会话
        if agent_session is None:
            from time import time
            from copy import deepcopy
            
            log_debug(f"Creating new AgentSession: {session_id}")
            
            session_data = {}
            if session_state is not None:
                session_data["session_state"] = deepcopy(session_state)
            
            agent_session = AgentSession(
                session_id=session_id,
                agent_id=agent_id,
                user_id=user_id,
                agent_data=agent_data or {},
                session_data=session_data,
                metadata=metadata,
                created_at=int(time()),
            )
        
        # 缓存会话
        if self.cache_enabled:
            self._cached_session = agent_session
        
        return agent_session
    
    # ========================================
    # 会话保存
    # ========================================
    
    def save_session(self, session: AgentSession) -> None:
        """保存会话到数据库
        
        清理临时状态并持久化
        
        Args:
            session: 会话对象
        """
        # 如果Agent是团队/工作流成员，不直接保存
        if self.team_id is not None or self.workflow_id is not None:
            return
        
        if self.db is not None and session.session_data is not None:
            # 清理临时状态
            if "session_state" in session.session_data:
                session.session_data["session_state"].pop("current_session_id", None)
                session.session_data["session_state"].pop("current_user_id", None)
                session.session_data["session_state"].pop("current_run_id", None)
            
            self.upsert_session(session=session)
            log_debug(f"Saved AgentSession: {session.session_id}")
    
    async def asave_session(self, session: AgentSession) -> None:
        """异步保存会话到数据库
        
        Args:
            session: 会话对象
        """
        # 如果Agent是团队/工作流成员，不直接保存
        if self.team_id is not None or self.workflow_id is not None:
            return
        
        if self.db is not None and session.session_data is not None:
            # 清理临时状态
            if "session_state" in session.session_data:
                session.session_data["session_state"].pop("current_session_id", None)
                session.session_data["session_state"].pop("current_user_id", None)
                session.session_data["session_state"].pop("current_run_id", None)
            
            if self._has_async_db():
                await self.aupsert_session(session=session)
            else:
                self.upsert_session(session=session)
            
            log_debug(f"Saved AgentSession: {session.session_id}")
    
    # ========================================
    # 运行输出查询
    # ========================================
    
    def get_run_output(self, run_id: str, session_id: str) -> Optional[RunOutput]:
        """获取运行输出
        
        Args:
            run_id: 运行ID
            session_id: 会话ID
            
        Returns:
            运行输出对象
        """
        session = self.read_session(session_id=session_id)
        if session is None or session.runs is None:
            return None
        
        for run in session.runs:
            if run.run_id == run_id:
                return run
        
        return None
    
    async def aget_run_output(self, run_id: str, session_id: str) -> Optional[RunOutput]:
        """异步获取运行输出
        
        Args:
            run_id: 运行ID
            session_id: 会话ID
            
        Returns:
            运行输出对象
        """
        session = await self.aread_session(session_id=session_id)
        if session is None or session.runs is None:
            return None
        
        for run in session.runs:
            if run.run_id == run_id:
                return run
        
        return None
    
    def get_last_run_output(self, session_id: str) -> Optional[RunOutput]:
        """获取最后一次运行输出
        
        Args:
            session_id: 会话ID
            
        Returns:
            最后一次运行输出
        """
        session = self.read_session(session_id=session_id)
        if session is None or session.runs is None or len(session.runs) == 0:
            return None
        
        return session.runs[-1]
    
    async def aget_last_run_output(self, session_id: str) -> Optional[RunOutput]:
        """异步获取最后一次运行输出
        
        Args:
            session_id: 会话ID
            
        Returns:
            最后一次运行输出
        """
        session = await self.aread_session(session_id=session_id)
        if session is None or session.runs is None or len(session.runs) == 0:
            return None
        
        return session.runs[-1]
    
    # ========================================
    # 会话查询
    # ========================================
    
    def get_session(self, session_id: str) -> Optional[AgentSession]:
        """获取会话（支持缓存）
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话对象
        """
        # 检查缓存
        if self.cache_enabled and self._cached_session is not None:
            if self._cached_session.session_id == session_id:
                return self._cached_session
        
        if self._has_async_db():
            raise ValueError("Use aget_session for async database")
        
        # 从数据库加载
        if self.db is not None:
            loaded_session = None
            
            # 独立Agent
            if self.team_id is None and self.workflow_id is None:
                loaded_session = cast(
                    AgentSession,
                    self.read_session(session_id=session_id, session_type=SessionType.AGENT),
                )
            
            # 团队成员
            if loaded_session is None and self.team_id is not None:
                loaded_session = cast(
                    TeamSession,
                    self.read_session(session_id=session_id, session_type=SessionType.TEAM),
                )
            
            # 工作流成员
            if loaded_session is None and self.workflow_id is not None:
                loaded_session = cast(
                    WorkflowSession,
                    self.read_session(session_id=session_id, session_type=SessionType.WORKFLOW),
                )
            
            # 缓存会话
            if loaded_session is not None and self.cache_enabled:
                self._cached_session = loaded_session
            
            return loaded_session
        
        log_debug(f"Session {session_id} not found in db")
        return None
    
    async def aget_session(self, session_id: str) -> Optional[AgentSession]:
        """异步获取会话（支持缓存）
        
        Args:
            session_id: 会话ID
            
        Returns:
            会话对象
        """
        # 检查缓存
        if self.cache_enabled and self._cached_session is not None:
            if self._cached_session.session_id == session_id:
                return self._cached_session
        
        # 从数据库加载
        if self.db is not None:
            loaded_session = None
            
            # 独立Agent
            if self.team_id is None and self.workflow_id is None:
                loaded_session = cast(
                    AgentSession,
                    await self.aread_session(session_id=session_id, session_type=SessionType.AGENT),
                )
            
            # 团队成员
            if loaded_session is None and self.team_id is not None:
                loaded_session = cast(
                    TeamSession,
                    await self.aread_session(session_id=session_id, session_type=SessionType.TEAM),
                )
            
            # 工作流成员
            if loaded_session is None and self.workflow_id is not None:
                loaded_session = cast(
                    WorkflowSession,
                    await self.aread_session(session_id=session_id, session_type=SessionType.WORKFLOW),
                )
            
            # 缓存会话
            if loaded_session is not None and self.cache_enabled:
                self._cached_session = loaded_session
            
            return loaded_session
        
        log_debug(f"Session {session_id} not found in db")
        return None
    
    # ========================================
    # 缓存管理
    # ========================================
    
    def clear_cache(self) -> None:
        """清除缓存"""
        self._cached_session = None
    
    @property
    def cached_session(self) -> Optional[AgentSession]:
        """获取缓存的会话"""
        return self._cached_session