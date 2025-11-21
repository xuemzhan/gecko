# gecko/core/session.py
"""
会话管理系统

核心功能：
1. 会话状态管理（内存 + 持久化）
2. 会话元数据（创建时间、更新时间、访问次数等）
3. 会话生命周期（过期、自动清理）
4. 并发安全（异步锁）

优化日志：
1. 修复同步方法调用异步保存的并发问题
2. 引入延迟保存机制 (Debounce/Dirty Flag)
3. 增强 SessionManager 的清理逻辑
4. 完善类型注解和错误处理
5. 修复 SessionManager 缺失的方法
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from gecko.core.events import BaseEvent, EventBus
from gecko.core.logging import get_logger
from gecko.plugins.storage.interfaces import SessionInterface

logger = get_logger(__name__)


# ===== 会话元数据 =====

class SessionMetadata(BaseModel):
    """会话元数据"""
    session_id: str = Field(..., description="会话 ID")
    created_at: float = Field(default_factory=time.time, description="创建时间")
    updated_at: float = Field(default_factory=time.time, description="更新时间")
    accessed_at: float = Field(default_factory=time.time, description="访问时间")
    access_count: int = Field(default=0, description="访问次数")
    ttl: Optional[int] = Field(default=None, description="生存时间（秒）")
    tags: Set[str] = Field(default_factory=set, description="标签")
    custom: Dict[str, Any] = Field(default_factory=dict, description="自定义数据")
    
    def is_expired(self) -> bool:
        """检查会话是否过期"""
        if self.ttl is None:
            return False
        age = time.time() - self.created_at
        return age > self.ttl
    
    def time_to_expire(self) -> Optional[float]:
        """获取距离过期的剩余时间（秒）"""
        if self.ttl is None:
            return None
        age = time.time() - self.created_at
        return max(0.0, self.ttl - age)
    
    def touch(self):
        """更新访问时间和计数"""
        self.accessed_at = time.time()
        self.access_count += 1


# ===== 会话事件 =====

class SessionEvent(BaseEvent):
    """会话相关事件"""
    pass


# ===== 会话对象 =====

class Session:
    """
    会话对象
    
    管理内存中的会话状态，并提供可选的持久化支持。
    """
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None,
        storage: Optional[SessionInterface] = None,
        ttl: Optional[int] = None,
        event_bus: Optional[EventBus] = None,
        auto_save: bool = False, 
    ):
        self.session_id = session_id or self._generate_id()
        self.state: Dict[str, Any] = state or {}
        self.storage = storage
        self.event_bus = event_bus or EventBus()
        self.auto_save = auto_save
        
        # 元数据
        self.metadata = SessionMetadata(
            session_id=self.session_id,
            ttl=ttl
        )
        
        # 并发锁
        self._lock = asyncio.Lock()
        
        # 标记为已修改
        self._dirty = False
        
        logger.debug("Session created", session_id=self.session_id)
    
    @staticmethod
    def _generate_id() -> str:
        return f"session_{uuid.uuid4().hex[:16]}"
    
    # ===== 状态管理 (同步方法) =====
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取状态值（不自动更新访问计数）
        """
        return self.state.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        设置状态值（同步）
        """
        self.state[key] = value
        self.metadata.updated_at = time.time()
        self._dirty = True
        self._try_schedule_auto_save()
    
    def delete(self, key: str) -> bool:
        """删除状态值"""
        if key in self.state:
            del self.state[key]
            self.metadata.updated_at = time.time()
            self._dirty = True
            self._try_schedule_auto_save()
            return True
        return False
    
    def clear(self):
        """清空所有状态"""
        self.state.clear()
        self.metadata.updated_at = time.time()
        self._dirty = True
        self._try_schedule_auto_save()
    
    def update(self, data: Dict[str, Any]):
        """批量更新状态"""
        self.state.update(data)
        self.metadata.updated_at = time.time()
        self._dirty = True
        self._try_schedule_auto_save()
    
    def touch(self):
        """手动更新访问时间和计数"""
        self.metadata.touch()
    
    def _try_schedule_auto_save(self):
        """
        尝试调度自动保存任务
        """
        if not self.auto_save or not self.storage:
            return

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._auto_save_task())
        except RuntimeError:
            pass

    async def _auto_save_task(self):
        """后台自动保存任务"""
        try:
            await self.save()
        except Exception as e:
            logger.error("Auto-save failed", session_id=self.session_id, error=str(e))

    # ===== 字典接口 =====
    
    def keys(self) -> List[str]:
        return list(self.state.keys())
    
    def values(self) -> List[Any]:
        return list(self.state.values())
    
    def items(self) -> List[tuple]:
        return list(self.state.items())
    
    def __contains__(self, key: str) -> bool:
        return key in self.state
    
    def __getitem__(self, key: str) -> Any:
        return self.state[key]
    
    def __setitem__(self, key: str, value: Any):
        self.set(key, value)
    
    # ===== 生命周期管理 =====
    
    def is_expired(self) -> bool:
        return self.metadata.is_expired()
    
    def extend_ttl(self, extra_seconds: int):
        if self.metadata.ttl is not None:
            self.metadata.ttl += extra_seconds
            self._dirty = True
    
    def renew(self):
        self.metadata.created_at = time.time()
        self._dirty = True
    
    # ===== 标签管理 =====
    
    def add_tag(self, tag: str):
        self.metadata.tags.add(tag)
        self._dirty = True
    
    def remove_tag(self, tag: str):
        self.metadata.tags.discard(tag)
        self._dirty = True
    
    def has_tag(self, tag: str) -> bool:
        return tag in self.metadata.tags
    
    # ===== 持久化 (异步方法) =====
    
    async def save(self, force: bool = False):
        """
        保存会话到存储
        """
        if not self.storage:
            return
        
        if not force and not self._dirty:
            return
        
        async with self._lock:
            try:
                data = self.to_dict()
                await self.storage.set(self.session_id, data)
                self._dirty = False
                
                logger.debug("Session saved", session_id=self.session_id)
                
                await self.event_bus.publish(SessionEvent(
                    type="session_saved",
                    data={"session_id": self.session_id}
                ))
            except Exception as e:
                logger.error(
                    "Failed to save session",
                    session_id=self.session_id,
                    error=str(e)
                )
                raise
    
    async def load(self) -> bool:
        """从存储加载会话"""
        if not self.storage:
            return False
        
        async with self._lock:
            try:
                data = await self.storage.get(self.session_id)
                if not data:
                    return False
                
                self.from_dict(data)
                self._dirty = False
                
                logger.debug("Session loaded", session_id=self.session_id)
                
                await self.event_bus.publish(SessionEvent(
                    type="session_loaded",
                    data={"session_id": self.session_id}
                ))
                
                return True
            except Exception as e:
                logger.error(
                    "Failed to load session",
                    session_id=self.session_id,
                    error=str(e)
                )
                return False
    
    async def destroy(self):
        """销毁会话（从存储中删除）"""
        if self.storage:
            async with self._lock:
                try:
                    await self.storage.delete(self.session_id)
                    logger.info("Session destroyed", session_id=self.session_id)
                    
                    await self.event_bus.publish(SessionEvent(
                        type="session_destroyed",
                        data={"session_id": self.session_id}
                    ))
                except Exception as e:
                    logger.error(
                        "Failed to destroy session",
                        session_id=self.session_id,
                        error=str(e)
                    )
        
        self.state.clear()
    
    # ===== 序列化 =====
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "metadata": self.metadata.model_dump(),
        }
    
    def from_dict(self, data: Dict[str, Any]):
        self.state = data.get("state", {})
        
        metadata_data = data.get("metadata", {})
        if metadata_data:
            self.metadata = SessionMetadata(**metadata_data)
    
    def clone(self, new_id: Optional[str] = None) -> Session:
        cloned = Session(
            session_id=new_id,
            state=self.state.copy(),
            storage=self.storage,
            ttl=self.metadata.ttl,
            event_bus=self.event_bus,
            auto_save=False,
        )
        cloned.metadata.tags = self.metadata.tags.copy()
        cloned.metadata.custom = self.metadata.custom.copy()
        return cloned
    
    # ===== 上下文管理器支持 =====
    
    async def __aenter__(self):
        await self.load()
        self.touch()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._dirty:
            await self.save()
    
    def __repr__(self) -> str:
        return (
            f"Session("
            f"id={self.session_id}, "
            f"keys={len(self.state)}, "
            f"dirty={self._dirty}"
            f")"
        )


# ===== 会话管理器 =====

class SessionManager:
    """会话管理器"""
    
    def __init__(
        self,
        storage: Optional[SessionInterface] = None,
        default_ttl: Optional[int] = None,
        auto_cleanup: bool = True,
        cleanup_interval: int = 300,
    ):
        self.storage = storage
        self.default_ttl = default_ttl
        self.auto_cleanup = auto_cleanup
        self.cleanup_interval = cleanup_interval
        
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()
        
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info("SessionManager initialized", default_ttl=default_ttl)
    
    async def start(self):
        """启动管理器（主要是自动清理任务）"""
        if self.auto_cleanup and not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.debug("Cleanup task started", interval=self.cleanup_interval)

    async def _cleanup_loop(self):
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup task error", error=str(e))
                await asyncio.sleep(60)

    async def create_session(
        self,
        session_id: Optional[str] = None,
        ttl: Optional[int] = None,
        **initial_state
    ) -> Session:
        """创建新会话"""
        async with self._lock:
            session = Session(
                session_id=session_id,
                state=initial_state,
                storage=self.storage,
                ttl=ttl or self.default_ttl,
                auto_save=True, 
            )
            
            self._sessions[session.session_id] = session
            
            if self.storage:
                await session.save()
            
            return session
    
    async def get_session(
        self,
        session_id: str,
        create_if_missing: bool = False
    ) -> Optional[Session]:
        """获取会话"""
        if session_id in self._sessions:
            session = self._sessions[session_id]
            if session.is_expired():
                await self.destroy_session(session_id)
                if create_if_missing:
                    return await self.create_session(session_id=session_id)
                return None
            session.touch()
            return session
        
        if self.storage:
            session = Session(
                session_id=session_id,
                storage=self.storage,
                auto_save=True,
            )
            if await session.load():
                if session.is_expired():
                    await session.destroy()
                    if create_if_missing:
                        return await self.create_session(session_id=session_id)
                    return None
                async with self._lock:
                    self._sessions[session_id] = session
                session.touch()
                return session
        
        if create_if_missing:
            return await self.create_session(session_id=session_id)
        
        return None
    
    async def destroy_session(self, session_id: str) -> bool:
        """销毁会话"""
        async with self._lock:
            session = self._sessions.pop(session_id, None)
            
            if session:
                await session.destroy()
                return True
            elif self.storage:
                try:
                    await self.storage.delete(session_id)
                    return True
                except Exception as e:
                    logger.error("Failed to destroy session", session_id=session_id, error=str(e))
        return False
    
    async def cleanup_expired(self) -> int:
        """清理所有过期会话"""
        expired_ids = []
        async with self._lock:
            for session_id, session in self._sessions.items():
                if session.is_expired():
                    expired_ids.append(session_id)
        
        for session_id in expired_ids:
            await self.destroy_session(session_id)
        
        if expired_ids:
            logger.info("Expired sessions cleaned", count=len(expired_ids))
        return len(expired_ids)
    
    def get_active_count(self) -> int:
        """获取活跃会话数量"""
        return len(self._sessions)
    
    def get_all_sessions(self) -> List[Session]:
        """获取所有活跃会话"""
        return list(self._sessions.values())
    
    async def shutdown(self):
        """关闭管理器"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        sessions = list(self._sessions.values())
        for session in sessions:
            await session.save(force=True)
        
        logger.info("SessionManager shutdown", sessions_saved=len(sessions))