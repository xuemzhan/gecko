"""会话管理器"""
from __future__ import annotations
import asyncio
from typing import Dict, List, Optional
from gecko.plugins.storage.interfaces import SessionInterface
from gecko.core.session.entity import Session
from gecko.core.logging import get_logger

logger = get_logger(__name__)

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