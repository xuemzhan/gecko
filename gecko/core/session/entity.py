"""会话实体逻辑"""
from __future__ import annotations
import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional
from gecko.core.logging import get_logger
from gecko.core.events import EventBus, SessionEvent
# 引用新的 protocols 路径
from gecko.plugins.storage.interfaces import SessionInterface 
from gecko.core.session.schema import SessionMetadata

logger = get_logger(__name__)

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

