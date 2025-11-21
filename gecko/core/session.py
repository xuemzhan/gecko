# gecko/core/session.py
"""
会话管理系统

提供会话的创建、存储、检索和生命周期管理。

核心功能：
1. 会话状态管理（内存 + 持久化）
2. 会话元数据（创建时间、更新时间、访问次数等）
3. 会话生命周期（过期、自动清理）
4. 并发安全（异步锁）
5. 会话克隆和序列化

优化点：
1. 集成存储后端
2. 完善的元数据管理
3. TTL 和自动过期
4. 并发安全
5. 事件通知
"""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from gecko.core.events import BaseEvent, EventBus
from gecko.core.logging import get_logger
from gecko.plugins.storage.interfaces import SessionInterface

logger = get_logger(__name__)


# ===== 会话元数据 =====

class SessionMetadata(BaseModel):
    """
    会话元数据
    
    属性:
        session_id: 会话唯一标识
        created_at: 创建时间戳
        updated_at: 最后更新时间戳
        accessed_at: 最后访问时间戳
        access_count: 访问次数
        ttl: 生存时间（秒），None 表示永不过期
        tags: 会话标签
        custom: 自定义元数据
    """
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
        """
        获取距离过期的剩余时间（秒）
        
        返回:
            剩余时间，None 表示永不过期，负数表示已过期
        """
        if self.ttl is None:
            return None
        
        age = time.time() - self.created_at
        return self.ttl - age
    
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
    
    管理单个会话的状态和元数据。
    
    示例:
        ```python
        # 创建会话
        session = Session(session_id="user_123")
        
        # 设置状态
        session.set("user_name", "Alice")
        session.set("preferences", {"theme": "dark"})
        
        # 获取状态
        name = session.get("user_name")
        
        # 检查过期
        if not session.is_expired():
            print("会话有效")
        
        # 持久化
        await session.save()
        ```
    """
    
    def __init__(
        self,
        session_id: Optional[str] = None,
        state: Optional[Dict[str, Any]] = None,
        storage: Optional[SessionInterface] = None,
        ttl: Optional[int] = None,
        event_bus: Optional[EventBus] = None,
        auto_save: bool = True,
    ):
        """
        初始化会话
        
        参数:
            session_id: 会话 ID（None 则自动生成）
            state: 初始状态
            storage: 存储后端（可选）
            ttl: 生存时间（秒）
            event_bus: 事件总线（可选）
            auto_save: 是否自动保存到存储
        """
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
        
        # 标记为已修改（用于优化持久化）
        self._dirty = False
        
        logger.debug("Session created", session_id=self.session_id)
    
    @staticmethod
    def _generate_id() -> str:
        """生成唯一会话 ID"""
        return f"session_{uuid.uuid4().hex[:16]}"
    
    # ===== 状态管理 =====
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取状态值
        
        参数:
            key: 键
            default: 默认值
        
        返回:
            状态值
        """
        self.metadata.touch()
        return self.state.get(key, default)
    
    def set(self, key: str, value: Any):
        """
        设置状态值
        
        参数:
            key: 键
            value: 值
        """
        self.state[key] = value
        self.metadata.updated_at = time.time()
        self.metadata.touch()
        self._dirty = True
        
        # 发布事件
        self.event_bus.publish(SessionEvent(
            type="session_updated",
            data={"session_id": self.session_id, "key": key}
        ))
        
        # 自动保存
        if self.auto_save and self.storage:
            asyncio.create_task(self.save())
    
    def delete(self, key: str) -> bool:
        """
        删除状态值
        
        参数:
            key: 键
        
        返回:
            是否成功删除
        """
        if key in self.state:
            del self.state[key]
            self.metadata.updated_at = time.time()
            self._dirty = True
            
            if self.auto_save and self.storage:
                asyncio.create_task(self.save())
            
            return True
        return False
    
    def clear(self):
        """清空所有状态"""
        self.state.clear()
        self.metadata.updated_at = time.time()
        self._dirty = True
        
        if self.auto_save and self.storage:
            asyncio.create_task(self.save())
    
    def update(self, data: Dict[str, Any]):
        """
        批量更新状态
        
        参数:
            data: 要更新的数据
        """
        self.state.update(data)
        self.metadata.updated_at = time.time()
        self._dirty = True
        
        if self.auto_save and self.storage:
            asyncio.create_task(self.save())
    
    def keys(self) -> List[str]:
        """获取所有键"""
        return list(self.state.keys())
    
    def values(self) -> List[Any]:
        """获取所有值"""
        return list(self.state.values())
    
    def items(self) -> List[tuple]:
        """获取所有键值对"""
        return list(self.state.items())
    
    def __contains__(self, key: str) -> bool:
        """支持 in 操作"""
        return key in self.state
    
    def __getitem__(self, key: str) -> Any:
        """支持 [] 读取"""
        return self.state[key]
    
    def __setitem__(self, key: str, value: Any):
        """支持 [] 设置"""
        self.set(key, value)
    
    # ===== 生命周期管理 =====
    
    def is_expired(self) -> bool:
        """检查会话是否过期"""
        return self.metadata.is_expired()
    
    def extend_ttl(self, extra_seconds: int):
        """
        延长生存时间
        
        参数:
            extra_seconds: 额外的秒数
        """
        if self.metadata.ttl is not None:
            self.metadata.ttl += extra_seconds
            logger.debug(
                "Session TTL extended",
                session_id=self.session_id,
                new_ttl=self.metadata.ttl
            )
    
    def renew(self):
        """
        重置会话（从当前时间重新计算 TTL）
        """
        self.metadata.created_at = time.time()
        logger.debug("Session renewed", session_id=self.session_id)
    
    # ===== 标签管理 =====
    
    def add_tag(self, tag: str):
        """添加标签"""
        self.metadata.tags.add(tag)
        self._dirty = True
    
    def remove_tag(self, tag: str):
        """移除标签"""
        self.metadata.tags.discard(tag)
        self._dirty = True
    
    def has_tag(self, tag: str) -> bool:
        """检查是否有标签"""
        return tag in self.metadata.tags
    
    # ===== 持久化 =====
    
    async def save(self, force: bool = False):
        """
        保存会话到存储
        
        参数:
            force: 是否强制保存（忽略 _dirty 标记）
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
                
                self.event_bus.publish(SessionEvent(
                    type="session_saved",
                    data={"session_id": self.session_id}
                ))
            except Exception as e:
                logger.error(
                    "Failed to save session",
                    session_id=self.session_id,
                    error=str(e)
                )
    
    async def load(self) -> bool:
        """
        从存储加载会话
        
        返回:
            是否成功加载
        """
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
                
                self.event_bus.publish(SessionEvent(
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
        """
        销毁会话（从存储中删除）
        """
        if self.storage:
            async with self._lock:
                try:
                    await self.storage.delete(self.session_id)
                    logger.info("Session destroyed", session_id=self.session_id)
                    
                    self.event_bus.publish(SessionEvent(
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
        """
        序列化为字典
        
        返回:
            包含状态和元数据的字典
        """
        return {
            "state": self.state,
            "metadata": self.metadata.model_dump(),
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """
        从字典反序列化
        
        参数:
            data: 数据字典
        """
        self.state = data.get("state", {})
        
        metadata_data = data.get("metadata", {})
        if metadata_data:
            self.metadata = SessionMetadata(**metadata_data)
    
    def clone(self, new_id: Optional[str] = None) -> Session:
        """
        克隆会话
        
        参数:
            new_id: 新会话 ID（None 则自动生成）
        
        返回:
            新的 Session 实例
        """
        cloned = Session(
            session_id=new_id,
            state=self.state.copy(),
            storage=self.storage,
            ttl=self.metadata.ttl,
            event_bus=self.event_bus,
            auto_save=False,  # 克隆时不自动保存
        )
        
        # 复制标签
        cloned.metadata.tags = self.metadata.tags.copy()
        cloned.metadata.custom = self.metadata.custom.copy()
        
        return cloned
    
    # ===== 调试信息 =====
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取会话信息（用于调试/监控）
        
        返回:
            会话信息字典
        """
        return {
            "session_id": self.session_id,
            "state_keys": len(self.state),
            "created_at": datetime.fromtimestamp(self.metadata.created_at).isoformat(),
            "updated_at": datetime.fromtimestamp(self.metadata.updated_at).isoformat(),
            "accessed_at": datetime.fromtimestamp(self.metadata.accessed_at).isoformat(),
            "access_count": self.metadata.access_count,
            "ttl": self.metadata.ttl,
            "time_to_expire": self.metadata.time_to_expire(),
            "is_expired": self.is_expired(),
            "tags": list(self.metadata.tags),
            "has_storage": self.storage is not None,
            "is_dirty": self._dirty,
        }
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"Session("
            f"id={self.session_id}, "
            f"keys={len(self.state)}, "
            f"accessed={self.metadata.access_count}, "
            f"expired={self.is_expired()}"
            f")"
        )


# ===== 会话管理器 =====

class SessionManager:
    """
    会话管理器
    
    负责管理多个会话的生命周期、持久化和清理。
    
    示例:
        ```python
        # 创建管理器
        manager = SessionManager(storage=storage)
        
        # 创建会话
        session = await manager.create_session(ttl=3600)
        
        # 获取会话
        session = await manager.get_session(session_id)
        
        # 清理过期会话
        await manager.cleanup_expired()
        ```
    """
    
    def __init__(
        self,
        storage: Optional[SessionInterface] = None,
        default_ttl: Optional[int] = None,
        auto_cleanup: bool = True,
        cleanup_interval: int = 300,  # 5 分钟
    ):
        """
        初始化会话管理器
        
        参数:
            storage: 存储后端
            default_ttl: 默认 TTL（秒）
            auto_cleanup: 是否自动清理过期会话
            cleanup_interval: 清理间隔（秒）
        """
        self.storage = storage
        self.default_ttl = default_ttl
        self.auto_cleanup = auto_cleanup
        self.cleanup_interval = cleanup_interval
        
        # 内存缓存（session_id -> Session）
        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()
        
        # 自动清理任务
        self._cleanup_task: Optional[asyncio.Task] = None
        
        if auto_cleanup:
            self._start_cleanup_task()
        
        logger.info("SessionManager initialized", default_ttl=default_ttl)
    
    def _start_cleanup_task(self):
        """启动自动清理任务"""
        async def _cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.cleanup_interval)
                    await self.cleanup_expired()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Cleanup task error", error=str(e))
        
        self._cleanup_task = asyncio.create_task(_cleanup_loop())
        logger.debug("Cleanup task started", interval=self.cleanup_interval)
    
    async def create_session(
        self,
        session_id: Optional[str] = None,
        ttl: Optional[int] = None,
        **initial_state
    ) -> Session:
        """
        创建新会话
        
        参数:
            session_id: 会话 ID（None 则自动生成）
            ttl: 生存时间（None 则使用默认值）
            **initial_state: 初始状态
        
        返回:
            Session 实例
        """
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
            
            logger.info("Session created", session_id=session.session_id)
            return session
    
    async def get_session(
        self,
        session_id: str,
        create_if_missing: bool = False
    ) -> Optional[Session]:
        """
        获取会话
        
        参数:
            session_id: 会话 ID
            create_if_missing: 如果不存在是否创建
        
        返回:
            Session 实例，如果不存在返回 None
        """
        # 1. 检查内存缓存
        if session_id in self._sessions:
            session = self._sessions[session_id]
            
            # 检查过期
            if session.is_expired():
                await self.destroy_session(session_id)
                if create_if_missing:
                    return await self.create_session(session_id=session_id)
                return None
            
            return session
        
        # 2. 尝试从存储加载
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
                
                return session
        
        # 3. 创建新会话（如果允许）
        if create_if_missing:
            return await self.create_session(session_id=session_id)
        
        return None
    
    async def destroy_session(self, session_id: str) -> bool:
        """
        销毁会话
        
        参数:
            session_id: 会话 ID
        
        返回:
            是否成功
        """
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
        """
        清理所有过期会话
        
        返回:
            清理的会话数量
        """
        expired_ids = []
        
        async with self._lock:
            for session_id, session in list(self._sessions.items()):
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
        """关闭管理器（取消清理任务，保存所有会话）"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # 保存所有会话
        for session in self._sessions.values():
            await session.save(force=True)
        
        logger.info("SessionManager shutdown", sessions_saved=len(self._sessions))