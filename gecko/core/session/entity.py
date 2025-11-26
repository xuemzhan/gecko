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
        auto_save_debounce: float = 0.1,  # 新增: 防抖延迟
    ):
        self.session_id = session_id or self._generate_id()
        self.state: Dict[str, Any] = state or {}
        self.storage = storage
        self.event_bus = event_bus or EventBus()
        self.auto_save = auto_save
        self.auto_save_debounce = auto_save_debounce  # 新增
        
        # 元数据
        self.metadata = SessionMetadata(
            session_id=self.session_id,
            ttl=ttl
        )
        
        # 并发锁
        self._lock = asyncio.Lock()
        # 标记为已修改
        self._dirty = False
        self._save_scheduled = False  # ✅ 新增: 保存调度标记
        self._save_task: Optional[asyncio.Task] = None  # ✅ 新增: 保存任务引用
        
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
        修复: 使用防抖机制调度自动保存
        """
        if not self.auto_save or not self.storage:
            return
        
        # 如果已有保存任务在等待，不重复调度
        if self._save_scheduled:
            return
        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 没有运行中的事件循环
            return
        
        self._save_scheduled = True
        
        async def _debounced_save():
            """防抖保存任务"""
            try:
                # 等待防抖延迟
                await asyncio.sleep(self.auto_save_debounce)
                
                # 执行保存
                if self._dirty:  # 再次检查是否需要保存
                    await self.save()
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error("Auto-save failed", session_id=self.session_id, error=str(e))
            finally:
                self._save_scheduled = False
                self._save_task = None
        
        # 取消之前的任务（如果存在）
        if self._save_task and not self._save_task.done():
            self._save_task.cancel()
        
        self._save_task = loop.create_task(_debounced_save())

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
        [优化] 保存会话到存储 (线程安全与一致性保证)
        
        策略：
        1. 持有锁时进行状态检查和数据快照 (Snapshotting)。
        2. 使用 utils.safe_serialize_context 确保数据深拷贝和清洗。
        3. 持有锁进行 IO 操作，确保写入顺序性（针对单个 Session 的并发保护）。
        """
        if not self.storage:
            return
        
        # 全程持有锁，防止在 Snapshotting 和 IO 之间状态发生变更
        # 虽然这会略微增加锁的持有时间，但对于单 Session 粒度来说，一致性优于微小的并发性能提升。
        async with self._lock:
            # 双重检查
            if not force and not self._dirty:
                return

            try:
                # 1. 准备数据快照 (CPU 密集型)
                # 引入 utils 中的序列化工具，它会递归处理并返回纯净的 dict/list 副本
                # 这实际上切断了 storage 数据与内存 self.state 的引用关系
                from gecko.core.utils import safe_serialize_context
                
                # 获取当前状态的字典表示
                raw_data = self.to_dict()
                
                # 清洗并深拷贝
                clean_data = safe_serialize_context(raw_data)
                
                # 2. 执行 IO (IO 密集型)
                # 这里的 await 会释放 GIL，但不会释放 self._lock (协程锁)
                # 因此其他协程无法在此期间修改 self.state 或发起新的 save
                await self.storage.set(self.session_id, clean_data)
                
                # 3. 状态重置
                # 只有成功写入后才清除 dirty 标记
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
                # 发生异常时，保持 dirty = True，以便下次重试
                # 抛出异常让上层知道保存失败
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
    
    def get_info(self) -> Dict[str, Any]:
        """
        [新增] 获取会话的统计信息
        """
        return {
            "session_id": self.session_id,
            "created_at": self.metadata.created_at,
            "updated_at": self.metadata.updated_at,
            "access_count": self.metadata.access_count,
            "state_keys": len(self.state),
            "ttl": self.metadata.ttl,
            "tags": list(self.metadata.tags)
        }
    
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

