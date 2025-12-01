# gecko/plugins/storage/backends/sqlite.py
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy import text
from sqlmodel import Field, Session, SQLModel, create_engine, select, delete

from gecko.core.exceptions import StorageError
from gecko.core.logging import get_logger
from gecko.plugins.storage.abc import AbstractStorage
from gecko.plugins.storage.interfaces import SessionInterface
from gecko.plugins.storage.mixins import (
    AtomicWriteMixin,
    JSONSerializerMixin,
    ThreadOffloadMixin,
)
from gecko.plugins.storage.registry import register_storage
from gecko.plugins.storage.utils import parse_storage_url
from gecko.config import get_settings

logger = get_logger(__name__)


class SessionModel(SQLModel, table=True):
    """SQLModel 表定义"""
    __tablename__ = "gecko_sessions" # type: ignore
    session_id: str = Field(primary_key=True)
    state_json: str = Field(default="{}")
    # 新增字段用于 TTL 支持
    expire_at: Optional[float] = Field(default=None, index=True)


@register_storage("sqlite")
class SQLiteStorage(
    AbstractStorage,
    SessionInterface,
    ThreadOffloadMixin,
    AtomicWriteMixin,
    JSONSerializerMixin
):
    def __init__(self, url: str, **kwargs):
        super().__init__(url, **kwargs)
        
        scheme, path, params = parse_storage_url(url)
        
        self.db_path = path
        # [修复] 兼容 sqlite:///:memory: (path=/:memory:) 和 sqlite://:memory: (path=:memory:)
        self.is_memory = path in (":memory:", "/:memory:")

        # [Fix] 动态获取最新配置
        settings = get_settings()
        
        try:
            if self.is_memory:
                # 内存模式统一使用 sqlalchemy 可识别的 URL
                sqlalchemy_url = "sqlite:///:memory:"
                connect_args = {"check_same_thread": False}
                pool_args = {}
            else:
                # 文件模式
                try:
                    db_file = Path(self.db_path)
                    if not db_file.parent.exists():
                        db_file.parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise StorageError(f"Failed to configure SQLite path: {e}") from e

                # 仅在文件模式下启用文件锁
                self.setup_multiprocess_lock(self.db_path)

                sqlalchemy_url = f"sqlite:///{self.db_path}"
                connect_args = {"check_same_thread": False, "timeout": 30}
                
                pool_args = {
                    "pool_size": kwargs.get("pool_size", settings.storage_pool_size),
                    "max_overflow": kwargs.get("max_overflow", settings.storage_max_overflow),
                    "pool_recycle": 3600,
                }

            self.engine = create_engine(
                sqlalchemy_url,
                connect_args=connect_args,
                **pool_args
            )
        except Exception as e:
            if isinstance(e, StorageError):
                raise e
            raise StorageError(f"Failed to configure SQLite: {e}") from e

    async def initialize(self) -> None:
        if self.is_initialized:
            return

        try:
            # 确保表结构最新
            await self._run_sync(SQLModel.metadata.create_all, self.engine)
            
            if not self.is_memory:
                def _enable_wal():
                    with self.engine.connect() as conn:
                        conn.execute(text("PRAGMA journal_mode=WAL;"))
                        conn.execute(text("PRAGMA synchronous=NORMAL;"))
                
                await self._run_sync(_enable_wal)
                logger.debug("SQLite WAL mode enabled", path=self.db_path)
                
            self._is_initialized = True
            logger.info("SQLite storage initialized", url=self.url)
        except Exception as e:
            raise StorageError(f"Failed to initialize SQLite: {e}") from e

    async def shutdown(self) -> None:
        try:
            self.engine.dispose()
        except Exception as e:
            logger.warning(f"Error during SQLite shutdown: {e}")
        finally:
            self._is_initialized = False

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        # 性能优化：反序列化移入子线程
        def _sync_get_and_deserialize():
            try:
                with Session(self.engine) as session:
                    statement = select(SessionModel).where(
                        SessionModel.session_id == session_id
                    )
                    result = session.exec(statement).first()
                    
                    if not result:
                        return None
                    
                    # 懒惰过期检查
                    if result.expire_at and result.expire_at < time.time():
                        return None

                    return self._deserialize(result.state_json)
            except Exception as e:
                raise e

        try:
            return await self._run_sync(_sync_get_and_deserialize)
        except Exception as e:
            raise StorageError(f"SQLite get failed: {e}") from e

    async def set(self, session_id: str, state: Dict[str, Any]) -> None:
        # 性能优化：序列化移入子线程
        def _sync_serialize_and_set():
            # 获取文件锁（内存模式下 file_lock_guard 为空操作）
            with self.file_lock_guard():
                try:
                    # 1. 序列化
                    json_str = self._serialize(state)
                    
                    # 2. 计算过期时间
                    expire_at = None
                    metadata = state.get("metadata", {})
                    ttl = metadata.get("ttl")
                    if isinstance(ttl, (int, float)) and ttl > 0:
                        updated_at = metadata.get("updated_at", time.time())
                        expire_at = updated_at + ttl

                    # 3. 写入
                    with Session(self.engine) as session:
                        statement = select(SessionModel).where(
                            SessionModel.session_id == session_id
                        )
                        existing = session.exec(statement).first()
                        
                        if existing:
                            existing.state_json = json_str
                            existing.expire_at = expire_at
                            session.add(existing)
                        else:
                            new_rec = SessionModel(
                                session_id=session_id, 
                                state_json=json_str,
                                expire_at=expire_at
                            )
                            session.add(new_rec)
                        session.commit()
                except Exception as e:
                    raise e

        try:
            async with self.write_guard():
                await self._run_sync(_sync_serialize_and_set)
        except Exception as e:
            raise StorageError(f"SQLite set failed: {e}") from e

    async def delete(self, session_id: str) -> None:
        def _sync_delete():
            with self.file_lock_guard():
                try:
                    with Session(self.engine) as session:
                        statement = select(SessionModel).where(
                            SessionModel.session_id == session_id
                        )
                        result = session.exec(statement).first()
                        if result:
                            session.delete(result)
                            session.commit()
                except Exception as e:
                    raise e
        
        try:
            async with self.write_guard():
                await self._run_sync(_sync_delete)
        except Exception as e:
            raise StorageError(f"SQLite delete failed: {e}") from e

    async def cleanup_expired(self) -> int:
        """物理删除过期会话"""
        def _sync_cleanup():
            with self.file_lock_guard():
                try:
                    now = time.time()
                    with Session(self.engine) as session:
                        statement = delete(SessionModel).where(
                            SessionModel.expire_at != None, # type: ignore
                            SessionModel.expire_at < now # type: ignore
                        )
                        result = session.exec(statement) # type: ignore
                        session.commit()
                        return result.rowcount
                except Exception as e:
                    raise e
        
        try:
            async with self.write_guard():
                return await self._run_sync(_sync_cleanup)
        except Exception as e:
            logger.error("Failed to cleanup expired sessions", error=str(e))
            return 0