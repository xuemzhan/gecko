# gecko/plugins/storage/backends/sqlite.py
"""
SQLite 存储后端 (高并发优化版)

优化策略：
1. Mixin 组合：继承 ThreadOffloadMixin 防止阻塞 Loop，继承 AtomicWriteMixin 防止写冲突。
2. WAL 模式：开启 Write-Ahead Logging，大幅提升读写并发性能。
3. 线程安全：配置 check_same_thread=False 以支持在线程池中使用连接。
4. 进程安全：启用 FileLock 防止多进程环境下的 Database is locked 错误。

更新日志：
- [Arch] 初始化时配置 FileLock。
- [Robustness] 所有操作统一抛出 StorageError。
- [Perf] 保持 WAL 模式优化。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy import text
from sqlmodel import Field, Session, SQLModel, create_engine, select

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

logger = get_logger(__name__)


class SessionModel(SQLModel, table=True):
    """SQLModel 表定义"""
    __tablename__ = "gecko_sessions" # type: ignore
    session_id: str = Field(primary_key=True)
    state_json: str = Field(default="{}")

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
        self.is_memory = path == ":memory:"
        
        try:
            if not self.is_memory:
                # 确保父目录存在
                db_file = Path(path)
                if not db_file.parent.exists():
                    db_file.parent.mkdir(parents=True, exist_ok=True)
                
                # [架构优化] 配置跨进程文件锁
                self.setup_multiprocess_lock(self.db_path)
            
            if self.is_memory:
                sqlalchemy_url = "sqlite:///:memory:"
            else:
                sqlalchemy_url = f"sqlite:///{self.db_path}"

            self.engine = create_engine(
                sqlalchemy_url,
                connect_args={"check_same_thread": False, "timeout": 30},
            )
        except Exception as e:
            raise StorageError(f"Failed to configure SQLite: {e}") from e

    async def initialize(self) -> None:
        if self.is_initialized:
            return

        try:
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
        def _sync_get():
            try:
                with Session(self.engine) as session:
                    statement = select(SessionModel).where(
                        SessionModel.session_id == session_id
                    )
                    result = session.exec(statement).first()
                    return result.state_json if result else None
            except Exception as e:
                raise e

        try:
            json_str = await self._run_sync(_sync_get)
            return self._deserialize(json_str)
        except Exception as e:
            raise StorageError(f"SQLite get failed: {e}") from e

    async def set(self, session_id: str, state: Dict[str, Any]) -> None:
        json_str = self._serialize(state)

        def _sync_set():
            # [修复] 在 Worker 线程内部获取文件锁
            with self.file_lock_guard():
                try:
                    with Session(self.engine) as session:
                        statement = select(SessionModel).where(
                            SessionModel.session_id == session_id
                        )
                        existing = session.exec(statement).first()
                        
                        if existing:
                            existing.state_json = json_str
                            session.add(existing)
                        else:
                            new_rec = SessionModel(
                                session_id=session_id, 
                                state_json=json_str
                            )
                            session.add(new_rec)
                        session.commit()
                except Exception as e:
                    raise e

        try:
            # 获取协程锁 -> 切换线程 -> 获取文件锁 -> 操作 -> 释放
            async with self.write_guard():
                await self._run_sync(_sync_set)
        except Exception as e:
            raise StorageError(f"SQLite set failed: {e}") from e

    async def delete(self, session_id: str) -> None:
        def _sync_delete():
            # [修复] 在 Worker 线程内部获取文件锁
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