# gecko/plugins/storage/backends/sqlite.py
"""
SQLite 存储后端 (高并发优化版)

优化策略：
1. Mixin 组合：继承 ThreadOffloadMixin 防止阻塞 Loop，继承 AtomicWriteMixin 防止写冲突。
2. WAL 模式：开启 Write-Ahead Logging，大幅提升读写并发性能。
3. 线程安全：配置 check_same_thread=False 以支持在线程池中使用连接。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy import text
from sqlmodel import Field, Session, SQLModel, create_engine, select

from gecko.core.logging import get_logger
from gecko.plugins.storage.abc import AbstractStorage
from gecko.plugins.storage.interfaces import SessionInterface
from gecko.plugins.storage.mixins import (
    AtomicWriteMixin,
    JSONSerializerMixin,
    ThreadOffloadMixin,
)
from gecko.plugins.storage.utils import parse_storage_url

logger = get_logger(__name__)


class SessionModel(SQLModel, table=True):
    """SQLModel 表定义"""
    __tablename__ = "gecko_sessions" # type: ignore
    session_id: str = Field(primary_key=True)
    state_json: str = Field(default="{}")


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
        
        # 处理文件路径
        self.db_path = path
        self.is_memory = path == ":memory:"
        
        if not self.is_memory:
            # 确保父目录存在
            db_file = Path(path)
            if not db_file.parent.exists():
                db_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 构造 SQLAlchemy URL
        # 注意：sqlite://<path> (3 slashes for relative, 4 for absolute)
        # 这里简化处理，假设 utils 解析出的 path 已经是文件系统路径
        if self.is_memory:
            sqlalchemy_url = "sqlite:///:memory:"
        else:
            sqlalchemy_url = f"sqlite:///{self.db_path}"

        # 创建引擎
        # check_same_thread=False: 允许在不同线程使用连接（必须，因为用了 ThreadOffload）
        # timeout=30: 增加 SQLite 忙等待时间
        self.engine = create_engine(
            sqlalchemy_url,
            connect_args={"check_same_thread": False, "timeout": 30},
            # echo=True # 调试时开启
        )

    async def initialize(self) -> None:
        """异步初始化：建表 + 开启 WAL"""
        if self.is_initialized:
            return

        # 1. 建表 (在线程中执行)
        await self._run_sync(SQLModel.metadata.create_all, self.engine)
        
        # 2. 开启 WAL 模式 (提升并发)
        # 内存数据库不需要 WAL
        if not self.is_memory:
            def _enable_wal():
                with self.engine.connect() as conn:
                    conn.execute(text("PRAGMA journal_mode=WAL;"))
                    conn.execute(text("PRAGMA synchronous=NORMAL;"))
            
            await self._run_sync(_enable_wal)
            logger.debug("SQLite WAL mode enabled", path=self.db_path)
            
        self._is_initialized = True
        logger.info("SQLite storage initialized", url=self.url)

    async def shutdown(self) -> None:
        """释放资源"""
        self.engine.dispose()
        self._is_initialized = False

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取 Session"""
        def _sync_get():
            with Session(self.engine) as session:
                statement = select(SessionModel).where(
                    SessionModel.session_id == session_id
                )
                result = session.exec(statement).first()
                return result.state_json if result else None

        # 卸载到线程池读取
        json_str = await self._run_sync(_sync_get)
        return self._deserialize(json_str)

    async def set(self, session_id: str, state: Dict[str, Any]) -> None:
        """保存 Session"""
        json_str = self._serialize(state)

        def _sync_set():
            with Session(self.engine) as session:
                # 查询是否存在
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

        # 写锁 + 线程卸载
        async with self.write_guard():
            await self._run_sync(_sync_set)

    async def delete(self, session_id: str) -> None:
        """删除 Session"""
        def _sync_delete():
            with Session(self.engine) as session:
                statement = select(SessionModel).where(
                    SessionModel.session_id == session_id
                )
                result = session.exec(statement).first()
                if result:
                    session.delete(result)
                    session.commit()
        
        # 写锁 + 线程卸载
        async with self.write_guard():
            await self._run_sync(_sync_delete)