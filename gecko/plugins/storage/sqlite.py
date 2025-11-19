# gecko/plugins/storage/sqlite.py
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

# 引入 SQLModel (Pydantic + SQLAlchemy)
try:
    from sqlmodel import SQLModel, Field, create_engine, select, Session as SQLSession
except ImportError:
    raise ImportError("请安装 sqlmodel: pip install sqlmodel")

from gecko.plugins.storage.registry import register_storage
from gecko.plugins.storage.interfaces import SessionInterface

# 定义表结构
class SessionModel(SQLModel, table=True):
    __tablename__ = "gecko_sessions"
    session_id: str = Field(primary_key=True)
    state_json: str = Field(default="{}")

@register_storage("sqlite")
class SQLiteSessionStorage(SessionInterface):
    """
    基于 SQLite 的本地 Session 存储
    URL 示例: sqlite://./local_memory.db
    """
    def __init__(self, storage_url: str, **kwargs):
        # 解析路径: sqlite://./data/db.sqlite -> ./data/db.sqlite
        db_path = storage_url.removeprefix("sqlite://")
        
        # 自动创建父目录
        if db_path != ":memory:":
            path_obj = Path(db_path)
            if not path_obj.parent.exists():
                path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        # 创建引擎
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        
        # 自动建表
        SQLModel.metadata.create_all(self.engine)

    async def get(self, session_id: str) -> Dict[str, Any] | None:
        # SQLite 本身是同步的，但在 gecko 中我们包装成异步接口以统一协议
        # 生产环境如果使用 SQLite，建议配合 run_in_executor，但本地开发直接调用即可
        with SQLSession(self.engine) as session:
            statement = select(SessionModel).where(SessionModel.session_id == session_id)
            result = session.exec(statement).first()
            if result:
                return json.loads(result.state_json)
            return None

    async def set(self, session_id: str, state: Dict[str, Any]) -> None:
        with SQLSession(self.engine) as session:
            statement = select(SessionModel).where(SessionModel.session_id == session_id)
            existing = session.exec(statement).first()
            
            json_str = json.dumps(state, ensure_ascii=False)
            
            if existing:
                existing.state_json = json_str
                session.add(existing)
            else:
                new_session = SessionModel(session_id=session_id, state_json=json_str)
                session.add(new_session)
            session.commit()

    async def delete(self, session_id: str) -> None:
        with SQLSession(self.engine) as session:
            statement = select(SessionModel).where(SessionModel.session_id == session_id)
            result = session.exec(statement).first()
            if result:
                session.delete(result)
                session.commit()