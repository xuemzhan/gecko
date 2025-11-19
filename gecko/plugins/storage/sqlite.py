# gecko/plugins/storage/sqlite.py
from __future__ import annotations
import json
from pathlib import Path
from sqlmodel import SQLModel, Field, create_engine, select
from sqlalchemy.orm import sessionmaker
from gecko.plugins.storage.registry import register_storage
from gecko.plugins.storage.interfaces import SessionInterface

class SessionState(SQLModel, table=True):
    """SQLite 表结构：session_id 主键 + state JSON 字符串"""
    session_id: str = Field(primary_key=True)
    state: str = Field(default="{}")

@register_storage("sqlite")
class SQLiteSessionStorage(SessionInterface):
    """
    快速开发专用 Session 存储插件
    - URL 示例：sqlite://./dev_sessions.db
    - 特点：0 外部依赖、单文件持久化、自动创建表
    - 适用：本地/Codespaces 验证 Session 持久化
    """
    def __init__(self, storage_url: str, **kwargs):
        db_path = storage_url.removeprefix("sqlite://")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        SQLModel.metadata.create_all(engine)
        self.SessionLocal = sessionmaker(bind=engine)

    async def get(self, session_id: str):
        """获取会话状态"""
        with self.SessionLocal() as db:
            result = db.exec(select(SessionState).where(SessionState.session_id == session_id)).first()
            return json.loads(result.state) if result else None

    async def set(self, session_id: str, state: Dict):
        """设置会话状态"""
        with self.SessionLocal() as db:
            obj = db.exec(select(SessionState).where(SessionState.session_id == session_id)).first()
            if obj:
                obj.state = json.dumps(state)
            else:
                db.add(SessionState(session_id=session_id, state=json.dumps(state)))
            db.commit()

    async def delete(self, session_id: str):
        """删除会话"""
        with self.SessionLocal() as db:
            obj = db.exec(select(SessionState).where(SessionState.session_id == session_id)).first()
            if obj:
                db.delete(obj)
                db.commit()