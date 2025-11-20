# gecko/plugins/storage/sqlite.py (改进版)
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, Optional

from sqlmodel import SQLModel, Field, create_engine, select, Session as SQLSession

from gecko.plugins.storage.registry import register_storage
from gecko.plugins.storage.interfaces import SessionInterface
from gecko.plugins.storage.utils import parse_storage_url, validate_storage_url  # ✅ 使用统一解析
from gecko.core.logging import get_logger

logger = get_logger(__name__)

class SessionModel(SQLModel, table=True):
    """会话数据模型"""
    __tablename__ = "gecko_sessions"
    session_id: str = Field(primary_key=True)
    state_json: str = Field(default="{}")

@register_storage("sqlite")
class SQLiteSessionStorage(SessionInterface):
    """
    SQLite 会话存储（改进版）
    
    改进：
    1. 使用统一 URL 解析
    2. 支持 URL 参数
    3. 更好的错误处理
    """
    
    def __init__(self, storage_url: str, **kwargs):
        # ✅ 使用统一解析
        validate_storage_url(storage_url, required_scheme="sqlite")
        scheme, db_path, params = parse_storage_url(storage_url)
        
        # 处理 :memory:
        if db_path == ":memory:":
            self.db_path = ":memory:"
        else:
            self.db_path = db_path
            # 创建父目录
            if db_path != ":memory:":
                path_obj = Path(db_path)
                if not path_obj.parent.exists():
                    path_obj.parent.mkdir(parents=True, exist_ok=True)
                    logger.info("Created database directory", path=str(path_obj.parent))
        
        # 应用 URL 参数
        timeout = params.get("timeout", "30")
        
        # 创建引擎
        connect_args = {"timeout": int(timeout)}
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,
            connect_args=connect_args
        )
        
        # 建表
        SQLModel.metadata.create_all(self.engine)
        logger.info("SQLite storage initialized", db_path=self.db_path)

    async def get(self, session_id: str) -> Dict[str, Any] | None:
        """获取会话状态"""
        try:
            with SQLSession(self.engine) as session:
                statement = select(SessionModel).where(
                    SessionModel.session_id == session_id
                )
                result = session.exec(statement).first()
                
                if result:
                    return json.loads(result.state_json)
                return None
        except Exception as e:
            logger.error("Failed to get session", session_id=session_id, error=str(e))
            raise

    async def set(self, session_id: str, state: Dict[str, Any]) -> None:
        """设置会话状态"""
        try:
            with SQLSession(self.engine) as session:
                statement = select(SessionModel).where(
                    SessionModel.session_id == session_id
                )
                existing = session.exec(statement).first()
                
                json_str = json.dumps(state, ensure_ascii=False)
                
                if existing:
                    existing.state_json = json_str
                    session.add(existing)
                else:
                    new_session = SessionModel(
                        session_id=session_id,
                        state_json=json_str
                    )
                    session.add(new_session)
                
                session.commit()
        except Exception as e:
            logger.error("Failed to set session", session_id=session_id, error=str(e))
            raise

    async def delete(self, session_id: str) -> None:
        """删除会话"""
        try:
            with SQLSession(self.engine) as session:
                statement = select(SessionModel).where(
                    SessionModel.session_id == session_id
                )
                result = session.exec(statement).first()
                
                if result:
                    session.delete(result)
                    session.commit()
        except Exception as e:
            logger.error("Failed to delete session", session_id=session_id, error=str(e))
            raise