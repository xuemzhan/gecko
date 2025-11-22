"""会话元数据定义"""
from __future__ import annotations
import time
from typing import Any, Dict, Optional, Set
from pydantic import BaseModel, Field
from gecko.core.logging import get_logger

logger = get_logger(__name__)

class SessionMetadata(BaseModel):
    """会话元数据"""
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
        """获取距离过期的剩余时间（秒）"""
        if self.ttl is None:
            return None
        age = time.time() - self.created_at
        return max(0.0, self.ttl - age)
    
    def touch(self):
        """更新访问时间和计数"""
        self.accessed_at = time.time()
        self.access_count += 1
