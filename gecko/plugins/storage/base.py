# gecko/plugins/storage/base.py
from __future__ import annotations
from pydantic import BaseModel, Field, AnyUrl

class BaseStorageConfig(BaseModel):
    """
    纯配置类：只包含 Pydantic 可验证的字段
    - 用于所有存储插件的统一配置
    - 避免运行时对象混入 Pydantic 模型导致冲突
    """
    storage_url: AnyUrl = Field(..., description="存储连接 URL，例如 sqlite://./db.db 或 lancedb://./db")
    collection_name: str = Field(default="gecko_default", description="集合/表名")
    embedding_dim: int = Field(default=1536, description="向量维度，默认兼容主流 LLM")
    
    model_config = {"arbitrary_types_allowed": False, "extra": "forbid"}  # 严格模式，避免任意类型