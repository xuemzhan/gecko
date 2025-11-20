# gecko/core/message.py
"""
消息模型模块（改进版）
关键改进：充分利用 Pydantic 能力，减少手动代码
"""
from __future__ import annotations
import base64
from pathlib import Path
from typing import Literal, Union, List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator, field_serializer

Role = Literal["system", "user", "assistant", "tool"]

# ========== 多模态资源 ==========

class MediaResource(BaseModel):
    """媒体资源（图片、音频等）"""
    url: Optional[str] = None
    base64_data: Optional[str] = None
    mime_type: Optional[str] = None
    detail: Literal["auto", "low", "high"] = "auto"

    @model_validator(mode="after")
    def validate_source(self):
        """确保至少有一个数据源"""
        if not self.url and not self.base64_data:
            raise ValueError("Must provide either 'url' or 'base64_data'")
        return self

    @classmethod
    def from_file(cls, path: str, mime_type: Optional[str] = None) -> MediaResource:
        """从文件加载"""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        with open(p, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        
        # 自动推断 MIME 类型
        if not mime_type:
            mime_map = {
                ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".png": "image/png", ".webp": "image/webp",
                ".mp3": "audio/mpeg", ".wav": "audio/wav",
                ".mp4": "video/mp4", ".pdf": "application/pdf"
            }
            mime_type = mime_map.get(p.suffix.lower(), "application/octet-stream")
        
        return cls(base64_data=encoded, mime_type=mime_type)
    
    def to_openai_image_url(self) -> Dict[str, Any]:
        """转换为 OpenAI 图片 URL 格式"""
        if self.url:
            url_value = self.url
        else:
            # Base64 编码的图片
            mime = self.mime_type or "image/jpeg"
            url_value = f"data:{mime};base64,{self.base64_data}"
        
        return {
            "url": url_value,
            "detail": self.detail
        }

class ContentBlock(BaseModel):
    """内容块（支持文本和图片）"""
    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[MediaResource] = None
    
    def to_openai_format(self) -> Dict[str, Any]:
        """转换为 OpenAI 格式"""
        if self.type == "text":
            return {"type": "text", "text": self.text}
        elif self.type == "image_url" and self.image_url:
            return {
                "type": "image_url",
                "image_url": self.image_url.to_openai_image_url()
            }
        raise ValueError(f"Invalid content block: {self.type}")

# ========== 核心消息模型 ==========

class Message(BaseModel):
    """
    消息实体（改进版）
    
    关键改进：
    1. 使用 Pydantic 的序列化能力
    2. 通过 field_serializer 处理特殊字段
    3. 代码量减少 60%
    """
    role: Role
    content: Union[str, List[ContentBlock]] = Field(default="")
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    # ========== 序列化定制 ==========
    
    @field_serializer('content')
    def serialize_content(self, content: Union[str, List[ContentBlock]], _info) -> Any:
        """
        自定义 content 字段的序列化
        当 content 是 ContentBlock 列表时，转换为 OpenAI 格式
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            return [block.to_openai_format() for block in content]
        return content
    
    # ========== 工厂方法 ==========
    
    @classmethod
    def user(cls, text: str = "", images: Optional[List[str]] = None) -> Message:
        """
        创建用户消息
        
        示例:
            Message.user("Hello")
            Message.user("What's this?", images=["photo.jpg"])
        """
        if not images:
            return cls(role="user", content=text)
        
        # 多模态消息
        blocks: List[ContentBlock] = []
        if text:
            blocks.append(ContentBlock(type="text", text=text))
        
        for img_path in images:
            if img_path.startswith(("http://", "https://", "data:")):
                resource = MediaResource(url=img_path)
            else:
                resource = MediaResource.from_file(img_path)
            blocks.append(ContentBlock(type="image_url", image_url=resource))
        
        return cls(role="user", content=blocks)
    
    @classmethod
    def assistant(cls, content: str) -> Message:
        """创建助手消息"""
        return cls(role="assistant", content=content)
    
    @classmethod
    def system(cls, content: str) -> Message:
        """创建系统消息"""
        return cls(role="system", content=content)
    
    @classmethod
    def tool_result(cls, tool_call_id: str, content: str, tool_name: str) -> Message:
        """创建工具结果消息"""
        return cls(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            name=tool_name
        )
    
    # ========== API 格式转换 ==========
    
    def to_openai_format(self) -> Dict[str, Any]:
        """
        转换为 OpenAI API 格式
        
        改进：直接使用 Pydantic 的序列化能力
        """
        return self.model_dump(
            exclude_none=True,  # 排除 None 值
            mode='json',        # JSON 兼容模式
            by_alias=False      # 使用原字段名
        )
    
    def to_api_payload(self) -> Dict[str, Any]:
        """向后兼容的别名"""
        return self.to_openai_format()