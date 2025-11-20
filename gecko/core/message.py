# gecko/core/message.py
from __future__ import annotations
import base64
from pathlib import Path
from typing import Literal, Union, List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator

Role = Literal["system", "user", "assistant", "tool"]
MediaType = Literal["text", "image_url", "audio_url", "video_url", "file_url"]

class MediaResource(BaseModel):
    """统一媒体资源定义"""
    url: Optional[str] = None
    base64_data: Optional[str] = None
    mime_type: Optional[str] = None
    detail: Literal["auto", "low", "high"] = "auto"

    @model_validator(mode="after")
    def check_source(self):
        if not self.url and not self.base64_data:
            raise ValueError("Must provide either 'url' or 'base64_data'")
        return self

    @classmethod
    def from_file(cls, path: str, mime_type: str = None):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        with open(p, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        
        if not mime_type:
            # 简单 MIME 推断
            suffix = p.suffix.lower()
            if suffix in [".jpg", ".jpeg"]: mime_type = "image/jpeg"
            elif suffix == ".png": mime_type = "image/png"
            elif suffix == ".webp": mime_type = "image/webp"
            elif suffix in [".mp3", ".wav"]: mime_type = "audio/mpeg"
            elif suffix == ".mp4": mime_type = "video/mp4"
            elif suffix == ".pdf": mime_type = "application/pdf"
        return cls(base64_data=encoded, mime_type=mime_type)

class ContentBlock(BaseModel):
    type: MediaType
    text: Optional[str] = None
    image_url: Optional[MediaResource] = None

class Message(BaseModel):
    role: Role
    # Pydantic v2 建议使用 Union 时尽量明确
    content: Union[str, List[ContentBlock]] = Field(default="")
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    # [关键修复] 移除 model_dump 重写，避免破坏 Pydantic 内部机制

    @classmethod
    def user(cls, text: str = None, images: List[str] = None) -> 'Message':
        """快捷构造工厂"""
        blocks = []
        if text:
            blocks.append(ContentBlock(type="text", text=text))
        if images:
            for img in images:
                if img.startswith("http") or img.startswith("data:"):
                    res = MediaResource(url=img)
                else:
                    res = MediaResource.from_file(img)
                blocks.append(ContentBlock(type="image_url", image_url=res))
        
        # 如果只有文本，简化存储
        if len(blocks) == 1 and blocks[0].type == "text":
            return cls(role="user", content=blocks[0].text)
            
        return cls(role="user", content=blocks)

    def to_api_payload(self) -> Dict[str, Any]:
        """
        转换为 OpenAI API 标准格式
        """
        # 显式构建字典，不依赖 model_dump 的复杂行为
        payload = {"role": self.role}
        
        # 处理 Content
        if isinstance(self.content, str):
            payload["content"] = self.content
        elif isinstance(self.content, list):
            formatted_content = []
            for block in self.content:
                if block.type == "text":
                    formatted_content.append({"type": "text", "text": block.text})
                elif block.type == "image_url" and block.image_url:
                    res = block.image_url
                    url_val = res.url
                    if not url_val and res.base64_data:
                        mime = res.mime_type or "image/jpeg"
                        url_val = f"data:{mime};base64,{res.base64_data}"
                    
                    formatted_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": url_val,
                            "detail": res.detail
                        }
                    })
            payload["content"] = formatted_content
        
        # 处理其他字段
        if self.name: payload["name"] = self.name
        if self.tool_calls: payload["tool_calls"] = self.tool_calls
        if self.tool_call_id: payload["tool_call_id"] = self.tool_call_id
        
        return payload