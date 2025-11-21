# gecko/core/message.py  
"""  
消息模型（增强版）  
  
改进：  
1. MediaResource.from_file 支持文件大小限制与 MIME 自动推断  
2. Message.tool_result 允许 content 为 dict/list，会在序列化时自动转换  
3. 增加 from_openai classmethod，便于解析模型返回  
4. ContentBlock 在初始化阶段校验 image_url 是否存在  
"""  
  
from __future__ import annotations  
  
import base64  
import mimetypes  
from pathlib import Path  
from typing import Any, Dict, List, Literal, Optional, Union  
  
from pydantic import BaseModel, Field, field_serializer, model_validator  
  
Role = Literal["system", "user", "assistant", "tool"]  
  
  
class MediaResource(BaseModel):  
    url: Optional[str] = None  
    base64_data: Optional[str] = None  
    mime_type: Optional[str] = None  
    detail: Literal["auto", "low", "high"] = "auto"  
  
    @model_validator(mode="after")  
    def validate_source(self):  
        if not self.url and not self.base64_data:  
            raise ValueError("必须提供 url 或 base64_data")  
        return self  
  
    @classmethod  
    def from_file(cls, path: str, mime_type: Optional[str] = None, max_size_mb: int = 5) -> MediaResource:  
        p = Path(path)  
        if not p.exists():  
            raise FileNotFoundError(f"File not found: {path}")  
        if p.stat().st_size > max_size_mb * 1024 * 1024:  
            raise ValueError(f"文件过大，超过 {max_size_mb} MB")  
  
        with open(p, "rb") as f:  
            encoded = base64.b64encode(f.read()).decode("utf-8")  
  
        mime = mime_type or mimetypes.guess_type(p.name)[0] or "application/octet-stream"  
        return cls(base64_data=encoded, mime_type=mime)  
  
    def to_openai_image_url(self) -> Dict[str, Any]:  
        url_value = self.url or f"data:{self.mime_type or 'image/jpeg'};base64,{self.base64_data}"  
        return {"url": url_value, "detail": self.detail}  
  
  
class ContentBlock(BaseModel):  
    type: Literal["text", "image_url"]  
    text: Optional[str] = None  
    image_url: Optional[MediaResource] = None  
  
    @model_validator(mode="after")  
    def ensure_valid(self):  
        if self.type == "text" and not self.text:  
            raise ValueError("文本块缺少 text")  
        if self.type == "image_url" and not self.image_url:  
            raise ValueError("图片块缺少 image_url")  
        return self  
  
    def to_openai_format(self) -> Dict[str, Any]:  
        if self.type == "text":  
            return {"type": "text", "text": self.text}  
        return {"type": "image_url", "image_url": self.image_url.to_openai_image_url()}  
  
  
class Message(BaseModel):  
    role: Role  
    content: Union[str, List[ContentBlock]] = Field(default="")  
    name: Optional[str] = None  
    tool_calls: Optional[List[Dict[str, Any]]] = None  
    tool_call_id: Optional[str] = None  
  
    @field_serializer("content")  
    def serialize_content(self, content: Union[str, List[ContentBlock]], _info):  
        if isinstance(content, str):  
            return content  
        return [block.to_openai_format() for block in content]  
  
    @classmethod  
    def user(cls, text: str = "", images: Optional[List[str]] = None) -> Message:  
        if not images:  
            return cls(role="user", content=text)  
  
        blocks: List[ContentBlock] = []  
        if text:  
            blocks.append(ContentBlock(type="text", text=text))  
  
        for img in images:  
            if img.startswith(("http://", "https://", "data:")):  
                resource = MediaResource(url=img)  
            else:  
                resource = MediaResource.from_file(img)  
            blocks.append(ContentBlock(type="image_url", image_url=resource))  
  
        return cls(role="user", content=blocks)  
  
    @classmethod  
    def assistant(cls, content: str) -> Message:  
        return cls(role="assistant", content=content)  
  
    @classmethod  
    def system(cls, content: str) -> Message:  
        return cls(role="system", content=content)  
  
    @classmethod  
    def tool_result(cls, tool_call_id: str, content: Any, tool_name: str) -> Message:  
        if isinstance(content, (dict, list)):  
            import json  
            serialized = json.dumps(content, ensure_ascii=False)  
        else:  
            serialized = str(content)  
        return cls(role="tool", content=serialized, tool_call_id=tool_call_id, name=tool_name)  
  
    @classmethod  
    def from_openai(cls, payload: Dict[str, Any]) -> Message:  
        return cls(**payload)  
  
    def to_openai_format(self) -> Dict[str, Any]:  
        return self.model_dump(exclude_none=True, mode="json")  
