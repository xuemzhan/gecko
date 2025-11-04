# agno/media.py
"""
媒体对象统一模型模块

提供 Image、Audio、Video、File 四类媒体的标准化表示，
支持从 URL、本地文件路径或原始字节三种方式加载内容，
并提供 Base64 编码、字典序列化等通用操作。
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import httpx
from pydantic import BaseModel, field_validator, model_validator


# ---------- 媒体基类：封装共用逻辑 ----------
class _MediaBase(BaseModel):
    """
    所有媒体类型（图片、音频、视频）的抽象基类。
    强制要求：必须且只能提供 url / filepath / content 中的一个。
    """

    # === 核心内容源（三选一）===
    url: Optional[str] = None          # 远程 URL
    filepath: Optional[Union[Path, str]] = None  # 本地文件路径
    content: Optional[bytes] = None    # 原始字节数据

    # === 通用元数据 ===
    id: Optional[str] = None           # 唯一标识符（自动生成）
    format: Optional[str] = None       # 文件格式，如 'png', 'mp3'
    mime_type: Optional[str] = None    # MIME 类型，如 'image/png'

    @model_validator(mode="before")
    def validate_one_source(cls, data: Any) -> Any:
        """校验输入：确保恰好提供一个内容源，并自动生成 ID"""
        if isinstance(data, dict):
            # 检查三个内容源中非 None 的数量
            sources = [data.get(k) for k in ("url", "filepath", "content") if data.get(k) is not None]
            if len(sources) == 0:
                raise ValueError("必须提供 url、filepath 或 content 中的至少一个")
            if len(sources) > 1:
                raise ValueError("只能提供 url、filepath、content 中的一个，不能同时提供多个")
            # 自动生成唯一 ID
            if data.get("id") is None:
                data["id"] = str(uuid4())
        return data

    def get_content_bytes(self) -> Optional[bytes]:
        """获取媒体原始字节数据"""
        if self.content is not None:
            return self.content
        if self.url:
            # 从 URL 下载
            with httpx.Client() as client:
                resp = client.get(self.url, follow_redirects=True)
                resp.raise_for_status()
                return resp.content
        if self.filepath:
            # 从本地文件读取
            return Path(self.filepath).read_bytes()
        return None

    def to_base64(self) -> Optional[str]:
        """将内容转换为 Base64 字符串"""
        content = self.get_content_bytes()
        if content is None:
            return None
        return base64.b64encode(content).decode("utf-8")

    @classmethod
    def from_base64(
        cls,
        base64_content: str,
        id: Optional[str] = None,
        mime_type: Optional[str] = None,
        format: Optional[str] = None,
        **kwargs,
    ) -> _MediaBase:
        """
        从 Base64 字符串创建媒体对象。
        若解码失败，则将输入视为 UTF-8 字符串并编码为字节。
        """
        try:
            content_bytes = base64.b64decode(base64_content)
        except Exception:
            content_bytes = base64_content.encode("utf-8")
        return cls(
            content=content_bytes,
            id=id or str(uuid4()),
            mime_type=mime_type,
            format=format,
            **kwargs,
        )

    def to_dict(self, include_base64_content: bool = True) -> Dict[str, Any]:
        """转换为字典，可选是否包含 Base64 编码的内容"""
        result = {
            "id": self.id,
            "url": self.url,
            "filepath": str(self.filepath) if self.filepath else None,
            "format": self.format,
            "mime_type": self.mime_type,
        }
        if include_base64_content and self.content is not None:
            result["content"] = self.to_base64()
        # 移除值为 None 的字段
        return {k: v for k, v in result.items() if v is not None}


# ---------- 具体媒体类型 ----------

class Image(_MediaBase):
    """图片媒体类型，支持 OpenAI Vision 的 detail 字段等扩展属性"""

    # === 输入特有字段（如 OpenAI Vision）===
    detail: Optional[str] = None  # 图像理解精度：'low', 'high', 'auto'

    # === 输出特有字段（由模型生成）===
    original_prompt: Optional[str] = None  # 原始生成提示
    revised_prompt: Optional[str] = None   # 修正后的提示
    alt_text: Optional[str] = None         # 替代文本描述

    def to_dict(self, include_base64_content: bool = True) -> Dict[str, Any]:
        base = super().to_dict(include_base64_content)
        base.update({
            "detail": self.detail,
            "original_prompt": self.original_prompt,
            "revised_prompt": self.revised_prompt,
            "alt_text": self.alt_text,
        })
        return {k: v for k, v in base.items() if v is not None}


class Audio(_MediaBase):
    """音频媒体类型"""

    # === 音频特有元数据 ===
    duration: Optional[float] = None   # 时长（秒）
    sample_rate: int = 24000           # 采样率（Hz）
    channels: int = 1                  # 声道数

    # === 输出字段 ===
    transcript: Optional[str] = None   # 语音转文字结果
    expires_at: Optional[int] = None   # 临时 URL 过期时间戳

    def to_dict(self, include_base64_content: bool = True) -> Dict[str, Any]:
        base = super().to_dict(include_base64_content)
        base.update({
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "transcript": self.transcript,
            "expires_at": self.expires_at,
        })
        return {k: v for k, v in base.items() if v is not None}

    @classmethod
    def from_base64( # type: ignore
        cls,
        base64_content: str,
        id: Optional[str] = None,
        mime_type: Optional[str] = None,
        transcript: Optional[str] = None,
        expires_at: Optional[int] = None,
        sample_rate: int = 24000,
        channels: int = 1,
        **kwargs,
    ) -> Audio:
        try:
            content_bytes = base64.b64decode(base64_content)
        except Exception:
            content_bytes = base64_content.encode("utf-8")
        return cls(
            content=content_bytes,
            id=id or str(uuid4()),
            mime_type=mime_type,
            transcript=transcript,
            expires_at=expires_at,
            sample_rate=sample_rate,
            channels=channels,
            **kwargs,
        )


class Video(_MediaBase):
    """视频媒体类型"""

    # === 视频特有元数据 ===
    duration: Optional[float] = None   # 时长（秒）
    width: Optional[int] = None        # 宽度（像素）
    height: Optional[int] = None       # 高度（像素）
    fps: Optional[float] = None        # 帧率

    # === 输出字段 ===
    eta: Optional[str] = None          # 预估生成时间
    original_prompt: Optional[str] = None
    revised_prompt: Optional[str] = None

    def to_dict(self, include_base64_content: bool = True) -> Dict[str, Any]:
        base = super().to_dict(include_base64_content)
        base.update({
            "duration": self.duration,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "eta": self.eta,
            "original_prompt": self.original_prompt,
            "revised_prompt": self.revised_prompt,
        })
        return {k: v for k, v in base.items() if v is not None}


# ---------- 通用文件类型（支持文本/二进制）----------

# 支持的 MIME 类型白名单
_VALID_MIME_TYPES = {
    "application/pdf",
    "application/json",
    "application/x-javascript",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/javascript",
    "application/x-python",
    "text/x-python",
    "text/plain",
    "text/html",
    "text/css",
    "text/md",
    "text/csv",
    "text/xml",
    "text/rtf",
    "application/octet-stream",
}

class File(BaseModel):
    """
    通用文件类型，适用于非媒体类文档（如 PDF、TXT、CSV 等）。
    支持四种内容源：url / filepath / content / external。
    """

    id: Optional[str] = None
    url: Optional[str] = None
    filepath: Optional[Union[Path, str]] = None
    content: Optional[Any] = None      # 原始内容（bytes 或 str）
    mime_type: Optional[str] = None

    file_type: Optional[str] = None
    filename: Optional[str] = None
    size: Optional[int] = None
    external: Optional[Any] = None     # 外部文件对象（如 GeminiFile）
    format: Optional[str] = None       # 文件扩展名
    name: Optional[str] = None         # Bedrock 等平台所需字段

    @model_validator(mode="before")
    @classmethod
    def check_at_least_one_source(cls, data):
        """校验：至少提供一种内容源"""
        fields = ["url", "filepath", "content", "external"]
        if isinstance(data, dict) and not any(data.get(f) is not None for f in fields):
            raise ValueError("必须提供 url、filepath、content 或 external 中的至少一个")
        if data.get("id") is None:
            data["id"] = str(uuid4())
        return data

    @field_validator("mime_type")
    @classmethod
    def validate_mime_type(cls, v):
        """校验 MIME 类型是否在白名单中"""
        if v is not None and v not in _VALID_MIME_TYPES:
            raise ValueError(f"不支持的 MIME 类型: {v}。支持类型: {sorted(_VALID_MIME_TYPES)}")
        return v

    @classmethod
    def from_base64(
        cls,
        base64_content: str,
        id: Optional[str] = None,
        mime_type: Optional[str] = None,
        filename: Optional[str] = None,
        name: Optional[str] = None,
        format: Optional[str] = None,
    ) -> File:
        """从 Base64 创建文件对象"""
        content_bytes = base64.b64decode(base64_content)
        return cls(
            content=content_bytes,
            id=id or str(uuid4()),
            mime_type=mime_type,
            filename=filename,
            name=name,
            format=format,
        )

    def _normalize_content(self) -> Optional[str]:
        """
        将 content 标准化为字符串：
        - 文本类：解码为 UTF-8 字符串
        - 二进制类：Base64 编码
        - 其他：转为字符串
        """
        if self.content is None:
            return None
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, bytes):
            if self.mime_type and self.mime_type.startswith("text/"):
                try:
                    return self.content.decode("utf-8")
                except UnicodeDecodeError:
                    pass
            return base64.b64encode(self.content).decode("utf-8")
        return str(self.content)

    def to_dict(self) -> Dict[str, Any]:
        norm_content = self._normalize_content()
        result = {
            "id": self.id,
            "url": self.url,
            "filepath": str(self.filepath) if self.filepath else None,
            "content": norm_content,
            "mime_type": self.mime_type,
            "file_type": self.file_type,
            "filename": self.filename,
            "size": self.size,
            "external": self.external,
            "format": self.format,
            "name": self.name,
        }
        return {k: v for k, v in result.items() if v is not None}


# ---------- 测试代码 ----------
if __name__ == "__main__":
    import tempfile
    import os

    print("=== 开始测试 agno/media.py ===\n")

    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. 测试 Image
        print("[1] 测试 Image")
        img = Image(content=b"fake image bytes", format="png", detail="high")
        print("  ID:", img.id)
        print("  Base64 长度:", len(img.to_base64() or ""))
        print("  Dict:", img.to_dict(include_base64_content=False))

        # 2. 测试 Audio
        print("\n[2] 测试 Audio")
        audio = Audio.from_base64("SGVsbG8gQXVkaW8h", transcript="Hello Audio!")
        print("  Transcript:", audio.transcript)
        print("  Sample rate:", audio.sample_rate)

        # 3. 测试 File（文本）
        print("\n[3] 测试 File")
        txt_file = File(
            content=b"Hello, File!",
            mime_type="text/plain",
            filename="test.txt"
        )
        print("  Normalized content:", txt_file._normalize_content())

        # 4. 测试 File（二进制）
        bin_file = File(
            content=b"\x00\x01\x02",
            mime_type="application/octet-stream",
            filename="data.bin"
        )
        norm = bin_file._normalize_content()
        print("  Binary normalized (Base64):", norm)
        print("  Decode back:", base64.b64decode(norm)) # pyright: ignore[reportArgumentType]

        # 5. 测试错误情况
        print("\n[4] 测试错误输入")
        try:
            Image(url="http://example.com", filepath="local.jpg")  # 两个源
        except ValueError as e:
            print("  多源错误:", e)

        try:
            Image()  # 无源
        except ValueError as e:
            print("  无源错误:", e)

        try:
            File(mime_type="application/unknown")
        except ValueError as e:
            print("  MIME 类型错误:", e)

    print("\n=== 测试结束 ===")