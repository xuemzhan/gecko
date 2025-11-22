"""多模态资源定义"""
from __future__ import annotations
import asyncio
import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, model_validator
from gecko.core.logging import get_logger

logger = get_logger(__name__)

class MediaResource(BaseModel):
    """
    媒体资源（主要用于图片）
    
    支持：
    - URL（http/https）
    - Base64 编码的数据
    - 本地文件路径（通过工厂方法）
    
    示例:
        ```python
        # 从 URL
        img = MediaResource(url="https://example.com/image.jpg")
        
        # 从本地文件（同步）
        img = MediaResource.from_file("./image.png")
        
        # 从本地文件（异步）
        img = await MediaResource.from_file_async("./large_image.png")
        
        # 从 base64
        img = MediaResource(
            base64_data="iVBORw0KG...",
            mime_type="image/png"
        )
        ```
    """
    url: Optional[str] = None
    base64_data: Optional[str] = None
    mime_type: Optional[str] = None
    detail: Literal["auto", "low", "high"] = "auto"

    @model_validator(mode="after")
    def validate_source(self):
        """验证至少提供了一个数据源"""
        if not self.url and not self.base64_data:
            raise ValueError("必须提供 url 或 base64_data")
        return self

    @classmethod
    def from_file(
        cls,
        path: str,
        mime_type: Optional[str] = None,
        max_size_mb: int = 5,
        detail: Literal["auto", "low", "high"] = "auto"
    ) -> MediaResource:
        """
        从本地文件加载（同步版本）
        
        参数:
            path: 文件路径
            mime_type: MIME 类型（None 则自动推断）
            max_size_mb: 最大文件大小（MB）
            detail: 图片质量（OpenAI API 参数）
        
        返回:
            MediaResource 实例
        
        异常:
            FileNotFoundError: 文件不存在
            ValueError: 文件过大
        
        注意:
            这是同步方法，会阻塞事件循环。
            对于大文件，建议使用 from_file_async()
        """
        p = Path(path)
        
        # 检查文件是否存在
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        
        if not p.is_file():
            raise ValueError(f"路径不是文件: {path}")
        
        # ✅ 优化：先检查文件大小，再读取
        file_size = p.stat().st_size
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if file_size > max_size_bytes:
            raise ValueError(
                f"文件过大: {file_size / 1024 / 1024:.2f} MB "
                f"(最大 {max_size_mb} MB)"
            )
        
        # 读取并编码
        try:
            with open(p, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            raise IOError(f"文件读取失败: {e}") from e
        
        # 推断 MIME 类型
        mime = mime_type or mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        
        logger.debug(
            "Media loaded from file",
            path=path,
            size_kb=file_size / 1024,
            mime_type=mime
        )
        
        return cls(
            base64_data=encoded,
            mime_type=mime,
            detail=detail
        )

    @classmethod
    async def from_file_async(
        cls,
        path: str,
        mime_type: Optional[str] = None,
        max_size_mb: int = 5,
        detail: Literal["auto", "low", "high"] = "auto"
    ) -> MediaResource:
        """
        从本地文件加载（异步版本）
        
        对于大文件，使用此方法避免阻塞事件循环
        
        参数:
            同 from_file()
        
        返回:
            MediaResource 实例
        """
        p = Path(path)
        
        # 检查文件
        if not p.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        
        if not p.is_file():
            raise ValueError(f"路径不是文件: {path}")
        
        # 检查大小
        file_size = p.stat().st_size
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if file_size > max_size_bytes:
            raise ValueError(
                f"文件过大: {file_size / 1024 / 1024:.2f} MB "
                f"(最大 {max_size_mb} MB)"
            )
        
        # ✅ 异步读取文件（在线程池中执行）
        def _read_file():
            with open(p, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        
        try:
            encoded = await asyncio.to_thread(_read_file)
        except Exception as e:
            raise IOError(f"文件读取失败: {e}") from e
        
        # 推断 MIME 类型
        mime = mime_type or mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        
        logger.debug(
            "Media loaded from file (async)",
            path=path,
            size_kb=file_size / 1024,
            mime_type=mime
        )
        
        return cls(
            base64_data=encoded,
            mime_type=mime,
            detail=detail
        )

    def to_openai_image_url(self) -> Dict[str, Any]:
        """
        转换为 OpenAI API 所需的 image_url 格式
        
        返回:
            符合 OpenAI 规范的字典
        """
        # 构建 URL
        if self.url:
            url_value = self.url
        elif self.base64_data:
            mime = self.mime_type or "image/jpeg"
            url_value = f"data:{mime};base64,{self.base64_data}"
        else:
            raise ValueError("MediaResource 缺少 URL 或 base64_data")
        
        return {
            "url": url_value,
            "detail": self.detail
        }

    def get_size_estimate(self) -> int:
        """
        估算数据大小（字节）
        
        返回:
            估算的字节数
        """
        if self.base64_data:
            # Base64 编码后的大小约为原始大小的 4/3
            return int(len(self.base64_data) * 3 / 4)
        elif self.url:
            # URL 无法估算实际大小
            return 0
        return 0


# ===== 内容块 =====

class ContentBlock(BaseModel):
    """
    消息内容块（用于多模态消息）
    
    支持：
    - 文本块
    - 图片块
    
    示例:
        ```python
        # 文本块
        text = ContentBlock(type="text", text="Hello")
        
        # 图片块
        image = ContentBlock(
            type="image_url",
            image_url=MediaResource(url="https://...")
        )
        ```
    """
    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[MediaResource] = None

    @model_validator(mode="after")
    def ensure_valid(self):
        """验证块的完整性"""
        if self.type == "text":
            if self.text is None:
                raise ValueError("文本块缺少 text 字段")
        elif self.type == "image_url":
            if self.image_url is None:
                raise ValueError("图片块缺少 image_url 字段")
        return self

    def to_openai_format(self) -> Dict[str, Any]:
        """转换为 OpenAI API 格式"""
        if self.type == "text":
            return {"type": "text", "text": self.text}
        elif self.type == "image_url":
            return {
                "type": "image_url",
                "image_url": self.image_url.to_openai_image_url() # type: ignore
            }
        else:
            raise ValueError(f"未知的内容类型: {self.type}")

    def get_text_content(self) -> str:
        """
        提取文本内容（用于调试/日志）
        
        返回:
            文本内容或占位符
        """
        if self.type == "text":
            return self.text or ""
        elif self.type == "image_url":
            return "[image]"
        return ""


# ===== 消息 =====

