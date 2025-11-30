"""消息主体定义"""
from __future__ import annotations
import asyncio
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_serializer, field_validator
from gecko.core.logging import get_logger
from gecko.core.message.resources import MediaResource, ContentBlock

logger = get_logger(__name__)
Role = Literal["system", "user", "assistant", "tool"]

class Message(BaseModel):
    """
    标准消息对象
    
    兼容 OpenAI Chat Completion API 格式
    
    示例:
        ```python
        # 简单文本消息
        msg = Message.user("Hello!")
        
        # 多模态消息
        msg = Message.user(
            text="What's in this image?",
            images=["./photo.jpg"]
        )
        
        # 助手消息
        msg = Message.assistant("I'm here to help!")
        
        # 工具返回消息
        msg = Message.tool_result(
            tool_call_id="call_123",
            content="Search results: ...",
            tool_name="search"
        )
        
        # 从 OpenAI 格式解析
        msg = Message.from_openai({
            "role": "user",
            "content": "Hello"
        })
        ```
    """
    role: Role
    content: Union[str, List[ContentBlock]] = Field(default="")
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    @field_validator("content", mode="before")
    @classmethod
    def validate_content(cls, v):
        """验证并规范化 content"""
        if v is None:
            return ""
        return v

    @field_serializer("content")
    def serialize_content(self, content: Union[str, List[ContentBlock]], _info):
        """序列化 content 为 OpenAI 格式"""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            return [block.to_openai_format() for block in content]
        return str(content)

    # ===== 工厂方法 =====

    @classmethod
    def user(
        cls,
        text: str = "",
        images: Optional[List[str]] = None,
        name: Optional[str] = None
    ) -> Message:
        """
        创建用户消息
        
        参数:
            text: 文本内容
            images: 图片路径列表（URL 或本地文件）
            name: 用户名称（可选）
        
        返回:
            Message 实例
        
        示例:
            ```python
            # 纯文本
            msg = Message.user("Hello")
            
            # 文本 + 图片
            msg = Message.user(
                text="What's this?",
                images=["./photo.jpg", "https://example.com/img.png"]
            )
            ```
        """
        if not images:
            return cls(role="user", content=text, name=name)
        
        # 构建多模态内容
        blocks: List[ContentBlock] = []
        
        # 添加文本块
        if text:
            blocks.append(ContentBlock(type="text", text=text))
        
        # 添加图片块
        for img in images:
            try:
                # 判断是 URL 还是本地路径
                if img.startswith(("http://", "https://", "data:")):
                    resource = MediaResource(url=img)
                else:
                    resource = MediaResource.from_file(img)
                
                blocks.append(ContentBlock(type="image_url", image_url=resource))
            except Exception as e:
                logger.error("Failed to load image", path=img, error=str(e))
                # 继续处理其他图片
        
        return cls(role="user", content=blocks, name=name)

    @classmethod
    async def user_async(
        cls,
        text: str = "",
        images: Optional[List[str]] = None,
        name: Optional[str] = None
    ) -> Message:
        """
        创建用户消息（异步版本）
        
        对于大量或大文件图片，使用此方法避免阻塞
        """
        if not images:
            return cls(role="user", content=text, name=name)
        
        blocks: List[ContentBlock] = []
        
        if text:
            blocks.append(ContentBlock(type="text", text=text))
        
        # 异步加载所有图片
        async def _load_image(img: str) -> Optional[ContentBlock]:
            try:
                if img.startswith(("http://", "https://", "data:")):
                    resource = MediaResource(url=img)
                else:
                    resource = await MediaResource.from_file_async(img)
                return ContentBlock(type="image_url", image_url=resource)
            except Exception as e:
                logger.error("Failed to load image (async)", path=img, error=str(e))
                return None
        
        # 并发加载所有图片
        image_blocks = await asyncio.gather(*[_load_image(img) for img in images])
        blocks.extend([b for b in image_blocks if b is not None])
        
        return cls(role="user", content=blocks, name=name)

    @classmethod
    def assistant(
        cls, 
        content: str = "", 
        tool_calls: Optional[List[Dict[str, Any]]] = None, # 新增参数
        name: Optional[str] = None
    ) -> Message:
        """
        创建助手消息
        
        参数:
            content: 回复内容
            tool_calls: 工具调用列表
            name: 助手名称（可选）
        """
        return cls(
            role="assistant", 
            content=content, 
            tool_calls=tool_calls, 
            name=name
        )
    
    @property
    def safe_tool_calls(self) -> List[Dict[str, Any]]:
        """
        安全获取 tool_calls，确保返回列表而不是 None。
        用于减少 if msg.tool_calls is not None 的检查。
        """
        return self.tool_calls or []

    @classmethod
    def system(cls, content: str) -> Message:
        """
        创建系统消息
        
        参数:
            content: 系统提示词
        """
        return cls(role="system", content=content)

    @classmethod
    def tool_result(
        cls,
        tool_call_id: str,
        content: Any,
        tool_name: str
    ) -> Message:
        """
        创建工具返回消息
        
        参数:
            tool_call_id: 工具调用 ID
            content: 工具返回结果（任意类型，会自动序列化）
            tool_name: 工具名称
        
        返回:
            Message 实例
        """
        # 序列化 content
        if isinstance(content, str):
            serialized = content
        elif isinstance(content, (dict, list)):
            import json
            serialized = json.dumps(content, ensure_ascii=False, indent=2)
        else:
            serialized = str(content)
        
        return cls(
            role="tool",
            content=serialized,
            tool_call_id=tool_call_id,
            name=tool_name
        )

    @classmethod
    def from_openai(cls, payload: Dict[str, Any]) -> Message:
        """
        从 OpenAI API 格式解析消息
        
        参数:
            payload: OpenAI 格式的消息字典
        
        返回:
            Message 实例
        
        示例:
            ```python
            openai_msg = {
                "role": "assistant",
                "content": "Hello!",
                "tool_calls": [...]
            }
            msg = Message.from_openai(openai_msg)
            ```
        """
        try:
            return cls(**payload)
        except Exception as e:
            logger.error("Failed to parse OpenAI message", error=str(e), payload=payload)
            raise ValueError(f"无效的 OpenAI 消息格式: {e}") from e

    # ===== 转换方法 =====

    def to_openai_format(self) -> Dict[str, Any]:
        """
        转换为 OpenAI API 格式
        
        返回:
            符合 OpenAI 规范的字典
        """
        # 使用 Pydantic 的序列化（会调用 field_serializer）
        data = self.model_dump(exclude_none=True, mode="json")
        
        # 确保必要字段存在
        if "role" not in data:
            raise ValueError("消息缺少 role 字段")
        
        return data

    # ===== 工具方法 =====

    def get_text_content(self) -> str:
        """
        提取文本内容（忽略多模态部分）
        
        返回:
            纯文本内容
        
        用途:
            - 日志记录
            - 文本搜索
            - Token 估算
        """
        if isinstance(self.content, str):
            return self.content
        elif isinstance(self.content, list):
            text_parts = []
            for block in self.content:
                text = block.get_text_content()
                if text:
                    text_parts.append(text)
            return " ".join(text_parts)
        return ""

    def is_empty(self) -> bool:
        """
        检查消息是否为空
        
        返回:
            是否为空消息
        """
        if isinstance(self.content, str):
            return not self.content.strip()
        elif isinstance(self.content, list):
            return len(self.content) == 0
        return True

    def has_images(self) -> bool:
        """
        检查消息是否包含图片
        
        返回:
            是否包含图片
        """
        if isinstance(self.content, list):
            return any(block.type == "image_url" for block in self.content)
        return False

    def get_image_count(self) -> int:
        """
        获取图片数量
        
        返回:
            图片数量
        """
        if isinstance(self.content, list):
            return sum(1 for block in self.content if block.type == "image_url")
        return 0

    def clone(self) -> Message:
        """
        创建消息的深拷贝
        
        返回:
            新的 Message 实例
        """
        return Message.model_validate(self.model_dump())

    def truncate_content(self, max_length: int) -> Message:
        """
        截断消息内容（返回新消息）
        
        参数:
            max_length: 最大字符长度
        
        返回:
            截断后的新消息
        
        注意:
            仅截断文本内容，图片保持不变
        """
        if isinstance(self.content, str):
            if len(self.content) > max_length:
                truncated = self.content[:max_length] + "..."
                return Message(
                    role=self.role,
                    content=truncated,
                    name=self.name,
                    tool_calls=self.tool_calls,
                    tool_call_id=self.tool_call_id
                )
        
        # 多模态消息：截断文本块
        elif isinstance(self.content, list):
            new_blocks = []
            for block in self.content:
                if block.type == "text" and block.text:
                    if len(block.text) > max_length:
                        new_blocks.append(ContentBlock(
                            type="text",
                            text=block.text[:max_length] + "..."
                        ))
                    else:
                        new_blocks.append(block)
                else:
                    new_blocks.append(block)
            
            return Message(
                role=self.role,
                content=new_blocks,
                name=self.name,
                tool_calls=self.tool_calls,
                tool_call_id=self.tool_call_id
            )
        
        return self

    def __str__(self) -> str:
        """字符串表示（用于调试）"""
        text = self.get_text_content()
        preview = text[:50] + "..." if len(text) > 50 else text
        
        extra = []
        if self.has_images():
            extra.append(f"{self.get_image_count()} images")
        if self.tool_calls:
            extra.append(f"{len(self.tool_calls)} tool_calls")
        
        extra_str = f" ({', '.join(extra)})" if extra else ""
        
        return f"Message(role={self.role}, content='{preview}'{extra_str})"

    def __repr__(self) -> str:
        """详细表示"""
        return (
            f"Message("
            f"role={self.role!r}, "
            f"content={self.get_text_content()[:30]!r}, "
            f"has_images={self.has_images()}, "
            f"tool_calls={'Yes' if self.tool_calls else 'No'}"
            f")"
        )