# agno/models/message.py

"""
消息模型模块

定义了与大模型交互的核心消息结构，支持多模态内容（文本、图像、音频、视频、文件），
并包含引用、工具调用、指标、日志等扩展字段。
"""

import base64
import json
from time import time
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

# 尝试导入真实模块；若失败则使用简化版（用于独立测试）
try:
    from media import Audio, File, Image, Video  # type: ignore
    from models.metrics import Metrics
    from utils.log import log_debug, log_error, log_info, log_warning  # type: ignore
except ImportError:
    # --- 简化版依赖（仅用于测试）---
    from pathlib import Path
    from typing import Any
    from uuid import uuid4

    class MockBaseModel(BaseModel):
        pass

    class Image(MockBaseModel):
        url: Optional[str] = None
        filepath: Optional[Union[Path, str]] = None
        content: Optional[bytes] = None
        id: Optional[str] = None
        format: Optional[str] = None
        mime_type: Optional[str] = None
        detail: Optional[str] = None

        @classmethod
        def from_base64(cls, b64: str, **kwargs) -> "Image":
            return cls(content=base64.b64decode(b64), **kwargs)

        def to_dict(self, include_base64_content: bool = True) -> dict:
            d = {k: v for k, v in self.model_dump().items() if v is not None}
            if include_base64_content and self.content is not None:
                d["content"] = base64.b64encode(self.content).decode("utf-8")
            return d

    Audio = Video = File = Image

    class Metrics(MockBaseModel):
        input_tokens: Optional[int] = None
        output_tokens: Optional[int] = None
        total_tokens: Optional[int] = None
        duration: Optional[float] = None
        time_to_first_token: Optional[float] = None
        reasoning_tokens: Optional[int] = None
        audio_total_tokens: Optional[int] = None
        cache_read_tokens: Optional[int] = None
        cache_write_tokens: Optional[int] = None
        provider_metrics: Optional[Dict[str, Any]] = None
        additional_metrics: Optional[Dict[str, Any]] = None

        def to_dict(self) -> dict:
            return {k: v for k, v in self.model_dump().items() if v is not None}

        def __eq__(self, other):
            return isinstance(other, Metrics) and self.model_dump() == other.model_dump()

    def log_debug(msg, **kwargs):
        if kwargs.get("center"):
            print(f"=== {msg} ===")
        else:
            print(f"[DEBUG] {msg}")

    log_info = log_warning = log_error = log_debug


class MessageReferences(BaseModel):
    """用户消息中附加的引用信息（用于 RAG）"""
    query: str  # 检索所用的查询语句
    references: Optional[List[Union[Dict[str, Any], str]]] = None  # 引用内容
    time: Optional[float] = None  # 检索耗时（秒）


class UrlCitation(BaseModel):
    """URL 引用"""
    url: Optional[str] = None
    title: Optional[str] = None


class DocumentCitation(BaseModel):
    """文档引用"""
    document_title: Optional[str] = None
    cited_text: Optional[str] = None
    file_name: Optional[str] = None


class Citations(BaseModel):
    """消息中的引用信息"""
    raw: Optional[Any] = None
    urls: Optional[List[UrlCitation]] = None
    documents: Optional[List[DocumentCitation]] = None


class Message(BaseModel):
    """与大模型交互的消息对象"""

    # === 基础字段 ===
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: str  # 角色：system/user/assistant/tool
    content: Optional[Union[List[Any], str]] = None
    name: Optional[str] = None  # 用于区分同角色的不同参与者
    created_at: int = Field(default_factory=lambda: int(time()))

    # === 工具调用相关 ===
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Any] = None
    tool_call_error: Optional[bool] = None
    stop_after_tool_call: bool = False

    # === 多模态输入 ===
    audio: Optional[Sequence[Audio]] = None # type: ignore
    images: Optional[Sequence[Image]] = None # type: ignore
    videos: Optional[Sequence[Video]] = None # type: ignore
    files: Optional[Sequence[File]] = None # type: ignore

    # === 多模态输出 ===
    audio_output: Optional[Audio] = None # type: ignore
    image_output: Optional[Image] = None # type: ignore
    video_output: Optional[Video] = None # type: ignore
    file_output: Optional[File] = None # type: ignore

    # === 模型输出扩展 ===
    reasoning_content: Optional[str] = None  # 模型推理内容（不发送给 API）
    redacted_reasoning_content: Optional[str] = None  # 脱敏后的推理内容（发送给 API）
    provider_data: Optional[Dict[str, Any]] = None  # 提供商特定数据 # type: ignore
    citations: Optional[Citations] = None  # 引用信息

    # === 元数据与控制 ===
    from_history: bool = False  # 是否来自历史记忆
    add_to_agent_memory: bool = True  # 是否加入代理记忆
    metrics: Metrics = Field(default_factory=Metrics)
    references: Optional[MessageReferences] = None

    model_config = ConfigDict(extra="allow", populate_by_name=True, arbitrary_types_allowed=True)

    def get_content_string(self) -> str:
        """将 content 转换为字符串形式"""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, list) and self.content:
            if isinstance(self.content[0], dict) and "text" in self.content[0]:
                return self.content[0].get("text", "")
            return json.dumps(self.content, ensure_ascii=False)
        return ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """从字典重建 Message 对象"""
        # 重建输入媒体
        for field, media_cls in [("images", Image), ("audio", Audio), ("videos", Video), ("files", File)]:
            if field in data and data[field]:
                reconstructed = []
                for item in data[field]:
                    if isinstance(item, dict):
                        if "content" in item and isinstance(item["content"], str):
                            kwargs = {k: v for k, v in item.items() if k != "content"}
                            reconstructed.append(media_cls.from_base64(item["content"], **kwargs))
                        else:
                            reconstructed.append(media_cls(**item))
                    else:
                        reconstructed.append(item)
                data[field] = reconstructed

        # 重建输出媒体
        for field, media_cls in [("image_output", Image), ("audio_output", Audio), ("video_output", Video), ("file_output", File)]:
            if field in data and data[field]:
                item = data[field]
                if isinstance(item, dict):
                    if "content" in item and isinstance(item["content"], str):
                        kwargs = {k: v for k, v in item.items() if k != "content"}
                        data[field] = media_cls.from_base64(item["content"], **kwargs)
                    else:
                        data[field] = media_cls(**item)

        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，确保所有内容可 JSON 序列化"""
        def _make_json_safe(obj):
            if isinstance(obj, bytes):
                return base64.b64encode(obj).decode("utf-8")
            elif isinstance(obj, dict):
                return {k: _make_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_make_json_safe(v) for v in obj]
            else:
                return obj

        base_fields = {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "name": self.name,
            "tool_call_id": self.tool_call_id,
            "tool_calls": self.tool_calls,
            "reasoning_content": self.reasoning_content,
            "redacted_reasoning_content": self.redacted_reasoning_content,
            "provider_data": self.provider_data, # type: ignore
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tool_call_error": self.tool_call_error,
            "stop_after_tool_call": self.stop_after_tool_call,
            "from_history": self.from_history,
            "created_at": self.created_at,
        }
        result = {k: v for k, v in base_fields.items() if v is not None and not (isinstance(v, (list, dict)) and len(v) == 0)}

        if self.images:
            result["images"] = [img.to_dict(include_base64_content=True) for img in self.images]
        if self.audio:
            result["audio"] = [aud.to_dict(include_base64_content=True) for aud in self.audio]
        if self.videos:
            result["videos"] = [vid.to_dict(include_base64_content=True) for vid in self.videos]
        if self.files:
            result["files"] = [f.to_dict() for f in self.files]
        if self.audio_output:
            result["audio_output"] = self.audio_output.to_dict(include_base64_content=True)
        if self.image_output:
            result["image_output"] = self.image_output.to_dict(include_base64_content=True)
        if self.video_output:
            result["video_output"] = self.video_output.to_dict(include_base64_content=True)
        if self.file_output:
            result["file_output"] = self.file_output.to_dict()

        if self.references:
            result["references"] = self.references.model_dump()
        if self.metrics and self.metrics != Metrics():
            result["metrics"] = self.metrics.to_dict()

        return _make_json_safe(result) # type: ignore

    def to_function_call_dict(self) -> Dict[str, Any]:
        """返回工具调用相关的字典"""
        return {
            "content": self.content,
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "tool_args": self.tool_args,
            "tool_call_error": self.tool_call_error,
            "metrics": self.metrics,
            "created_at": self.created_at,
        }

    def log(self, metrics: bool = True, level: Optional[str] = None):
        """将消息内容打印到控制台"""
        _logger = log_debug
        if level == "info":
            _logger = log_info
        elif level == "warning":
            _logger = log_warning
        elif level == "error":
            _logger = log_error

        try:
            import shutil
            width = shutil.get_terminal_size().columns
        except Exception:
            width = 80

        header = f" {self.role.upper()} "
        _logger(header.center(width - 20, "="))

        if self.name:
            _logger(f"Name: {self.name}")
        if self.tool_call_id:
            _logger(f"Tool call ID: {self.tool_call_id}")
        if self.reasoning_content:
            _logger(f"<reasoning>\n{self.reasoning_content}\n</reasoning>")
        if self.content:
            _logger(self.content if isinstance(self.content, (str, list)) else json.dumps(self.content, indent=2))
        if self.tool_calls:
            lines = ["Tool Calls:"]
            for tc in self.tool_calls:
                func = tc.get("function", {})
                lines.append(f"  - ID: {tc.get('id', 'N/A')}, Name: {func.get('name', 'N/A')}")
                args = func.get("arguments")
                if args:
                    try:
                        parsed = args if isinstance(args, dict) else json.loads(args)
                        lines.append(f"    Args: {parsed}")
                    except Exception:
                        lines.append(f"    Args: <invalid JSON>")
            _logger("\n".join(lines))

        if self.images:
            _logger(f"Images added: {len(self.images)}")
        if self.videos:
            _logger(f"Videos added: {len(self.videos)}")
        if self.audio:
            _logger(f"Audio files added: {len(self.audio)}")
        if self.files:
            _logger(f"Files added: {len(self.files)}")

        if metrics and self.metrics != Metrics():
            header = f" {'TOOL' if self.role == 'tool' else ''} METRICS ".strip()
            _logger(header, center=True, symbol="*") # type: ignore
            m = self.metrics
            tokens = []
            for attr in ["input_tokens", "output_tokens", "total_tokens", "reasoning_tokens", "audio_total_tokens"]:
                v = getattr(m, attr, None)
                if v:
                    tokens.append(f"{attr.split('_')[0]}={v}")
            if tokens:
                _logger(f"* Tokens: {', '.join(tokens)}")
            if m.duration:
                _logger(f"* Duration: {m.duration:.4f}s")
                if m.output_tokens:
                    _logger(f"* Tokens/sec: {m.output_tokens / m.duration:.2f}")
            if m.time_to_first_token:
                _logger(f"* Time to first token: {m.time_to_first_token:.4f}s")
            if m.provider_metrics:
                _logger(f"* Provider metrics: {m.provider_metrics}")
            if m.additional_metrics:
                _logger(f"* Additional metrics: {m.additional_metrics}")
            _logger(header, center=True, symbol="*") # type: ignore

    def content_is_valid(self) -> bool:
        """检查消息内容是否有效（非空）"""
        if self.content is None:
            return False
        if isinstance(self.content, str):
            return len(self.content.strip()) > 0
        if isinstance(self.content, (list, dict)):
            return len(self.content) > 0
        return True


# --- 测试代码 ---
if __name__ == "__main__":
    # 创建测试消息
    msg = Message(
        role="user",
        content="Hello, world!",
        images=[Image(content=b"fake image content", format="png")],
        references=MessageReferences(query="test query", references=["ref1"]),
        metrics=Metrics(input_tokens=10, output_tokens=20, duration=0.5)
    )

    print("=== 原始消息 ===")
    print(f"Content: {msg.get_content_string()}")
    print(f"Images: {len(msg.images) if msg.images else 0}")

    # 序列化（关键测试点）
    msg_dict = msg.to_dict()
    print("\n=== 序列化字典（JSON 安全）===")
    print(json.dumps(msg_dict, indent=2, ensure_ascii=False))

    # 反序列化
    msg2 = Message.from_dict(msg_dict)
    print("\n=== 反序列化验证 ===")
    print(f"Content same? {msg.content == msg2.content}")
    print(f"Images same? {len(msg.images) == len(msg2.images)}") # type: ignore

    # 日志输出
    print("\n=== 日志输出 ===")
    msg.log(level="info")

    print("\n✅ 测试完成")