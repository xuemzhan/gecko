# agno/utils/media_utils.py

"""
媒体处理工具模块

- 提供统一接口：支持 bytes / filepath / URL
- 自动 MIME 类型推断
- 安全的 Base64 Data URI 生成
- 与 `agno.media` 无缝集成
"""

import base64
import mimetypes
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import httpx
import logging

logger = logging.getLogger(__name__)

# 增强 MIME 类型支持
_MIME_OVERRIDES = {
    ".webp": "image/webp",
    ".ico": "image/x-icon",
    ".svg": "image/svg+xml",
    ".mp3": "audio/mpeg",
    ".wav": "audio/wav",
    ".mp4": "video/mp4",
    ".mov": "video/quicktime",
    ".bin": "application/octet-stream",
}
for ext, mime in _MIME_OVERRIDES.items():
    mimetypes.add_type(mime, ext)


class MediaProcessor:
    """集中式媒体处理工具"""

    @staticmethod
    def _is_valid_url(url: str) -> bool:
        try:
            parsed = urlparse(url.strip())
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False

    @staticmethod
    def get_content_bytes_from_url(url: str, timeout: int = 10) -> Optional[bytes]:
        clean_url = url.strip()
        if not MediaProcessor._is_valid_url(clean_url):
            logger.error(f"Invalid URL: {url}")
            return None
        try:
            with httpx.Client() as client:
                resp = client.get(clean_url, timeout=timeout, follow_redirects=True)
                resp.raise_for_status()
                return resp.content
        except Exception as e:
            logger.error(f"Failed to fetch {clean_url}: {e}")
            return None

    @staticmethod
    def get_content_bytes(
        content: Optional[bytes] = None,
        filepath: Optional[Union[str, Path]] = None,
        url: Optional[str] = None,
    ) -> Optional[bytes]:
        if content is not None:
            return content
        if filepath is not None:
            try:
                return Path(filepath).read_bytes()
            except OSError as e:
                logger.error(f"Failed to read file {filepath}: {e}")
                return None
        if url is not None:
            return MediaProcessor.get_content_bytes_from_url(url)
        return None

    @staticmethod
    def _guess_mime(filename: Optional[str]) -> str:
        if not filename:
            return "application/octet-stream"
        mime, _ = mimetypes.guess_type(Path(filename).name)
        return mime or "application/octet-stream"

    @staticmethod
    def to_base64_data_uri(
        content_bytes: bytes,
        mime_type: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> str:
        if not isinstance(content_bytes, bytes):
            raise TypeError("content_bytes must be bytes")
        final_mime = mime_type or MediaProcessor._guess_mime(filename)
        b64 = base64.b64encode(content_bytes).decode("utf-8")
        return f"data:{final_mime};base64,{b64}"


# --- 测试代码（解耦，不依赖 agno.media）---
if __name__ == "__main__":
    import shutil
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    tmp = Path("./temp_media_test")
    tmp.mkdir(exist_ok=True)

    # 创建测试文件
    txt_file = tmp / "test.txt"
    txt_file.write_text("Hello")

    # 测试
    print("Bytes:", len(MediaProcessor.get_content_bytes(content=b"hi"))) # type: ignore
    print("File:", len(MediaProcessor.get_content_bytes(filepath=txt_file))) # type: ignore
    print("URL:", len(MediaProcessor.get_content_bytes(url="https://httpbin.org/image/png")) or 0) # type: ignore

    uri = MediaProcessor.to_base64_data_uri(b"test", filename="test.txt")
    print("Data URI:", uri[:50] + "...")

    shutil.rmtree(tmp)
    print("Test completed.")