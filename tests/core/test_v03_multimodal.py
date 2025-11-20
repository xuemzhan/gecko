# tests/core/test_v03_multimodal.py
import pytest
import tempfile
import os
import base64
from gecko.core.message import Message

def test_message_user_factory_mixed_content():
    """验证快捷工厂能否正确混合文本和图片"""
    raw_bytes = b"fake_image_bytes"
    expected_b64 = base64.b64encode(raw_bytes).decode("utf-8")
    
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        # 使用工厂方法
        msg = Message.user(
            text="Analyze this",
            images=["https://example.com/remote.jpg", tmp_path]
        )

        # 1. 验证内部结构
        assert len(msg.content) == 3
        assert msg.content[0].type == "text"
        assert msg.content[1].type == "image_url"
        assert msg.content[2].type == "image_url"

        # 2. 验证 API Payload 序列化
        payload = msg.to_api_payload()

        # 检查远程图片
        remote_img = payload["content"][1]
        assert remote_img["image_url"]["url"] == "https://example.com/remote.jpg"

        # 检查本地图片
        local_img = payload["content"][2]
        # [修复] 断言检查 Base64 编码后的内容
        assert local_img["image_url"]["url"] == f"data:image/jpeg;base64,{expected_b64}"

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_message_serialization_purity():
    """验证序列化后的字典不包含 Pydantic 的私有字段"""
    msg = Message(role="user", content="Simple text")
    payload = msg.to_api_payload()
    
    assert isinstance(payload, dict)
    assert payload["role"] == "user"
    assert payload["content"] == "Simple text"