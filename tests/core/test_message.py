# tests/core/test_message.py
from gecko.core.message import Message, ContentBlock

def test_message_serialization():
    # 1. 普通文本
    msg = Message.user("hello")
    dump = msg.to_openai_format()
    assert dump["role"] == "user"
    assert dump["content"] == "hello"
    
    # 2. 多模态
    msg_multi = Message.user("look", images=["http://fake.url/img.jpg"])
    dump_multi = msg_multi.to_openai_format()
    assert isinstance(dump_multi["content"], list)
    assert dump_multi["content"][0]["type"] == "text"
    assert dump_multi["content"][1]["type"] == "image_url"
    
def test_message_factories():
    assert Message.system("sys").role == "system"
    assert Message.assistant("hi").role == "assistant"
    assert Message.tool_result("id", "res", "tool").role == "tool"