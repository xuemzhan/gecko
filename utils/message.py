# agno/utils/message.py

"""
消息处理工具模块

该模块提供了用于处理和操作 Message 对象列表的辅助函数。
这些功能对于管理对话历史、过滤内容以及从复杂消息结构中提取信息至关重要。

主要功能:
- filter_tool_calls: 从消息历史中裁剪旧的工具调用，保持上下文窗口整洁
- get_text_from_message: 递归地从各种格式的消息输入中提取纯文本内容
"""

import json
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel

# 角色常量
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"
ROLE_SYSTEM = "system"

# 消息字段常量
FIELD_ROLE = "role"
FIELD_CONTENT = "content"
FIELD_TOOL_CALLS = "tool_calls"
FIELD_TOOL_CALL_ID = "tool_call_id"
FIELD_ID = "id"
FIELD_TYPE = "type"
FIELD_TEXT = "text"

# 日志配置
logger = logging.getLogger(__name__)


def log_debug(message: str) -> None:
    """记录调试级别日志"""
    logger.debug(message)


# 动态导入以避免循环依赖
try:
    from models.message import Message
except ImportError:
    # 独立测试时定义桩类
    class Message:
        """消息类的简化实现（用于测试）"""
        
        def __init__(self, **kwargs):
            self.role: Optional[str] = kwargs.get(FIELD_ROLE)
            self.content: Any = kwargs.get(FIELD_CONTENT)
            self.tool_calls: Optional[List[Dict[str, Any]]] = kwargs.get(FIELD_TOOL_CALLS)
            self.tool_call_id: Optional[str] = kwargs.get(FIELD_TOOL_CALL_ID)
        
        def __repr__(self) -> str:
            return f"Message(role={self.role}, content={self.content})"


# --- 工具调用过滤 ---

def filter_tool_calls(messages: List[Message], max_tool_calls: int) -> List[Message]:
    """
    过滤消息列表，仅保留最近的 N 个工具调用及其对应的助理消息
    
    该函数返回新列表，不修改原始列表。它会：
    1. 保留最新的 max_tool_calls 个工具响应
    2. 过滤对应助理消息中的工具调用
    3. 保留所有非工具相关的消息
    
    Args:
        messages: 要过滤的消息列表
        max_tool_calls: 要保留的最近工具调用数量
        
    Returns:
        过滤后的新消息列表
        
    Examples:
        >>> messages = [
        ...     Message(role="assistant", tool_calls=[{"id": "call1"}]),
        ...     Message(role="tool", tool_call_id="call1", content="result"),
        ... ]
        >>> filtered = filter_tool_calls(messages, max_tool_calls=1)
        >>> len([m for m in filtered if m.role == "tool"])
        1
    """
    if max_tool_calls < 0:
        raise ValueError("max_tool_calls 必须为非负整数")
    
    # 统计工具调用数量
    tool_call_count = sum(1 for m in messages if m.role == ROLE_TOOL)
    
    # 无需过滤
    if tool_call_count <= max_tool_calls:
        log_debug(f"工具调用数量 ({tool_call_count}) 未超过限制 ({max_tool_calls})，无需过滤")
        return messages
    
    # 收集需要保留的工具调用 ID
    tool_call_ids_to_keep = _collect_recent_tool_call_ids(messages, max_tool_calls)
    
    # 过滤消息
    filtered_messages = _filter_messages_by_tool_ids(messages, tool_call_ids_to_keep)
    
    # 记录过滤结果
    num_filtered = tool_call_count - len(tool_call_ids_to_keep)
    log_debug(
        f"过滤了 {num_filtered} 个旧工具调用，"
        f"保留了最新的 {len(tool_call_ids_to_keep)} 个"
    )
    
    return filtered_messages


def _collect_recent_tool_call_ids(messages: List[Message], max_count: int) -> Set[str]:
    """
    从后向前收集最近的工具调用 ID
    
    Args:
        messages: 消息列表
        max_count: 最大收集数量
        
    Returns:
        工具调用 ID 集合
    """
    tool_call_ids: Set[str] = set()
    
    for msg in reversed(messages):
        if msg.role == ROLE_TOOL and msg.tool_call_id:
            if len(tool_call_ids) < max_count:
                tool_call_ids.add(msg.tool_call_id)
            else:
                break
    
    return tool_call_ids


def _filter_messages_by_tool_ids(
    messages: List[Message],
    keep_ids: Set[str]
) -> List[Message]:
    """
    根据工具调用 ID 集合过滤消息
    
    Args:
        messages: 原始消息列表
        keep_ids: 要保留的工具调用 ID 集合
        
    Returns:
        过滤后的消息列表
    """
    filtered_messages: List[Message] = []
    
    for msg in messages:
        if msg.role == ROLE_TOOL:
            # 仅保留白名单中的工具响应
            if msg.tool_call_id in keep_ids:
                filtered_messages.append(msg)
        
        elif msg.role == ROLE_ASSISTANT and msg.tool_calls:
            # 过滤助理消息中的工具调用
            filtered_msg = _filter_assistant_tool_calls(msg, keep_ids)
            if filtered_msg is not None:
                filtered_messages.append(filtered_msg)
        
        else:
            # 保留所有其他消息（user, system 等）
            filtered_messages.append(msg)
    
    return filtered_messages


def _filter_assistant_tool_calls(
    msg: Message,
    keep_ids: Set[str]
) -> Optional[Message]:
    """
    过滤助理消息中的工具调用
    
    Args:
        msg: 助理消息
        keep_ids: 要保留的工具调用 ID 集合
        
    Returns:
        过滤后的消息，如果消息应被完全移除则返回 None
    """
    # 深拷贝以安全修改
    filtered_msg = deepcopy(msg)
    
    # 过滤工具调用列表
    if filtered_msg.tool_calls:
        filtered_msg.tool_calls = [
            tc for tc in filtered_msg.tool_calls
            if tc.get(FIELD_ID) in keep_ids
        ]
    
    # 决定是否保留该消息
    if filtered_msg.tool_calls:
        return filtered_msg
    elif filtered_msg.content:
        # 没有工具调用但有内容，清空 tool_calls 后保留
        filtered_msg.tool_calls = None
        return filtered_msg
    
    # 既没有工具调用也没有内容，丢弃
    return None


# --- 文本提取 ---

def get_text_from_message(message: Union[List, Dict, str, Message, BaseModel]) -> str:
    """
    从复杂消息结构中递归提取用户可读的文本内容
    
    支持的输入格式：
    - 字符串：直接返回
    - Message 实例：提取 user 角色的 content
    - 字典：处理 OpenAI 格式消息
    - 列表：递归处理每个元素
    - Pydantic 模型：提取 content 属性或序列化
    
    Args:
        message: 消息数据，可以是多种格式
        
    Returns:
        提取的文本内容，多个部分用换行符连接
        
    Examples:
        >>> get_text_from_message("Hello")
        'Hello'
        >>> get_text_from_message({"role": "user", "content": "Hi"})
        'Hi'
        >>> get_text_from_message([{"role": "user", "content": "A"}, "B"])
        'A\\nB'
    """
    # 处理字符串
    if isinstance(message, str):
        return message
    
    # 处理 Pydantic 模型
    if isinstance(message, BaseModel):
        return _extract_from_pydantic_model(message)
    
    # 处理列表
    if isinstance(message, list):
        return _extract_from_list(message)
    
    # 处理字典
    if isinstance(message, dict):
        return _extract_from_dict(message)
    
    # 处理 Message 实例
    if isinstance(message, Message):
        return _extract_from_message_instance(message)
    
    # 其他类型
    logger.warning(f"无法从类型 {type(message).__name__} 提取文本")
    return ""


def _extract_from_pydantic_model(model: BaseModel) -> str:
    """从 Pydantic 模型提取文本"""
    # 尝试提取 content 属性
    if hasattr(model, FIELD_CONTENT):
        content = getattr(model, FIELD_CONTENT)
        return get_text_from_message(content)
    
    # 序列化为 JSON
    try:
        return model.model_dump_json(indent=2, exclude_none=True)
    except Exception as e:
        logger.warning(f"无法序列化 Pydantic 模型: {e}")
        return str(model)


def _extract_from_list(message_list: List) -> str:
    """从列表中递归提取文本"""
    if not message_list:
        return ""
    
    text_parts = []
    for item in message_list:
        text = get_text_from_message(item)
        if text:
            text_parts.append(text)
    
    return "\n".join(text_parts)


def _extract_from_dict(message_dict: Dict) -> str:
    """从字典中提取文本"""
    role = message_dict.get(FIELD_ROLE)
    content = message_dict.get(FIELD_CONTENT)
    
    # 如果字典包含角色信息
    if role is not None:
        # 只处理 user 角色的消息
        if role == ROLE_USER and content is not None:
            # 如果 content 是列表（多模态内容）
            if isinstance(content, list):
                return _extract_from_multimodal_content(content)
            # 否则递归处理 content
            return get_text_from_message(content)
        # 非 user 角色，返回空字符串
        return ""
    
    # 没有角色信息的字典
    # 检查是否是多模态内容列表
    if isinstance(content, list):
        return _extract_from_multimodal_content(content)
    
    # 如果有 content 字段，递归处理
    if FIELD_CONTENT in message_dict:
        return get_text_from_message(content)
    
    # 序列化整个字典
    return _serialize_dict(message_dict)


def _extract_from_multimodal_content(content_list: List) -> str:
    """从多模态内容列表中提取文本部分"""
    text_parts = []
    
    for part in content_list:
        if isinstance(part, dict) and part.get(FIELD_TYPE) == FIELD_TEXT:
            text = part.get(FIELD_TEXT, "")
            if text:
                text_parts.append(text)
    
    return "\n".join(text_parts)


def _extract_from_message_instance(message: Message) -> str:
    """从 Message 实例提取文本"""
    if message.role == ROLE_USER and message.content:
        return get_text_from_message(message.content)
    return ""


def _serialize_dict(data: Dict) -> str:
    """将字典序列化为 JSON 字符串"""
    try:
        return json.dumps(data, indent=2, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        logger.warning(f"无法序列化字典: {e}")
        return str(data)


# --- 测试代码 ---

def _test_filter_tool_calls() -> None:
    """测试工具调用过滤功能"""
    print("\n[1] 测试 filter_tool_calls:")
    
    # 构建测试消息
    messages = [
        Message(role=ROLE_USER, content="搜索天气和新闻"),
        Message(role=ROLE_ASSISTANT, tool_calls=[{FIELD_ID: "call1"}, {FIELD_ID: "call2"}]),
        Message(role=ROLE_TOOL, tool_call_id="call1", content="天气晴朗"),
        Message(role=ROLE_TOOL, tool_call_id="call2", content="新闻头条..."),
        Message(role=ROLE_USER, content="再帮我订个票"),
        Message(role=ROLE_ASSISTANT, tool_calls=[{FIELD_ID: "call3"}]),
        Message(role=ROLE_TOOL, tool_call_id="call3", content="订票成功"),
        Message(role=ROLE_ASSISTANT, content="好的，都办完了。"),
    ]
    
    original_tool_count = sum(1 for m in messages if m.role == ROLE_TOOL)
    print(f"  原始消息数量: {len(messages)}")
    print(f"  原始工具调用数量: {original_tool_count}")
    
    # 测试保留 1 个工具调用
    filtered = filter_tool_calls(messages, max_tool_calls=1)
    filtered_tool_count = sum(1 for m in filtered if m.role == ROLE_TOOL)
    
    print(f"\n  保留最近 1 个工具调用:")
    print(f"  过滤后消息数量: {len(filtered)}")
    print(f"  过滤后工具调用数量: {filtered_tool_count}")
    
    # 验证
    try:
        assert filtered_tool_count == 1, "应该只保留 1 个工具调用"
        
        tool_ids = {m.tool_call_id for m in filtered if m.role == ROLE_TOOL}
        assert "call3" in tool_ids, "应该保留最新的工具调用"
        assert "call1" not in tool_ids, "应该移除旧的工具调用"
        
        print("  ✓ 验证通过")
    except AssertionError as e:
        print(f"  ✗ 验证失败: {e}")
    
    # 测试边界情况
    print("\n  测试边界情况:")
    
    # 保留所有工具调用
    all_filtered = filter_tool_calls(messages, max_tool_calls=10)
    assert sum(1 for m in all_filtered if m.role == ROLE_TOOL) == original_tool_count
    print("  ✓ 保留所有工具调用正常")
    
    # 空列表
    empty_filtered = filter_tool_calls([], max_tool_calls=1)
    assert len(empty_filtered) == 0
    print("  ✓ 空列表处理正常")
    
    # 无工具调用的消息
    no_tool_msgs = [Message(role=ROLE_USER, content="Hello")]
    no_tool_filtered = filter_tool_calls(no_tool_msgs, max_tool_calls=1)
    assert len(no_tool_filtered) == 1
    print("  ✓ 无工具调用消息处理正常")


def _test_get_text_from_message() -> None:
    """测试文本提取功能"""
    print("\n[2] 测试 get_text_from_message:")
    
    class DummyModel(BaseModel):
        content: str
        value: int
    
    test_cases = [
        ("简单字符串", "你好，世界", "你好，世界"),
        ("用户消息字典", {FIELD_ROLE: ROLE_USER, FIELD_CONTENT: "用户消息"}, "用户消息"),
        ("助理消息字典", {FIELD_ROLE: ROLE_ASSISTANT, FIELD_CONTENT: "助理消息"}, ""),
        ("多模态内容", {
            FIELD_ROLE: ROLE_USER,
            FIELD_CONTENT: [
                {FIELD_TYPE: FIELD_TEXT, FIELD_TEXT: "描述这张图片"},
                {FIELD_TYPE: "image_url"}
            ]
        }, "描述这张图片"),
        ("用户 Message", Message(role=ROLE_USER, content="来自 Message"), "来自 Message"),
        ("助理 Message", Message(role=ROLE_ASSISTANT, content="助理消息"), ""),
        ("Pydantic 模型", DummyModel(content="模型内容", value=1), "模型内容"),
        ("嵌套列表", ["第一行", Message(role=ROLE_USER, content="第二行")], "第一行\n第二行"),
        ("空列表", [], ""),
    ]
    
    passed = 0
    failed = 0
    
    for name, input_msg, expected in test_cases:
        result = get_text_from_message(input_msg)
        
        if result == expected:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name}")
            print(f"    期望: '{expected}'")
            print(f"    实际: '{result}'")
            failed += 1
    
    print(f"\n  测试结果: {passed} 通过, {failed} 失败")


def _test_edge_cases() -> None:
    """测试边缘情况"""
    print("\n[3] 测试边缘情况:")
    
    try:
        # 测试无效的 max_tool_calls
        try:
            filter_tool_calls([], max_tool_calls=-1)
            print("  ✗ 应该拒绝负数的 max_tool_calls")
        except ValueError:
            print("  ✓ 正确拒绝负数的 max_tool_calls")
        
        # 测试 None 值
        result = get_text_from_message(None)
        print(f"  ✓ None 值处理")
        
        # 测试嵌套字典
        nested = {FIELD_CONTENT: {FIELD_CONTENT: "深层内容"}}
        result = get_text_from_message(nested)
        assert result == "深层内容"
        print(f"  ✓ 嵌套字典处理")
        
        # 测试包含 tool_calls 但 ID 不匹配的情况
        msg_with_invalid_tool = Message(
            role=ROLE_ASSISTANT,
            tool_calls=[{FIELD_ID: "invalid_id"}],
            content=None
        )
        filtered = filter_tool_calls([msg_with_invalid_tool], max_tool_calls=0)
        print(f"  ✓ 无效 tool_call ID 处理正常")
        
        # 测试多行多模态内容
        multimodal = {
            FIELD_ROLE: ROLE_USER,
            FIELD_CONTENT: [
                {FIELD_TYPE: FIELD_TEXT, FIELD_TEXT: "第一行"},
                {FIELD_TYPE: FIELD_TEXT, FIELD_TEXT: "第二行"},
                {FIELD_TYPE: "image"}
            ]
        }
        result = get_text_from_message(multimodal)
        assert result == "第一行\n第二行"
        print(f"  ✓ 多行多模态内容处理")
        
    except Exception as e:
        print(f"  ✗ 边缘情况测试失败: {e}")
        import traceback
        traceback.print_exc()


def _run_tests() -> None:
    """运行所有测试"""
    print("=" * 60)
    print("运行 agno/utils/message.py 测试")
    print("=" * 60)
    
    _test_filter_tool_calls()
    _test_get_text_from_message()
    _test_edge_cases()
    
    print("\n" + "=" * 60)
    print("✓ 测试完成")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s: %(message)s'
    )
    _run_tests()