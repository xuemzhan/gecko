# agno/models/formatters/base.py

"""
模型提供商格式化器抽象基类

该模块定义了 `BaseProviderFormatter` 抽象基类（ABC），它为所有特定于
模型提供商的数据格式化器提供了一个统一的接口。

Formatter 的核心职责是将 Agno 内部统一的数据结构（如 `Message` 对象、工具定义）
转换为特定 LLM 提供商 API 所需的精确格式。

通过使用这个抽象层，Agent 的核心逻辑可以与具体模型的 API 细节解耦，
从而轻松支持新的模型提供商，只需为其实现一个新的 Formatter 即可。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

from pydantic import BaseModel

# 动态导入以避免循环依赖
try:
    from agno.media import Audio, File, Image, Video
    from agno.models.message import Message
except ImportError:
    # 定义桩类以供独立测试
    class Message: pass
    class Image: pass
    class Audio: pass
    class Video: pass
    class File: pass


class BaseProviderFormatter(ABC):
    """
    一个抽象基类，定义了所有模型提供商格式化器必须实现的接口。
    """

    @abstractmethod
    def format_messages(self, messages: List[Message]) -> Any:
        """
        将 Agno 的 `Message` 对象列表转换为特定模型提供商 API 所需的格式。

        不同的提供商对此有不同的要求，例如：
        - OpenAI/Mistral: 需要一个字典列表 `[{"role": ..., "content": ...}]`。
        - Anthropic: 需要一个字典列表，并将 "system" 角色的消息分离出来。
        - Gemini: 需要 `Content` 对象列表。

        Args:
            messages (List[Message]): 待格式化的 `Message` 对象列表。

        Returns:
            Any: 格式化后的消息，其类型取决于具体提供商的要求。
                 通常是一个列表或元组。
        """
        raise NotImplementedError

    @abstractmethod
    def format_tools(self, tools: List[Dict[str, Any]]) -> Optional[Any]:
        """
        将 Agno 内部的工具定义转换为特定模型提供商 API 所需的工具格式。

        Args:
            tools (List[Dict[str, Any]]): 待格式化的工具定义列表。

        Returns:
            Optional[Any]: 格式化后的工具定义，如果模型不支持工具则返回 None。
        """
        raise NotImplementedError

    def format_structured_output_prompt(
        self, 
        output_schema: Union[Type[BaseModel], Dict]
    ) -> Optional[str]:
        """
        （可选）为结构化输出（JSON 模式）生成特定的提示字符串。

        某些模型（如旧版模型）可能需要通过系统提示来强制执行 JSON 输出格式。
        如果提供商通过 API 参数原生支持 JSON 模式，则此方法可以返回 None。

        Args:
            output_schema (Union[Type[BaseModel], Dict]): Pydantic 模型或 JSON Schema 字典。

        Returns:
            Optional[str]: 附加到系统提示中的指令字符串，或 None。
        """
        # 默认实现：使用 agno.utils.prompts 中的通用函数。
        # 子类可以覆盖此方法以提供特定于模型的实现。
        try:
            from agno.utils.prompts import get_json_output_prompt
            if isinstance(output_schema, dict):
                # 如果是字典，假设它已经是JSON Schema
                # 简单地将其转换为JSON字符串
                return get_json_output_prompt(json.dumps(output_schema, indent=2))
            return get_json_output_prompt(output_schema)
        except ImportError:
            # 在无法导入时提供回退
            import json
            return f"Please format your response as a valid JSON object. Schema: {json.dumps(output_schema, indent=2)}"

    def format_media(
        self,
        images: Optional[Sequence[Image]] = None,
        videos: Optional[Sequence[Video]] = None,
        audios: Optional[Sequence[Audio]] = None,
        files: Optional[Sequence[File]] = None,
    ) -> Dict[str, Any]:
        """
        （可选）格式化多模态媒体输入。

        大多数现代 Formatter 会将媒体信息直接整合到 `format_messages` 的 `content` 
        字段中。这个方法可以作为一个备用或辅助方法，用于处理不支持
        多部分 `content` 的模型。

        Args:
            images: 图像序列。
            videos: 视频序列。
            audios: 音频序列。
            files: 文件序列。

        Returns:
            Dict[str, Any]: 一个包含格式化后媒体数据的字典，准备好合并到 API 请求中。
        """
        # 默认实现为空，因为格式化逻辑通常在 `format_messages` 中处理。
        return {}


if __name__ == "__main__":
    # --- 测试代码 ---
    import json
    
    print("--- 正在运行 agno/models/formatters/base.py 的测试代码 ---")

    # 1. 验证抽象方法
    print("\n[1] 验证抽象方法:")
    
    # 尝试实例化一个没有实现抽象方法的子类，预期会失败
    class IncompleteFormatter(BaseProviderFormatter):
        pass

    try:
        formatter = IncompleteFormatter()
        print("  [失败] IncompleteFormatter 竟然可以被实例化。")
    except TypeError as e:
        print(f"  [成功] 无法实例化不完整的 Formatter: {e}")

    # 2. 创建一个最小化的完整实现用于测试
    class MockFormatter(BaseProviderFormatter):
        def format_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
            # 这是一个非常简化的模拟实现
            return [{"role": m.role, "content": m.content} for m in messages]

        def format_tools(self, tools: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
            # 假设模型直接接受这种格式
            return tools

    print("\n[2] 测试一个最小化的 Formatter 实现:")
    mock_formatter = MockFormatter()
    
    # 模拟 Message 对象
    class MockMessage:
        def __init__(self, role, content):
            self.role = role
            self.content = content

    mock_messages = [
        MockMessage("user", "Hello"),
        MockMessage("assistant", "Hi there!")
    ]
    
    formatted_messages = mock_formatter.format_messages(mock_messages)
    print(f"  格式化后的消息: {formatted_messages}")
    assert formatted_messages == [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]

    mock_tools = [{"type": "function", "function": {"name": "search", "description": "Search the web."}}]
    formatted_tools = mock_formatter.format_tools(mock_tools)
    print(f"  格式化后的工具: {formatted_tools}")
    assert formatted_tools == mock_tools

    # 3. 测试默认的 format_structured_output_prompt 方法
    print("\n[3] 测试默认的结构化输出提示生成:")
    
    class MyOutput(BaseModel):
        """一个简单的输出模型。"""
        name: str
        value: int

    # 模拟 agno.utils.prompts 模块
    class MockPrompts:
        def get_json_output_prompt(self, schema):
            return f"JSON PROMPT FOR: {schema.__name__ if hasattr(schema, '__name__') else schema}"
    
    import sys
    sys.modules['agno.utils.prompts'] = MockPrompts()

    prompt = mock_formatter.format_structured_output_prompt(MyOutput)
    print(f"  为 Pydantic 模型生成的提示:\n---\n{prompt}\n---")
    assert "MyOutput" in prompt

    del sys.modules['agno.utils.prompts']

    print("\n--- 测试结束 ---")