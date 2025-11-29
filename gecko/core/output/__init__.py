# gecko/core/output/__init__.py
"""
Gecko Agent 输出模型包（模块化版本）

本包对原来的 gecko.core.output 单文件实现进行了拆分与扩展，但：
✅ 对外 API 完全兼容：
    from gecko.core.output import (
        TokenUsage,
        AgentOutput,
        create_text_output,
        create_tool_output,
        merge_outputs,
    )
依然可以正常使用，无需修改其他模块的 import。

当前子模块说明：
- token_usage.py
    定义 TokenUsage：
    - prompt_tokens / completion_tokens / total_tokens
    - 自动校验与补全 total_tokens
    - get_cost_estimate 成本估算

- agent_output.py
    定义 AgentOutput：
    - 标准对话输出模型（content + tool_calls + usage + raw + metadata）
    - 提供格式化、统计、转换为 OpenAI 消息等方法

- factories.py
    定义便捷工厂函数：
    - create_text_output  : 纯文本 AgentOutput
    - create_tool_output  : 带工具调用的 AgentOutput
    - create_json_output  : 结构化 JsonOutput

- merge.py
    定义输出合并逻辑：
    - merge_outputs : 将多个 AgentOutput 合并为一个

- json_output.py
    定义 JsonOutput：
    - 用于承载结构化 JSON / dict / list 形式的结果
    - 可转成 AgentOutput 便于统一处理

- streaming_output.py
    定义流式输出类型：
    - StreamingChunk   : 单个增量片段
    - StreamingOutput  : 管理多个片段并在结束时汇总为 AgentOutput

后续扩展：
- 若需要新增其他输出类型（例如：
  HtmlOutput、MarkdownOutput、RichMediaOutput 等），
  建议在本包内创建新的模块文件，并在本 __init__ 中统一导出。
"""

from .token_usage import TokenUsage
from .agent_output import AgentOutput
from .factories import (
    create_text_output,
    create_tool_output,
    create_json_output,
)
from gecko.core.output.merge import merge_outputs
from gecko.core.output.json_output import JsonOutput
from gecko.core.output.streaming_output import StreamingOutput, StreamingChunk

# __all__ 明确列出对外暴露的符号，便于 IDE 补全与文档生成
__all__ = [
    # 基础 usage 模型
    "TokenUsage",
    # 标准 Agent 输出模型
    "AgentOutput",
    # 常用工厂函数
    "create_text_output",
    "create_tool_output",
    "create_json_output",
    # 合并工具
    "merge_outputs",
    # 扩展输出类型：结构化 JSON
    "JsonOutput",
    # 扩展输出类型：流式输出
    "StreamingOutput",
    "StreamingChunk",
]
