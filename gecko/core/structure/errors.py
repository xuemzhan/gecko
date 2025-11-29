# gecko/core/structure/errors.py
"""
错误类型定义模块

专门存放与结构化解析相关的自定义异常类型，避免在其他模块中
重复定义或产生循环依赖。
"""

from __future__ import annotations

from typing import Dict, List, Optional


class StructureParseError(ValueError):
    """
    结构化解析失败异常

    设计目标：
    - 在一次结构化解析过程中，我们会尝试多种解析策略（tool call、直接 JSON、
      markdown 代码块、括号匹配、清理重试等）。
    - 当所有策略都失败时，用一个统一的异常类型向调用方报告，并附带详细的错误轨迹。

    属性:
        message:
            主错误信息（继承自 ValueError 的 message）
        attempts:
            所有尝试的解析策略及其错误信息列表，
            每个元素形如 {"strategy": "...", "error": "..."}。
        raw_content:
            原始内容（通常是 LLM 的输出）。为了避免日志爆炸，应在外部使用时做长度截断。
    """

    def __init__(
        self,
        message: str,
        attempts: Optional[List[Dict[str, str]]] = None,
        raw_content: Optional[str] = None,
    ):
        super().__init__(message)
        # 保证属性始终存在，便于调用方无脑访问
        self.attempts: List[Dict[str, str]] = attempts or []
        self.raw_content: Optional[str] = raw_content

    def get_detailed_error(self) -> str:
        """
        将解析失败过程格式化为可读字符串

        用途：
            - 日志输出
            - 调试时打印详细信息
        """
        lines = [f"结构化解析失败: {self.args[0]}"]

        if self.attempts:
            lines.append("\n尝试的解析策略:")
            for i, attempt in enumerate(self.attempts, 1):
                strategy = attempt.get("strategy", "unknown")
                error = attempt.get("error", "unknown error")
                lines.append(f"  {i}. {strategy}: {error}")

        if self.raw_content:
            # 只展示前 200 字符，避免日志过长
            preview = self.raw_content[:200].replace("\n", "\\n")
            lines.append(f"\n原始内容预览: {preview}...")

        return "\n".join(lines)
