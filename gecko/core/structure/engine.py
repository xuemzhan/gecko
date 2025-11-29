# gecko/core/structure/engine.py
"""
结构化输出引擎模块

职责：
- 作为“协调者”，将不同解析策略串联起来：
    * 优先尝试从工具调用（tool calls）中提取结构化数据
    * 如果没有 tool calls 或解析失败，则回退到纯文本 JSON 提取
- 暴露统一的异步接口 `StructureEngine.parse`
- 将 schema 相关的工具（OpenAI tool schema、schema diff）作为静态/类方法对外暴露，
  方便调用方统一访问。

内部依赖：
- errors.StructureParseError：统一错误类型
- json_extractor.extract_structured_data：文本 JSON 提取的具体实现
- schema.to_openai_tool / schema.get_schema_diff：Schema 工具函数
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from gecko.core.logging import get_logger

from gecko.core.structure.errors import StructureParseError
from gecko.core.structure.json_extractor import extract_structured_data
from . import schema as schema_utils

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructureEngine:
    """
    结构化输出引擎

    使用示例：

        from pydantic import BaseModel

        class User(BaseModel):
            name: str
            age: int

        # 异步场景
        result = await StructureEngine.parse(
            content='{"name": "Alice", "age": 25}',
            model_class=User,
        )

        # 如果有 tool calls（例如来自 OpenAI / Zhipu 的工具调用结果）
        result = await StructureEngine.parse(
            content="",
            model_class=User,
            raw_tool_calls=[{
                "function": {
                    "arguments": '{"name": "Alice", "age": 25}'
                }
            }]
        )
    """

    # ===== Schema 相关静态方法 =====

    @staticmethod
    def to_openai_tool(model: type[BaseModel]) -> Dict[str, Any]:
        """
        将 Pydantic 模型转换为 OpenAI Function Calling 所需的 tool schema

        说明：
            - 内部委托给 schema_utils.to_openai_tool 实现
            - 之所以作为静态方法暴露，是为了保持与旧版本 API 一致
        """
        return schema_utils.to_openai_tool(model)

    @classmethod
    def get_schema_diff(
        cls,
        data: Dict[str, Any],
        model_class: type[BaseModel],
    ) -> Dict[str, Any]:
        """
        比较数据与模型 schema 的差异（基础版）

        说明：
            - 内部委托给 schema_utils.get_schema_diff 实现
            - 作为类方法暴露，方便调用方通过 StructureEngine 统一访问
        """
        return schema_utils.get_schema_diff(data, model_class)

    # ===== 核心解析方法 =====

    @classmethod
    async def parse(
        cls,
        content: str,
        model_class: Type[T],
        raw_tool_calls: Optional[List[Dict[str, Any]]] = None,
        strict: bool = True,
        auto_fix: bool = True,
    ) -> T:
        """
        解析文本为 Pydantic 模型（异步接口）

        参数:
            content:
                LLM 的原始文本输出
            model_class:
                目标 Pydantic 模型类
            raw_tool_calls:
                原始工具调用列表（例如 OpenAI tools 调用结果）
                如果提供，则优先使用 tool calls 作为结构化数据来源。
            strict:
                预留参数，当前版本中尚未改变具体行为；
                未来可用于控制“宽松模式”（例如更激进的类型转换）。
            auto_fix:
                是否对明显的 JSON 格式问题进行自动修复（去注释、去尾逗号等）

        返回:
            Pydantic 模型实例

        异常:
            StructureParseError:
                当所有解析策略均失败时抛出，并包含详细的尝试轨迹。
        """
        attempts: List[Dict[str, str]] = []

        # 策略 1: 从 tool calls 中解析（如果提供了）
        if raw_tool_calls:
            for idx, call in enumerate(raw_tool_calls):
                try:
                    result = cls._parse_from_tool_call(call, model_class)
                    logger.info(
                        "Parsed from tool call",
                        model=model_class.__name__,
                        tool_call_index=idx,
                    )
                    return result
                except Exception as e:
                    attempts.append(
                        {
                            "strategy": f"tool_call_{idx}",
                            "error": str(e),
                        }
                    )
                    logger.debug(
                        "Tool call parse failed",
                        index=idx,
                        error=str(e),
                    )

        # 策略 2-5: 纯文本 JSON 提取与解析
        try:
            return extract_structured_data(
                text=content,
                model_class=model_class,
                strict=strict,
                auto_fix=auto_fix,
            )
        except Exception as e:
            # 如果子异常本身就是 StructureParseError，并带有 attempts，
            # 则直接合并子层 attempts，避免信息丢失。
            if hasattr(e, "attempts") and getattr(e, "attempts"):  # type: ignore[attr-defined]
                attempts.extend(getattr(e, "attempts"))  # type: ignore[attr-defined]
            else:
                attempts.append(
                    {
                        "strategy": "text_extraction",
                        "error": str(e),
                    }
                )

            error_details = "\n".join(
                f"  - {a['strategy']}: {a['error'][:100]}" for a in attempts
            )

            raise StructureParseError(
                f"无法解析为 {model_class.__name__}。尝试了 {len(attempts)} 种策略:\n{error_details}",
                attempts=attempts,
                raw_content=content,
            ) from e

    # ===== 内部工具调用解析 =====

    @classmethod
    def _parse_from_tool_call(
        cls,
        call: Dict[str, Any],
        model_class: Type[T],
    ) -> T:
        """
        从单个工具调用中解析出目标模型实例

        典型工具调用结构（以 OpenAI 为例）：
        {
            "function": {
                "name": "xxx",
                "arguments": "{...}"   # JSON 字符串
            }
            ...
        }
        """
        func = call.get("function", {})
        args = func.get("arguments", "")

        # 解析 arguments 字段
        if isinstance(args, str):
            data = json.loads(args)
        elif isinstance(args, dict):
            data = args
        else:
            raise ValueError(f"Invalid arguments type: {type(args)}")

        return model_class(**data)
