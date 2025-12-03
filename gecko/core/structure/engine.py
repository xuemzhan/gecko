# gecko/core/structure/engine.py
"""
结构化输出引擎模块 (v0.4 Phase 3 Complete)

职责：
- 作为结构化解析的统一入口。
- 协调多种解析策略：Tool Call -> 文本提取 -> (新增) LLM 自愈修复。
- 提供 Schema 生成与差异比对工具。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Type, TypeVar, TYPE_CHECKING

from pydantic import BaseModel

from gecko.core.logging import get_logger
from gecko.core.structure.errors import StructureParseError
from gecko.core.structure.json_extractor import extract_structured_data
from . import schema as schema_utils

if TYPE_CHECKING:
    from gecko.core.protocols import ModelProtocol

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructureEngine:
    """
    结构化输出引擎
    """

    # ================= Schema 工具方法 (代理) =================

    @staticmethod
    def to_openai_tool(model: type[BaseModel]) -> Dict[str, Any]:
        """
        将 Pydantic 模型转换为 OpenAI Tool Schema
        """
        return schema_utils.to_openai_tool(model)

    @classmethod
    def get_schema_diff(cls, data: Dict[str, Any], model_class: type[BaseModel]) -> Dict[str, Any]:
        """
        比较数据与模型 Schema 的差异
        """
        return schema_utils.get_schema_diff(data, model_class)

    # ================= 核心解析流程 =================

    @classmethod
    async def parse(
        cls,
        content: str,
        model_class: Type[T],
        raw_tool_calls: Optional[List[Dict[str, Any]]] = None,
        strict: bool = True,
        auto_fix: bool = True,
        model: Optional["ModelProtocol"] = None,
    ) -> T:
        """
        解析文本为 Pydantic 模型 (Async)

        解析策略优先级：
        1. Tool Call: 如果 LLM 使用了工具调用，直接解析参数。
        2. Text Extraction: 尝试从文本中提取 JSON (正则/Markdown/括号匹配)。
        3. LLM Repair: (新增) 如果提供了 model 参数，调用 LLM 尝试修复格式错误的 JSON。

        Args:
            content: LLM 的原始文本输出
            model_class: 目标 Pydantic 模型类
            raw_tool_calls: 原始工具调用列表 (OpenAI 格式)
            strict: 是否启用严格模式 (预留)
            auto_fix: 是否启用基础的字符串清理修复
            model: [Phase 3 新增] 用于自愈修复的模型实例

        Returns:
            T: Pydantic 模型实例

        Raises:
            StructureParseError: 当所有策略均失败时抛出
        """
        attempts: List[Dict[str, str]] = []

        # --- 策略 1: Tool Call 解析 ---
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

        # --- 策略 2-5: 纯文本 JSON 提取 ---
        # 包含：Direct JSON, Markdown Block, Braced {}, Bracket [], Cleaned JSON
        try:
            return extract_structured_data(
                text=content,
                model_class=model_class,
                strict=strict,
                auto_fix=auto_fix,
            )
        except Exception as e:
            # 合并子解析器的尝试记录
            if hasattr(e, "attempts") and getattr(e, "attempts"):
                attempts.extend(getattr(e, "attempts"))
            else:
                attempts.append({"strategy": "text_extraction", "error": str(e)})

        # --- 策略 6: LLM 自愈 (Self-Healing) ---
        if model:
            try:
                # 延迟导入以避免循环依赖
                from gecko.core.structure.repair import repair_json_with_llm
                
                # 获取最后一次报错信息作为参考，帮助 LLM 理解错误原因
                last_error = attempts[-1]['error'] if attempts else "Unknown parsing error"
                
                logger.info("Attempting LLM-based JSON repair...")
                
                # 调用修复逻辑 (Phase 3 核心)
                fixed_dict = await repair_json_with_llm(content, last_error, model)
                
                # 再次校验 Pydantic Schema
                result = model_class.model_validate(fixed_dict)
                
                logger.info(f"StructureEngine: Successfully repaired output for {model_class.__name__}")
                return result
                
            except Exception as repair_err:
                attempts.append({
                    "strategy": "llm_repair", 
                    "error": str(repair_err)
                })

        # --- 最终失败处理 ---
        error_details = "\n".join(
            f"  - {a['strategy']}: {a['error'][:100]}" for a in attempts
        )

        raise StructureParseError(
            f"无法解析为 {model_class.__name__}。尝试了 {len(attempts)} 种策略:\n{error_details}",
            attempts=attempts,
            raw_content=content,
        )

    @classmethod
    def _parse_from_tool_call(cls, call: Dict[str, Any], model_class: Type[T]) -> T:
        """
        从单个工具调用中解析
        """
        func = call.get("function", {})
        args = func.get("arguments", "")

        if isinstance(args, str):
            # 处理可能的 JSON 解析错误
            try:
                data = json.loads(args)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in tool arguments: {e}")
        elif isinstance(args, dict):
            data = args
        else:
            raise ValueError(f"Invalid arguments type: {type(args)}")

        return model_class.model_validate(data)