# gecko/core/structure/json_extractor.py
"""
JSON 提取与模型验证模块（含 Strategy 插件机制）

职责：
- 从原始文本中提取 JSON 片段（支持多种启发式策略）
- 将 JSON 解析为 Python 对象，并使用 Pydantic 模型进行校验
- 在解析失败时抛出 StructureParseError，并记录详细策略日志
- 提供 Strategy 插件接口，允许外部按需扩展解析策略
  （例如 YAML、特殊标记、正则提取等）

分层思想：
- 本模块属于“算法层”，只关心“文本 → JSON → 模型”流程
- 上层的 StructureEngine 负责 tool call 等语义层逻辑
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

from gecko.core.logging import get_logger

from .errors import StructureParseError

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


# ============================================================================
# Strategy 插件接口定义
# ============================================================================

@dataclass
class ExtractionStrategy:
    """
    文本 → 模型 的解析策略对象

    属性:
        name:
            策略名称，用于日志和错误信息标识，例如 "yaml_fulltext"。
        func:
            策略函数，签名固定为：
                func(text: str, model_class: Type[T]) -> T

            要求：
            - 成功解析时直接返回 Pydantic 模型实例
            - 解析失败时抛出异常（异常类型不限）
            - 如果策略“不适用当前文本”，也可以抛出异常，外层会将其记录为一次尝试
              （未来如有需要，可以约定特定异常类型用于“跳过不计入 attempts”）
    """
    name: str
    func: Callable[[str, Type[T]], T] # type: ignore


# 全局插件策略列表（内置策略不在此列表中）
_EXTRA_STRATEGIES: List[ExtractionStrategy] = []


def register_extraction_strategy(strategy: ExtractionStrategy) -> None:
    """
    向全局注册一个额外的解析策略（插件）

    使用示例：
        from gecko.core.structure import ExtractionStrategy, register_extraction_strategy

        def my_strategy(text, model_class):
            # ... 自己的逻辑 ...
            return model_class(**data)

        register_extraction_strategy(
            ExtractionStrategy(name="my_strategy", func=my_strategy)
        )

    注意：
        - 插件策略会在所有内置策略失败之后被依次尝试
        - 插件策略的执行顺序与注册顺序一致
    """
    _EXTRA_STRATEGIES.append(strategy)


# ============================================================================
# 主入口函数：从文本解析结构化数据
# ============================================================================


def extract_structured_data(
    text: str,
    model_class: Type[T],
    *,
    strict: bool = True,  # 当前版本尚未真正启用 strict 行为，仅作为未来扩展预留
    auto_fix: bool = True,
    max_text_length: int = 100000,
) -> T:
    """
    从文本中提取 JSON 并解析为指定的 Pydantic 模型

    内置策略执行顺序（从快到慢，从精确到模糊）：
        1. direct_json    : 直接整体 json.loads(text)
        2. markdown_X     : 从 ``` ... ``` 代码块中提取 JSON
        3. braced_X       : 从 `{...}` 片段中提取 JSON
        4. bracket_X      : 从 `[...]` 片段中提取 JSON（数组）
        5. cleaned_json   : 清理文本后再整体解析
        ---- 以下为插件策略（按注册顺序执行）----
        6. plugin_<name>  : 所有通过 register_extraction_strategy 注册的策略

    参数:
        text:
            原始文本（通常是 LLM 输出）
        model_class:
            目标 Pydantic 模型类
        strict:
            预留参数，当前版本不改变行为；
            未来可用于控制“宽松模式”（自动类型转换等）。
        auto_fix:
            是否对明显的 JSON 格式问题进行自动修复（去注释、去尾逗号等）
        max_text_length:
            最大处理文本长度，超出部分将被截断，以防止极端长输出导致性能问题。

    返回:
        Pydantic 模型实例

    异常:
        StructureParseError:
            当所有内置策略和插件策略都失败时抛出，并包含详细的 attempts 列表。
    """
    # 超长文本截断，避免无意义的解析和性能问题
    if len(text) > max_text_length:
        logger.warning(
            "Text too long for JSON extraction, truncating",
            original_length=len(text),
            max_length=max_text_length,
        )
        text = text[:max_text_length]

    text = text.strip()
    attempts: List[Dict[str, str]] = []

    # 0. 快速失败检查：如果既没有 { 也没有 [，基本可以判断不是 JSON
    if not text or ("{" not in text and "[" not in text):
        raise StructureParseError(
            "Content does not contain JSON-like structure (missing '{' or '[')",
            raw_content=text[:500] if text else "",
        )

    # ----------------------------------------------------------------------
    # 策略 A: 整体解析
    # ----------------------------------------------------------------------
    try:
        data = json.loads(text)
        return _validate_model(data, model_class)
    except Exception as e:
        attempts.append(
            {
                "strategy": "direct_json",
                "error": str(e),
            }
        )

    # ----------------------------------------------------------------------
    # 策略 B: Markdown 代码块 ```json ...``` / ``` ... ```
    # ----------------------------------------------------------------------
    markdown_pattern = r"```(?:\w+)?\s*([\s\S]*?)```"
    for idx, match in enumerate(re.finditer(markdown_pattern, text)):
        candidate = match.group(1).strip()
        try:
            data = json.loads(candidate)
            return _validate_model(data, model_class)
        except Exception as e:
            # 仅记录前 3 个失败，避免 attempts 过长
            if idx < 3:
                attempts.append(
                    {
                        "strategy": f"markdown_{idx}",
                        "error": str(e),
                    }
                )

    # ----------------------------------------------------------------------
    # 策略 C: 花括号 `{...}` 片段提取
    # ----------------------------------------------------------------------
    obj_candidates = extract_braced_json(text)
    for idx, candidate in enumerate(obj_candidates):
        try:
            data = json.loads(candidate)
            return _validate_model(data, model_class)
        except Exception as e:
            if idx < 3:
                attempts.append(
                    {
                        "strategy": f"braced_{idx}",
                        "error": str(e),
                    }
                )

    # ----------------------------------------------------------------------
    # 策略 C2: 中括号 `[...]` 片段提取（数组 JSON）
    # ----------------------------------------------------------------------
    array_candidates = extract_bracket_json(text)
    for idx, candidate in enumerate(array_candidates):
        try:
            data = json.loads(candidate)
            return _validate_model(data, model_class)
        except Exception as e:
            if idx < 3:
                attempts.append(
                    {
                        "strategy": f"bracket_{idx}",
                        "error": str(e),
                    }
                )

    # ----------------------------------------------------------------------
    # 策略 D: 简单清理后整体重试
    # ----------------------------------------------------------------------
    if auto_fix:
        cleaned = clean_json_string(text)
        if cleaned != text:
            try:
                data = json.loads(cleaned)
                logger.info(
                    "Parsed after cleaning",
                    model=model_class.__name__,
                )
                return _validate_model(data, model_class)
            except Exception as e:
                attempts.append(
                    {
                        "strategy": "cleaned_json",
                        "error": str(e),
                    }
                )

    # ----------------------------------------------------------------------
    # 策略 E: 插件策略（Strategy 插件机制）
    #     - 按注册顺序逐个尝试
    #     - 每个策略以自身的 name 记录在 attempts 中
    # ----------------------------------------------------------------------
    for strategy in _EXTRA_STRATEGIES:
        try:
            return strategy.func(text, model_class) # type: ignore
        except Exception as e:
            attempts.append(
                {
                    "strategy": f"plugin_{strategy.name}",
                    "error": str(e),
                }
            )

    # 所有策略均失败，汇总错误信息并抛出统一异常
    error_details = "\n".join(
        f"  - {a['strategy']}: {a['error'][:100]}" for a in attempts
    )
    raise StructureParseError(
        f"无法解析为 {model_class.__name__}。尝试了 {len(attempts)} 种策略:\n{error_details}",
        attempts=attempts,
        raw_content=text,
    )


# ============================================================================
# JSON 片段提取工具函数
# ============================================================================


def extract_braced_json(
    text: str,
    max_text_length: int = 100000,
    max_candidates: int = 5,
) -> List[str]:
    """
    使用栈提取所有看起来像 JSON 对象的 `{...}` 片段

    改进点：
        - 返回多个候选，而不仅仅是第一个
        - 候选数量由 max_candidates 控制，避免极端文本导致过多尝试
        - 文本长度由 max_text_length 限制，避免处理超长输出

    简化假设：
        - 不区分字符串内部的 `{` / `}`，对于大多数 LLM 输出场景仍然够用。
    """
    if len(text) > max_text_length:
        text = text[:max_text_length]

    candidates: List[str] = []
    stack: List[str] = []
    start: Optional[int] = None

    # 只从第一次出现 '{' 的位置开始扫描，以减少无效遍历
    search_start = text.find("{")
    if search_start == -1:
        return []

    for idx, ch in enumerate(text[search_start:], start=search_start):
        if ch == "{":
            if not stack:
                start = idx
            stack.append(ch)
        elif ch == "}" and stack:
            stack.pop()
            if not stack and start is not None:
                # 记录一个完整的 `{...}` 片段
                candidates.append(text[start : idx + 1])
                start = None
                if len(candidates) >= max_candidates:
                    break

    # 按长度降序排序，优先尝试最长的（通常信息更完整）
    candidates.sort(key=len, reverse=True)
    return candidates


def extract_bracket_json(
    text: str,
    max_text_length: int = 100000,
    max_candidates: int = 3,
) -> List[str]:
    """
    使用栈提取所有看起来像 JSON 数组的 `[...]` 片段

    场景：
        - 顶层就是数组的 JSON，例如：
          `[{"a": 1}, {"a": 2}]`
        - 文本中包含数组片段，例如：
          `Here is the result: [{"a": 1}, {"a": 2}] Thanks.`

    说明：
        - 逻辑与 extract_braced_json 基本一致，只是括号改为 '[' / ']'
        - 候选数量一般比对象少一点就够用，因此默认 max_candidates=3
    """
    if len(text) > max_text_length:
        text = text[:max_text_length]

    candidates: List[str] = []
    stack: List[str] = []
    start: Optional[int] = None

    search_start = text.find("[")
    if search_start == -1:
        return []

    for idx, ch in enumerate(text[search_start:], start=search_start):
        if ch == "[":
            if not stack:
                start = idx
            stack.append(ch)
        elif ch == "]" and stack:
            stack.pop()
            if not stack and start is not None:
                candidates.append(text[start : idx + 1])
                start = None
                if len(candidates) >= max_candidates:
                    break

    candidates.sort(key=len, reverse=True)
    return candidates


def clean_json_string(text: str) -> str:
    """
    清理常见的 JSON 格式问题

    当前处理逻辑：
        1. 移除单行注释：// ...
        2. 移除块注释：/* ... */
        3. 移除尾部多余逗号：例如 `{ "a": 1, }` → `{ "a": 1 }`
        4. 移除控制字符（0x00-0x1f, 0x7f-0x9f）

    注意：
        - 没有启用“单引号 → 双引号”的粗暴替换，以避免破坏正常字符串内容。
        - 如果未来需要，可以在严格受控场景下增加更智能的修复策略。
    """
    # 移除单行注释
    text = re.sub(r"//.*?\n", "\n", text)
    # 移除块注释
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    # 移除对象 / 数组末尾多余的逗号
    text = re.sub(r",(\s*[}\]])", r"\1", text)

    # 移除控制字符
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)

    return text.strip()


# ============================================================================
# Pydantic 模型校验辅助函数
# ============================================================================


def _validate_model(data: Any, model_class: Type[T]) -> T:
    """
    使用 Pydantic 模型对数据进行校验并实例化

    参数:
        data:
            JSON 解析后的 Python 对象（通常是 dict / list）
        model_class:
            Pydantic 模型类（包括普通 BaseModel 和 RootModel）

    返回:
        通过校验的 Pydantic 模型实例

    异常:
        ValidationError:
            如果数据与模型 schema 不匹配，则抛出
    """
    try:
        # ⚠️ 修复点：
        # 之前使用 model_class(**data)，在 Pydantic v2 下对 RootModel 无法处理 list 作为根数据。
        # 这里统一使用 model_validate，可以同时支持：
        # - 普通 BaseModel: 字典 -> 模型
        # - RootModel[list[T]] / RootModel[...]：任意 Python 对象作为 root
        return model_class.model_validate(data)  # type: ignore[arg-type]
    except ValidationError as e:
        # 在这里统一打日志，方便调试异常数据
        logger.error(
            "Model validation failed",
            model=model_class.__name__,
            errors=e.errors(),
        )
        raise



# ============================================================================
# 内置 YAML 插件策略（可选：依赖 PyYAML）
# ============================================================================

# 这里尝试导入 PyYAML：
# - 如果环境已安装，则自动注册一个 YAML 策略
# - 如果未安装，不影响现有 JSON 策略的工作
try:
    import yaml  # type: ignore

    def _yaml_fulltext_strategy(text: str, model_class: Type[T]) -> T:
        """
        简单的 YAML 全文解析策略（作为插件示例）

        策略说明：
            - 当所有 JSON 基于策略都失败时，此策略会被调用
            - 使用 yaml.safe_load 尝试将全文解析为 Python 对象
            - 然后交给 Pydantic 模型进行校验

        适用场景：
            - LLM 输出使用 YAML 风格（key: value 列表）而不是严格 JSON
            - 或者用户更习惯写 YAML 配置

        风险与注意事项：
            - YAML 是 JSON 的超集，某些不规范 JSON 文本也可能被 YAML 解释为合法数据，
              因此它被放在所有 JSON 策略之后，只作为“最后一搏”的兜底方案。
        """
        # 如果文本为空或仅包含空白，这里直接抛异常交由上层尝试其他策略或失败
        if not text.strip():
            raise ValueError("Empty text for YAML strategy")

        data = yaml.safe_load(text)
        # data 可能是任意 Python 对象，这里仍然交由 Pydantic 做严格验证
        return _validate_model(data, model_class)

    # 自动注册 YAML 策略到插件列表
    register_extraction_strategy(
        ExtractionStrategy(name="yaml_fulltext", func=_yaml_fulltext_strategy)
    )

    logger.info("YAML extraction strategy registered for structure engine")

except ImportError:
    # 未安装 PyYAML，跳过 YAML 策略注册，不影响 JSON 解析能力
    logger.info("PyYAML not installed, YAML extraction strategy disabled")
