# gecko/core/structure/json_extractor.py
"""
JSON 提取与模型验证模块

负责：
1. 使用多种启发式策略（正则、括号匹配、Markdown块）从脏文本中提取 JSON。
2. 将提取出的数据转换为 Pydantic 模型或原生字典/列表。
3. 提供插件机制扩展解析策略。
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

# [修改] T 不再绑定 BaseModel，允许 dict/list 等任意类型
T = TypeVar("T")


# ============================================================================
# Strategy 插件接口定义
# ============================================================================

@dataclass
class ExtractionStrategy:
    """文本解析策略插件定义"""
    name: str
    func: Callable[[str, Type[T]], T] # type: ignore


# 全局插件策略列表
_EXTRA_STRATEGIES: List[ExtractionStrategy] = []


def register_extraction_strategy(strategy: ExtractionStrategy) -> None:
    """注册新的提取策略"""
    _EXTRA_STRATEGIES.append(strategy)


# ============================================================================
# 主入口函数
# ============================================================================

def extract_structured_data(
    text: str,
    model_class: Type[T],
    *,
    strict: bool = True,
    auto_fix: bool = True,
    max_text_length: int = 100000,
) -> T:
    """
    从文本中提取 JSON 并解析为指定类型（Pydantic 模型或 dict/list）。
    
    执行流程：
    1. 文本预处理与长度检查。
    2. 快速检查（Fast Fail）：如果不包含 '{' 或 '['，直接报错。
    3. 依次尝试内置策略：
       - Direct JSON (整体解析)
       - Markdown Code Block (```json ...```)
       - Braced Object ({...} 栈匹配)
       - Bracket Array ([...] 栈匹配)
       - Cleaned JSON (正则清理后重试)
    4. 尝试已注册的插件策略。
    5. 全部失败则抛出 StructureParseError，包含所有尝试的错误详情。
    """
    # 1. 防御性截断，防止正则 DoS
    if len(text) > max_text_length:
        logger.warning(
            "Text too long for JSON extraction, truncating",
            original_length=len(text),
            max_length=max_text_length,
        )
        text = text[:max_text_length]

    text = text.strip()
    attempts: List[Dict[str, str]] = []

    # 2. 快速失败检查
    if not text or ("{" not in text and "[" not in text):
        raise StructureParseError(
            "Content does not contain JSON-like structure (missing '{' or '[')",
            raw_content=text[:500] if text else "",
        )

    # --- 策略 A: 整体直接解析 ---
    try:
        data = json.loads(text)
        return _validate_model(data, model_class)
    except Exception as e:
        attempts.append({"strategy": "direct_json", "error": str(e)})

    # --- 策略 B: Markdown 代码块提取 ---
    # 匹配 ```json {...} ``` 或 ``` {...} ```
    markdown_pattern = r"```(?:\w+)?\s*([\s\S]*?)```"
    for idx, match in enumerate(re.finditer(markdown_pattern, text)):
        candidate = match.group(1).strip()
        try:
            data = json.loads(candidate)
            return _validate_model(data, model_class)
        except Exception as e:
            if idx < 3: # 仅记录前3次尝试，避免日志过大
                attempts.append({"strategy": f"markdown_{idx}", "error": str(e)})

    # --- 策略 C: 花括号对象提取 ({...}) ---
    obj_candidates = extract_braced_json(text)
    for idx, candidate in enumerate(obj_candidates):
        try:
            data = json.loads(candidate)
            return _validate_model(data, model_class)
        except Exception as e:
            if idx < 3:
                attempts.append({"strategy": f"braced_{idx}", "error": str(e)})

    # --- 策略 C2: 中括号数组提取 ([...]) ---
    array_candidates = extract_bracket_json(text)
    for idx, candidate in enumerate(array_candidates):
        try:
            data = json.loads(candidate)
            return _validate_model(data, model_class)
        except Exception as e:
            if idx < 3:
                attempts.append({"strategy": f"bracket_{idx}", "error": str(e)})

    # --- 策略 D: 简单清理后重试 ---
    if auto_fix:
        cleaned = clean_json_string(text)
        # 只有当清理产生了变化才重试
        if cleaned != text:
            try:
                data = json.loads(cleaned)
                logger.info("Parsed successfully after cleaning", model=model_class.__name__ if hasattr(model_class, "__name__") else "Unknown")
                return _validate_model(data, model_class)
            except Exception as e:
                attempts.append({"strategy": "cleaned_json", "error": str(e)})

    # --- 策略 E: 插件策略 ---
    for strategy in _EXTRA_STRATEGIES:
        try:
            return strategy.func(text, model_class) # type: ignore
        except Exception as e:
            attempts.append({"strategy": f"plugin_{strategy.name}", "error": str(e)})

    # --- 汇总失败 ---
    error_details = "\n".join(
        f"  - {a['strategy']}: {a['error'][:100]}" for a in attempts
    )
    raise StructureParseError(
        f"无法解析为 {getattr(model_class, '__name__', str(model_class))}。尝试了 {len(attempts)} 种策略:\n{error_details}",
        attempts=attempts,
        raw_content=text,
    )


# ============================================================================
# 辅助提取工具
# ============================================================================

def extract_braced_json(text: str, max_text_length: int = 100000, max_candidates: int = 5) -> List[str]:
    """使用栈提取 {...} 结构"""
    if len(text) > max_text_length:
        text = text[:max_text_length]

    candidates: List[str] = []
    stack: List[str] = []
    start: Optional[int] = None

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
                candidates.append(text[start : idx + 1])
                start = None
                if len(candidates) >= max_candidates:
                    break

    candidates.sort(key=len, reverse=True)
    return candidates


def extract_bracket_json(text: str, max_text_length: int = 100000, max_candidates: int = 3) -> List[str]:
    """使用栈提取 [...] 结构"""
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
    """清理 JSON 字符串中的注释和多余逗号"""
    # 移除单行注释
    text = re.sub(r"//.*?\n", "\n", text)
    # 移除块注释
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    # 移除尾部多余逗号
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    # 移除控制字符
    text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
    return text.strip()


# ============================================================================
# 验证逻辑 (包含本次关键修复)
# ============================================================================

def _validate_model(data: Any, model_class: Type[T]) -> T:
    """
    验证并将数据转换为目标类型。
    
    [修复]：增加了对 dict 和 list 的原生支持，防止 Pydantic 抛出 AttributeError。
    """
    # 1. 原生类型检查：如果是 dict 或 list，直接返回数据
    # 这对 StructureEngine 的 LLM 修复逻辑至关重要
    if model_class is dict or model_class is list:
        if isinstance(data, model_class):
            return data # type: ignore
        raise ValueError(f"Data type {type(data)} does not match expected {model_class}")

    # 2. 泛型别名检查 (typing.Dict, typing.List 等)
    # 简单检查 origin 是否为 dict/list
    origin = getattr(model_class, "__origin__", None)
    if origin in (dict, list):
        return data # type: ignore

    # 3. Pydantic 模型校验
    try:
        # Pydantic V2 推荐方式
        return model_class.model_validate(data)  # type: ignore
    except AttributeError:
        # 兼容 dataclass 或 Pydantic V1 (fallback)
        try:
            return model_class(**data) # type: ignore
        except Exception as e:
             raise ValidationError(f"Failed to validate model: {e}") from e
    except ValidationError as e:
        logger.error(
            "Model validation failed",
            model=model_class.__name__,
            errors=e.errors(),
        )
        raise e


# ============================================================================
# YAML 插件 (可选)
# ============================================================================
try:
    import yaml # type: ignore
    
    def _yaml_fulltext_strategy(text: str, model_class: Type[T]) -> T:
        if not text.strip(): 
            raise ValueError("Empty text for YAML strategy")
        data = yaml.safe_load(text)
        return _validate_model(data, model_class)
        
    register_extraction_strategy(
        ExtractionStrategy(name="yaml_fulltext", func=_yaml_fulltext_strategy)
    )
except ImportError:
    pass