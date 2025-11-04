# agno/utils/string_utils.py

"""
字符串处理工具模块

该模块提供了一系列与字符串操作、格式化和解析相关的辅助函数。
主要功能包括：
- ID 和 UUID 的生成与验证 (generate_id, is_valid_uuid)
- 字符串转换与清理 (url_safe_string, remove_indent, prepare_python_code)
- 哈希计算 (hash_string_sha256)
- 从字符串中安全地解析 JSON 对象 (parse_response_model_str)
- 安全的字符串格式化
"""

import hashlib
import json
import re
import string
import uuid
from typing import Optional, Type, List

from pydantic import BaseModel, ValidationError

# 日志配置（兼容独立运行）
import logging
logger = logging.getLogger(__name__)


# --- ID 和 UUID 相关 ---

def is_valid_uuid(uuid_str: str) -> bool:
    """检查字符串是否为有效 UUID。"""
    try:
        uuid.UUID(str(uuid_str))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def generate_id(seed: Optional[str] = None) -> str:
    """生成确定性（UUIDv5）或随机（UUIDv4）ID。"""
    if seed is None:
        return str(uuid.uuid4())
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, seed))


# --- 字符串转换与清理 ---

def url_safe_string(input_string: str) -> str:
    """
    将字符串转换为 kebab-case 的 URL 安全格式。
    """
    if not input_string:
        return ""
    # 替换空格
    s = input_string.replace(" ", "-")
    # 处理 camelCase → kebab-case
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", s).lower()
    # snake_case → kebab-case
    s = s.replace("_", "-")
    # 保留字母、数字、点、破折号
    s = re.sub(r"[^\w\-.]", "", s)
    # 合并多个破折号
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


def remove_indent(s: Optional[str]) -> Optional[str]:
    """移除多行字符串每行的前后空白（保留空行缩进为零）。"""
    if s is None or not isinstance(s, str):
        return s
    return "\n".join(line.strip() for line in s.splitlines())


def prepare_python_code(code: str) -> str:
    """将 LLM 生成的类 JSON 布尔值转换为 Python 常量（true → True 等）。"""
    if not code:
        return code
    replacements = {
        r"\btrue\b": "True",
        r"\bfalse\b": "False",
        r"\bnone\b": "None",
    }
    for pattern, repl in replacements.items():
        code = re.sub(pattern, repl, code, flags=re.IGNORECASE)
    return code


# --- 哈希计算 ---

def hash_string_sha256(input_string: str) -> str:
    """计算字符串的 SHA-256 十六进制哈希。"""
    return hashlib.sha256(input_string.encode("utf-8")).hexdigest()


# --- JSON 解析 ---

def _extract_json_objects(text: str) -> List[str]:
    """
    从文本中提取所有完整的顶层 JSON 对象（基于花括号匹配）。
    支持对象和数组（如 {...} 或 [...]）。
    """
    if not text:
        return []
    results: List[str] = []
    depth = 0
    start = None
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if not in_string:
            if ch == '"':
                in_string = True
            elif ch in "{[":
                if depth == 0:
                    start = i
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0 and start is not None:
                    results.append(text[start : i + 1])
                    start = None
        else:
            if not escaped and ch == '"':
                in_string = False
            elif ch == "\\":
                escaped = not escaped
            else:
                escaped = False
    return results


def _clean_json_content(content: str) -> str:
    """
    清理包含 Markdown 代码块的 JSON 字符串。
    - 支持 ```json ... ``` 和 ``` ... ```
    - 移除 Markdown 强调符（如 *、`）
    - 规范化空白字符
    """
    if not isinstance(content, str):
        raise TypeError(f"_clean_json_content expected str, got {type(content)}")

    cleaned = content.strip()

    # 提取 ```json 或 ``` 代码块内容
    if cleaned.startswith("```"):
        # 移除开头的 ```
        parts = cleaned.split("```", 2)
        if len(parts) >= 3:
            # ```json\n{...}\n```
            inner = parts[1]
            if inner.lower().startswith("json"):
                inner = inner[4:].lstrip()
            cleaned = inner
        elif len(parts) == 2:
            # ```{...}
            cleaned = parts[1]
        else:
            # 只有 ```
            cleaned = ""

    # 移除键名周围的 Markdown 标记（如 *"key"*）
    cleaned = re.sub(r'[*`#]?"([A-Za-z0-9_-]+)"[*`#]?', r'"\1"', cleaned)

    # 规范化空白：换行/回车 → 空格，多个空格 → 单空格
    cleaned = re.sub(r"[\r\n\t]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()


def parse_response_model_str(content: str, output_schema: Type[BaseModel]) -> Optional[BaseModel]:
    """
    从 LLM 输出中稳健解析 Pydantic 模型。
    策略：
      1. 直接解析清理后的完整 JSON。
      2. 提取并解析单个 JSON 对象。
      3. 合并多个 JSON 片段中的字段。
    """
    try:
        cleaned = _clean_json_content(content)

        # 策略 1: 直接解析
        try:
            return output_schema.model_validate_json(cleaned)
        except (ValidationError, json.JSONDecodeError):
            pass

        # 策略 2 & 3: 提取 JSON 对象
        candidates = _extract_json_objects(cleaned)
        if not candidates:
            logger.debug("No JSON objects found in content.")
            return None

        # 单个对象直接解析
        if len(candidates) == 1:
            try:
                return output_schema.model_validate_json(candidates[0])
            except (ValidationError, json.JSONDecodeError) as e:
                logger.debug(f"Failed to parse single JSON candidate: {e}")
                return None

        # 多个对象：按字段合并
        merged: dict = {}
        model_fields = set(output_schema.model_fields.keys())

        for cand in candidates:
            try:
                obj = json.loads(cand)
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k in model_fields and k not in merged:
                            merged[k] = v
            except json.JSONDecodeError:
                continue

        if not merged:
            logger.debug("No valid fields found after merging JSON objects.")
            return None

        return output_schema.model_validate(merged)

    except Exception as e:
        logger.warning(f"Unexpected error in parse_response_model_str: {e}")
        return None


# --- 安全的字符串格式化 ---

class SafeFormatter(string.Formatter):
    """
    安全的字符串格式化器，对缺失键或无效格式符保持原样。
    """
    def get_field(self, field_name, args, kwargs):
        try:
            return super().get_field(field_name, args, kwargs)
        except (KeyError, IndexError):
            return f"{{{field_name}}}", None

    def format_field(self, value, format_spec):
        try:
            return super().format_field(value, format_spec)
        except ValueError:
            return f"{{{value}:{format_spec}}}"


# --- 测试代码 ---
if __name__ == "__main__":
    print("--- 正在运行 agno/utils/string_utils.py 的测试代码 ---")

    # 1. UUID
    print("\n[1] UUID 相关:")
    u1, u2 = generate_id("seed"), generate_id("seed")
    print(f"确定性: {u1 == u2}, 有效: {is_valid_uuid(u1)}")

    # 2. URL 安全
    print("\n[2] URL 安全格式:")
    for s in ["My Agent", "myAgent", "my_agent", "Agent (V1.0)!"]:
        print(f"  {s!r} → {url_safe_string(s)!r}")

    # 3. 移除缩进
    print("\n[3] 移除缩进:")
    print(repr(remove_indent("  a\n    b\n  c")))

    # 4. Python 代码修正
    print("\n[4] 修正 Python 代码:")
    print(prepare_python_code("if x == true: return none"))

    # 5. 哈希
    print(f"\n[5] SHA256: {hash_string_sha256('hello')}")

    # 6. JSON 解析
    print("\n[6] JSON 解析测试:")
    from pydantic import BaseModel

    class UserInfo(BaseModel):
        name: str
        age: int

    test_cases = [
        '```json\n{"name": "Alice", "age": 30}\n```',
        '文本 {"name": "Bob"} 和 {"age": 25} 文本',
        '{"name": "Charlie", "age": 40, "extra": true}',
        '{"name": "Invalid", "age": "not-int"}',
    ]
    for case in test_cases:
        result = parse_response_model_str(case, UserInfo)
        print(f"  {case[:30]!r} → {result}")

    # 7. 安全格式化
    print("\n[7] 安全格式化:")
    fmt = SafeFormatter()
    out = fmt.format("Hello {name}! ID: {id}", name="Alice")
    print(f"  {out!r}")

    print("\n--- 测试结束 ---")