# gecko/core/structure/schema.py
"""
Schema 工具模块

职责：
- 将 Pydantic 模型转换为 OpenAI Function Calling / tools 所需的 schema
- 计算实际数据与模型 schema 的差异（缺失字段、多余字段、简单类型不匹配）

注意：
- 这里只做“schema 层”的工具函数，不参与具体的 JSON 提取流程。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from pydantic import BaseModel


def to_openai_tool(model: type[BaseModel]) -> Dict[str, Any]:
    """
    将 Pydantic 模型转换为 OpenAI Function Calling 所需的 tool schema

    参数:
        model: Pydantic 模型类

    返回:
        OpenAI tool 定义字典，例如：
        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": {...},  # JSON Schema
            },
        }

    设计要点：
        - 使用 Pydantic v2 的 `model_json_schema()` 生成 JSON Schema
        - 将 title 中的非单词字符去掉，生成合法的工具名称
        - 展开 `$defs` / `$ref`，生成更“扁平化”的 schema，方便 OpenAI 使用
    """
    schema = model.model_json_schema()

    # 提取模型名称（移除特殊字符，转换为小写）
    name = re.sub(r"\W+", "_", schema.get("title", "extract_data")).lower()

    # 去掉 title，工具 schema 中一般用不到
    schema.pop("title", None)

    # 如果模型包含子定义（$defs），则尝试展开 $ref 引用
    if "$defs" in schema:
        schema = _flatten_schema(schema)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": schema.get("description", f"Extract {name} data"),
            "parameters": schema,
        },
    }


def _flatten_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    展开 schema 中的 $ref 引用（简化版实现）

    说明：
        - Pydantic 通常会把一些嵌套模型放在 `$defs` 中，并通过 `$ref` 引用。
        - 为了更好地兼容 OpenAI 的 tools，我们希望尽量把这些引用“展开”成一个
          更扁平的 JSON Schema。

    处理策略：
        - 支持 "$defs" + "$ref" 最常见的模式
        - 对于 { "$ref": "#/$defs/Xxx", <其他字段> }：
          1) 先递归解析 "$defs.Xxx"
          2) 将当前节点除 "$ref" 外的字段与目标 dict 做“浅合并”，保留额外信息
        - 不完整支持 allOf/anyOf/oneOf 等高级特性，仅覆盖普通 Pydantic 模型场景
    """
    defs = schema.pop("$defs", {})
    if not defs:
        return schema

    def resolve_ref(obj: Any) -> Any:
        # 递归解析对象里的 $ref
        if isinstance(obj, dict):
            if "$ref" in obj:
                ref_key = obj["$ref"].split("/")[-1]
                if ref_key in defs:
                    # 递归解析被引用定义
                    target = resolve_ref(defs[ref_key])
                    # 当前节点中除了 $ref 外，其余字段视为“补充信息”
                    extra = {k: v for k, v in obj.items() if k != "$ref"}
                    if isinstance(target, dict):
                        # 被引用定义优先，然后由当前节点覆盖
                        merged = {**target, **extra}
                        return resolve_ref(merged)
                    return target
                # 找不到对应 defs 时，退化为普通 dict 递归
            # 普通 dict：递归处理子字段
            return {k: resolve_ref(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [resolve_ref(item) for item in obj]
        return obj

    return resolve_ref(schema)  # type: ignore[return-value]


def get_schema_diff(
    data: Dict[str, Any],
    model_class: type[BaseModel],
) -> Dict[str, Any]:
    """
    比较数据与模型 Schema 的差异（基础版）

    参数:
        data: 实际数据（通常是 JSON 解析后的 dict）
        model_class: 期望的 Pydantic 模型

    返回:
        差异信息字典，包含：
            - missing_required: 缺失的必填字段名列表
            - extra_fields: 数据中多余的字段名列表
            - type_mismatches: 简单字段类型不匹配列表
    """
    schema = model_class.model_json_schema()
    required = set(schema.get("required", []))
    properties = schema.get("properties", {})

    data_keys = set(data.keys())
    schema_keys = set(properties.keys())

    return {
        "missing_required": list(required - data_keys),
        "extra_fields": list(data_keys - schema_keys),
        "type_mismatches": _check_type_mismatches(data, properties),
    }


def _check_type_mismatches(
    data: Dict[str, Any],
    properties: Dict[str, Any],
) -> List[Dict[str, str]]:
    """
    检查顶层字段类型不匹配（简化版）

    说明：
        - 仅检查顶层字段（不递归数组元素或嵌套对象）
        - 类型信息来源于 JSON Schema 中的 "type" 字段
        - 结果主要用于诊断和日志，不作为硬性约束
    """
    mismatches: List[Dict[str, str]] = []

    for key, value in data.items():
        if key not in properties:
            continue

        expected_type = properties[key].get("type")
        actual_type = type(value).__name__

        # 简化 JSON Schema type -> Python 类型映射
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        expected_python_type = type_map.get(expected_type)
        if expected_python_type and not isinstance(value, expected_python_type):
            mismatches.append(
                {
                    "field": key,
                    "expected": expected_type,
                    "actual": actual_type,
                }
            )

    return mismatches
