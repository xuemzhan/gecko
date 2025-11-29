# gecko/core/structure/sync.py
"""
同步封装与轻量工具函数模块

职责：
- 提供同步版本的结构化解析函数 `parse_structured_output`，方便在纯同步脚本中使用
- 提供轻量级 JSON 提取函数 `extract_json_from_text`，用于简单场景

注意：
- 同步封装在已有事件循环中使用会抛出 RuntimeError，以避免错误地嵌套事件循环。
"""

from __future__ import annotations

import json
import asyncio
import re
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from .engine import StructureEngine
from .json_extractor import extract_braced_json, extract_bracket_json

T = TypeVar("T", bound=BaseModel)


def parse_structured_output(
    content: str,
    model_class: Type[T],
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> T:
    """
    同步版本的结构化输出解析（便捷函数）

    使用场景：
        - 在纯脚本 / 同步程序中使用结构化解析，不方便引入 async/await 时
        - 自动处理事件循环的创建与回收

    安全性考虑：
        - 如果当前存在正在运行的事件循环（例如在 FastAPI / Jupyter 中），
          本函数会抛出 RuntimeError，提示调用方改用异步接口：
          `await StructureEngine.parse(...)`
    """
    try:
        # 如果当前有运行中的事件循环，会抛 RuntimeError
        asyncio.get_running_loop()
    except RuntimeError:
        # 没有运行中的事件循环，可以安全地使用 asyncio.run
        return asyncio.run(
            StructureEngine.parse(
                content=content,
                model_class=model_class,
                raw_tool_calls=tool_calls,
            )
        )
    else:
        # 已在异步环境中，禁止同步封装，避免嵌套事件循环
        raise RuntimeError(
            "parse_structured_output() 不能在已有事件循环中调用，请使用 "
            "`await StructureEngine.parse(...)`。"
        )


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    从文本中提取第一个“看起来可靠”的 JSON 对象（轻量工具）

    参数:
        text: 文本内容（通常是 LLM 输出）

    返回:
        - 成功时返回 dict
        - 如果未找到合适的 JSON 对象，则返回 None

    实现策略（比 StructureEngine.parse 更轻量）：
        1. 直接整体 json.loads，且要求结果为 dict
        2. 从 ```json ...``` / ``` ...``` 代码块中尝试解析
        3. 使用 extract_braced_json 提取 `{...}` 片段并尝试解析
        4. 使用 extract_bracket_json 提取 `[...]` 片段：
           - 如果数组元素为对象，则返回第一个对象
    """
    text = (text or "").strip()
    if not text:
        return None

    # 1) 尝试整体解析
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # 2) 尝试 Markdown 代码块
    pattern = r"```(?:json)?\s*([\s\S]*?)```"
    for match in re.finditer(pattern, text):
        try:
            obj = json.loads(match.group(1).strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    # 3) 尝试 `{...}` 片段
    obj_candidates = extract_braced_json(text)
    for candidate in obj_candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    # 4) 尝试 `[...]` 片段（数组），返回第一个 dict 元素
    array_candidates = extract_bracket_json(text)
    for candidate in array_candidates:
        try:
            arr = json.loads(candidate)
            if isinstance(arr, list) and arr and isinstance(arr[0], dict):
                return arr[0]
        except json.JSONDecodeError:
            continue

    return None
