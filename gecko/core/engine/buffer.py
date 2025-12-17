# gecko/core/engine/buffer.py
"""
流式响应缓冲区模块
==================

这个模块实现了 StreamBuffer 类，用于处理 LLM 流式响应的增量数据。

主要功能：
1. 增量式累积文本内容
2. 解析和组装工具调用（tool calls）
3. 处理不完整的 JSON 数据
4. 防止数据溢出和内存泄漏
5. 构建完整的 Message 对象

使用场景：
- 处理 OpenAI/Claude 等模型的流式响应
- 在流式输出过程中逐步构建完整消息
- 处理工具调用的增量片段并正确组装
"""

from __future__ import annotations

import json
import re
import threading
from typing import Any, Dict, List, Optional

from gecko.core.logging import get_logger
from gecko.core.message import Message

logger = get_logger(__name__)

# ============================================================================
# 正则表达式模式
# ============================================================================
# 匹配 JSON 对象中的尾随逗号：如 {"a": 1, }
_RE_TRAILING_COMMA_OBJ = re.compile(r",\s*}")
# 匹配 JSON 数组中的尾随逗号：如 [1, 2, ]
_RE_TRAILING_COMMA_ARR = re.compile(r",\s*\]")
# 匹配 Markdown 代码块中的 JSON：如 ```json {...} ```
_RE_MARKDOWN_JSON = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")

# ============================================================================
# 安全类型限制
# ============================================================================
# 允许在 literal_eval 中使用的类型，防止代码注入
_ALLOWED_LITERAL_TYPES = (dict, list, str, int, float, bool, type(None))


class StreamBuffer:
    """
    流式响应缓冲区
    
    这个类用于累积和处理来自 LLM 的流式响应数据。它能够：
    - 处理增量文本内容
    - 解析和组装工具调用
    - 清理不完整或格式错误的 JSON
    - 构建完整的 Message 对象
    
    线程安全：所有公共方法都使用锁保护
    
    内存保护：
    - 限制内容总字符数
    - 限制单个工具参数大小
    - 限制工具索引范围
    
    使用 __slots__ 优化内存占用
    """
    
    __slots__ = (
        "content_parts",           # 文本内容片段列表
        "_content_len",            # 当前内容总长度
        "tool_calls_map",          # 工具调用映射表 {index: tool_call_data}
        "_max_tool_index",         # 当前最大工具索引
        "_args_len_map",           # 参数长度映射表 {index: length}
        "_max_content_chars",      # 内容最大字符数限制
        "_max_argument_chars",     # 单个工具参数最大字符数限制
        "_max_tool_index_limit",   # 工具索引上限
        "_lock"                    # 线程锁
    )

    def __init__(
        self,
        max_content_chars: int = 200_000,
        max_argument_chars: int = 100_000,
        max_tool_index: int = 1000,
    ):
        """
        初始化流式缓冲区
        
        参数:
            max_content_chars: 内容最大字符数（默认 200K）
            max_argument_chars: 单个工具参数最大字符数（默认 100K）
            max_tool_index: 最大工具索引（默认 1000）
        """
        # 文本内容存储
        self.content_parts: List[str] = []
        self._content_len: int = 0

        # 工具调用存储（使用索引作为键）
        self.tool_calls_map: Dict[int, Dict[str, Any]] = {}
        self._max_tool_index: int = -1

        # 每个工具的参数累积长度
        self._args_len_map: Dict[int, int] = {}

        # 限制参数
        self._max_content_chars: int = int(max_content_chars)
        self._max_argument_chars: int = int(max_argument_chars)
        self._max_tool_index_limit: int = int(max_tool_index)

        # 线程安全锁
        self._lock: threading.RLock = threading.RLock()

    def add_chunk(self, chunk: Any) -> Optional[str]:
        """
        添加流式响应的一个数据块
        
        这个方法是处理流式数据的核心入口。它会：
        1. 提取 delta（增量数据）
        2. 处理文本内容
        3. 处理工具调用
        
        参数:
            chunk: 来自 LLM 的原始响应块（可能是对象或字典）
            
        返回:
            新增的文本内容（如果有），否则返回 None
        """
        with self._lock:
            # 从响应块中提取 delta 数据
            delta = self._extract_delta(chunk)
            if delta is None:
                logger.debug("StreamChunk cannot extract delta, skipping", chunk_type=type(chunk).__name__)
                return None

            new_content: Optional[str] = None

            # 处理文本内容
            content = delta.get("content")
            if isinstance(content, str) and content:
                added = self._add_content(content)
                if added:
                    new_content = added

            # 处理工具调用
            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                self._add_tool_calls(tool_calls)

            return new_content

    def _extract_delta(self, chunk: Any) -> Optional[Dict[str, Any]]:
        """
        从响应块中提取 delta（增量数据）
        
        支持多种格式：
        1. 直接包含 delta 属性的对象
        2. 包含 choices[0].delta 的对象
        3. 字典格式的上述结构
        
        参数:
            chunk: 原始响应块
            
        返回:
            提取的 delta 字典，失败返回 None
        """
        # 尝试 1：直接获取 delta 属性
        delta = getattr(chunk, "delta", None)
        if isinstance(delta, dict):
            return delta

        # 尝试 2：通过 choices 获取
        choices = getattr(chunk, "choices", None)
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            if isinstance(c0, dict):
                # 字典格式
                d = c0.get("delta")
                if isinstance(d, dict):
                    return d
            else:
                # 对象格式
                d = getattr(c0, "delta", None)
                if isinstance(d, dict):
                    return d

        # 尝试 3：chunk 本身是字典
        if isinstance(chunk, dict):
            d = chunk.get("delta")
            if isinstance(d, dict):
                return d
            
            # 检查字典中的 choices
            choices = chunk.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                d = choices[0].get("delta")
                if isinstance(d, dict):
                    return d

        return None

    def _add_content(self, content: str) -> str:
        """
        添加文本内容片段
        
        实现了内容长度限制，防止内存溢出。
        
        参数:
            content: 要添加的文本片段
            
        返回:
            实际添加的内容（可能被截断）
        """
        incoming_len = len(content)

        # 检查是否超出限制
        if self._content_len + incoming_len > self._max_content_chars:
            logger.warning(
                "Content exceeds limit, truncating",
                current_len=self._content_len,
                incoming_len=incoming_len,
                limit=self._max_content_chars,
            )
            # 计算还能添加多少字符
            allowed = max(0, self._max_content_chars - self._content_len)
            if allowed <= 0:
                return ""
            # 截断内容
            truncated = content[:allowed]
            self.content_parts.append(truncated)
            self._content_len += len(truncated)
            return truncated

        # 正常添加
        self.content_parts.append(content)
        self._content_len += incoming_len
        return content

    def _add_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> None:
        """
        添加工具调用增量数据
        
        工具调用可能分多个块发送，这个方法负责：
        1. 按索引组织工具调用
        2. 增量式累积参数
        3. 检测异常情况（索引跳跃、ID 变化等）
        
        参数:
            tool_calls: 工具调用增量列表
        """
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue

            # 获取工具调用的索引（用于区分多个并行工具调用）
            idx = tc.get("index")
            if not isinstance(idx, int):
                continue

            # 安全检查：索引合法性
            if idx < 0:
                logger.warning("Received negative tool index, skipping", idx=idx)
                continue
            if idx > self._max_tool_index_limit:
                logger.warning("Tool index exceeds limit, skipping", idx=idx, limit=self._max_tool_index_limit)
                continue

            # 检测异常大的索引跳跃（可能表示数据错误）
            if self._max_tool_index >= 0:
                gap = idx - self._max_tool_index
                if gap > 500:
                    logger.warning(
                        "Detected abnormally large tool index gap",
                        prev_max=self._max_tool_index,
                        new_idx=idx,
                        gap=gap,
                    )

            # 更新最大索引
            if idx > self._max_tool_index:
                self._max_tool_index = idx

            # 如果是新的工具调用索引，初始化数据结构
            if idx not in self.tool_calls_map:
                self.tool_calls_map[idx] = {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
                self._args_len_map[idx] = 0

            target = self.tool_calls_map[idx]

            # 处理工具调用 ID（只在第一次收到时设置）
            tc_id = tc.get("id")
            if isinstance(tc_id, str) and tc_id:
                if not target.get("id"):
                    target["id"] = tc_id
                else:
                    # 检测 ID 变化（异常情况）
                    if target["id"] != tc_id:
                        logger.warning(
                            "Detected tool_call id change for same index, using latest",
                            index=idx,
                            old_id=target["id"],
                            new_id=tc_id,
                        )
                        target["id"] = tc_id

            # 处理函数信息（名称和参数）
            func = tc.get("function")
            if isinstance(func, dict):
                self._merge_function(idx, target["function"], func)

    def _merge_function(self, idx: int, target: Dict[str, str], incoming: Dict[str, Any]) -> None:
        """
        合并函数调用信息
        
        处理函数名称和参数的增量更新。
        参数会被增量式累积，实现了长度限制。
        
        参数:
            idx: 工具调用索引
            target: 目标函数信息字典
            incoming: 新到达的函数信息
        """
        # 处理函数名称（通常只在第一个块中出现）
        inc_name = incoming.get("name")
        if isinstance(inc_name, str) and inc_name:
            target["name"] = inc_name

        # 处理函数参数（JSON 字符串，可能分多个块发送）
        inc_args = incoming.get("arguments")
        if not (isinstance(inc_args, str) and inc_args):
            return

        current_len = self._args_len_map.get(idx, 0)
        incoming_len = len(inc_args)

        # 检查参数长度限制
        if current_len + incoming_len > self._max_argument_chars:
            logger.warning(
                "Tool arguments exceed limit, truncating",
                index=idx,
                current_len=current_len,
                incoming_len=incoming_len,
                limit=self._max_argument_chars,
            )
            # 计算还能添加多少字符
            allowed = max(0, self._max_argument_chars - current_len)
            if allowed <= 0:
                return
            to_add = inc_args[:allowed]
        else:
            to_add = inc_args

        # 累积参数字符串
        target["arguments"] += to_add
        self._args_len_map[idx] = current_len + len(to_add)

    def build_message(self) -> Message:
        """
        构建完整的 Message 对象
        
        这是缓冲区的最终输出方法。它会：
        1. 合并所有文本片段
        2. 验证和清理工具调用数据
        3. 清理 JSON 参数
        4. 构建标准的 Message 对象
        
        返回:
            包含完整内容和工具调用的 Message 对象
        """
        with self._lock:
            # 合并所有文本片段
            full_content = "".join(self.content_parts)

            # 处理工具调用
            tool_calls_list: List[Dict[str, Any]] = []

            # 按索引顺序处理工具调用
            for idx in sorted(self.tool_calls_map.keys()):
                raw_tc = self.tool_calls_map[idx]
                func = raw_tc.get("function", {})
                func_name = (func.get("name") or "").strip() if isinstance(func, dict) else ""

                # 跳过没有名称的工具调用（可能是不完整的数据）
                if not func_name:
                    logger.warning("Tool call missing name, skipping", index=idx)
                    continue

                # 获取参数字符串
                raw_args = ""
                if isinstance(func, dict):
                    raw_args = (func.get("arguments") or "").strip()

                # 空参数使用空对象
                if not raw_args:
                    raw_args = "{}"

                # 清理和验证 JSON 参数
                cleaned_args = self._clean_arguments(raw_args)

                # 构建标准格式的工具调用
                tc_out = {
                    "id": str(raw_tc.get("id") or ""),
                    "type": raw_tc.get("type", "function"),
                    "function": {
                        "name": func_name,
                        "arguments": cleaned_args,
                    },
                }
                tool_calls_list.append(tc_out)

            if tool_calls_list:
                logger.debug("Message build completed", tool_calls=len(tool_calls_list))

            # 创建 assistant 角色的消息
            return Message.assistant(
                content=full_content,
                tool_calls=tool_calls_list if tool_calls_list else None,
            )

    def _clean_arguments(self, raw_json: str) -> str:
        """
        清理和修复 JSON 参数字符串
        
        处理常见的 JSON 格式问题：
        1. Markdown 代码块包裹
        2. 外层引号包裹
        3. 尾随逗号
        4. 单引号代替双引号
        
        参数:
            raw_json: 原始 JSON 字符串（可能格式不正确）
            
        返回:
            清理后的有效 JSON 字符串，失败返回 "{}"
        """
        if not raw_json:
            return "{}"

        s = raw_json.strip()

        # 尝试直接解析（最常见情况）
        try:
            json.loads(s)
            return s
        except json.JSONDecodeError:
            pass

        # 处理 Markdown 代码块：```json {...} ```
        if s.startswith("```"):
            m = _RE_MARKDOWN_JSON.search(s)
            if m:
                s = m.group(1).strip()

        # 处理外层引号包裹：'{"a": 1}' 或 "{"a": 1}"
        if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
            inner = s[1:-1].strip()
            if inner.startswith("{") or inner.startswith("["):
                s = inner

        # 移除尾随逗号
        s = _RE_TRAILING_COMMA_OBJ.sub("}", s)
        s = _RE_TRAILING_COMMA_ARR.sub("]", s)

        # 处理单引号：如果有单引号但没有双引号，可能是 Python 字面量
        if "'" in s and '"' not in s:
            try:
                parsed = self._safe_literal_eval(s)
                if isinstance(parsed, (dict, list)):
                    s = json.dumps(parsed, ensure_ascii=False)
            except (ValueError, SyntaxError):
                pass

        # 最后尝试解析
        try:
            json.loads(s)
            return s
        except json.JSONDecodeError as e:
            logger.error(
                "JSON arguments cleanup failed, returning empty object",
                original=(raw_json[:200] + "...") if len(raw_json) > 200 else raw_json,
                parse_error=str(e),
            )
            return "{}"

    def _safe_literal_eval(self, expr: str) -> Any:
        """
        安全地评估 Python 字面量表达式
        
        这是 ast.literal_eval 的安全版本，只允许特定类型。
        用于处理单引号 JSON（Python 字面量格式）。
        
        参数:
            expr: Python 字面量表达式字符串
            
        返回:
            解析后的 Python 对象
            
        抛出:
            ValueError: 表达式无效或包含不允许的类型
        """
        import ast
        
        # 解析表达式为 AST
        try:
            node = ast.parse(expr, mode='eval')
        except SyntaxError as e:
            raise ValueError(f"Invalid expression: {e}")
        
        def _validate_node(n: ast.AST) -> Any:
            """
            递归验证和评估 AST 节点
            
            只允许安全的字面量类型，拒绝函数调用、变量引用等。
            """
            if isinstance(n, ast.Expression):
                return _validate_node(n.body)
            elif isinstance(n, ast.Constant):
                # 常量：字符串、数字、布尔值、None
                if type(n.value) not in _ALLOWED_LITERAL_TYPES:
                    raise ValueError(f"Unsupported type: {type(n.value)}")
                return n.value
            elif isinstance(n, (ast.List, ast.Tuple)):
                # 列表和元组
                return [_validate_node(el) for el in n.elts]
            elif isinstance(n, ast.Dict):
                # 字典
                keys = []
                for k in n.keys:
                    if k is None:
                        raise ValueError("Dict unpacking not supported")
                    keys.append(_validate_node(k))
                values = [_validate_node(v) for v in n.values]
                return dict(zip(keys, values))
            elif isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
                # 一元运算符：+x 或 -x（只允许用于数字）
                operand = _validate_node(n.operand)
                if not isinstance(operand, (int, float)):
                    raise ValueError("Unary operators only allowed on numbers")
                return -operand if isinstance(n.op, ast.USub) else operand
            else:
                # 拒绝其他类型（函数调用、变量等）
                raise ValueError(f"Unsupported node type: {type(n).__name__}")
        
        return _validate_node(node)

    def reset(self) -> None:
        """
        重置缓冲区到初始状态
        
        清空所有累积的数据，可以重新开始处理新的流。
        """
        with self._lock:
            self.content_parts.clear()
            self.tool_calls_map.clear()
            self._args_len_map.clear()
            self._max_tool_index = -1
            self._content_len = 0

    def get_current_content(self) -> str:
        """
        获取当前累积的文本内容
        
        返回:
            合并后的完整文本
        """
        with self._lock:
            return "".join(self.content_parts)

    def get_tool_call_count(self) -> int:
        """
        获取当前工具调用数量
        
        返回:
            工具调用的个数
        """
        with self._lock:
            return len(self.tool_calls_map)

    def __repr__(self) -> str:
        """
        字符串表示
        
        返回:
            缓冲区状态的可读表示
        """
        return (
            f"StreamBuffer("
            f"content_length={self._content_len}, "
            f"tool_calls={self.get_tool_call_count()}"
            f")"
        )


# ============================================================================
# 模块导出
# ============================================================================
__all__ = ["StreamBuffer"]