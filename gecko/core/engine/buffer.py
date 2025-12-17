# gecko/core/engine/buffer.py

from __future__ import annotations

import json
import re
import threading
from typing import Any, Dict, List, Optional

from gecko.core.logging import get_logger
from gecko.core.message import Message

logger = get_logger(__name__)

_RE_TRAILING_COMMA_OBJ = re.compile(r",\s*}")
_RE_TRAILING_COMMA_ARR = re.compile(r",\s*\]")
_RE_MARKDOWN_JSON = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")

_ALLOWED_LITERAL_TYPES = (dict, list, str, int, float, bool, type(None))


class StreamBuffer:
    __slots__ = (
        "content_parts", "_content_len", "tool_calls_map", "_max_tool_index",
        "_args_len_map", "_max_content_chars", "_max_argument_chars",
        "_max_tool_index_limit", "_lock"
    )

    def __init__(
        self,
        max_content_chars: int = 200_000,
        max_argument_chars: int = 100_000,
        max_tool_index: int = 1000,
    ):
        self.content_parts: List[str] = []
        self._content_len: int = 0

        self.tool_calls_map: Dict[int, Dict[str, Any]] = {}
        self._max_tool_index: int = -1

        self._args_len_map: Dict[int, int] = {}

        self._max_content_chars: int = int(max_content_chars)
        self._max_argument_chars: int = int(max_argument_chars)
        self._max_tool_index_limit: int = int(max_tool_index)

        self._lock: threading.RLock = threading.RLock()

    def add_chunk(self, chunk: Any) -> Optional[str]:
        with self._lock:
            delta = self._extract_delta(chunk)
            if delta is None:
                logger.debug("StreamChunk cannot extract delta, skipping", chunk_type=type(chunk).__name__)
                return None

            new_content: Optional[str] = None

            content = delta.get("content")
            if isinstance(content, str) and content:
                added = self._add_content(content)
                if added:
                    new_content = added

            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                self._add_tool_calls(tool_calls)

            return new_content

    def _extract_delta(self, chunk: Any) -> Optional[Dict[str, Any]]:
        delta = getattr(chunk, "delta", None)
        if isinstance(delta, dict):
            return delta

        choices = getattr(chunk, "choices", None)
        if isinstance(choices, list) and choices:
            c0 = choices[0]
            if isinstance(c0, dict):
                d = c0.get("delta")
                if isinstance(d, dict):
                    return d
            else:
                d = getattr(c0, "delta", None)
                if isinstance(d, dict):
                    return d

        if isinstance(chunk, dict):
            d = chunk.get("delta")
            if isinstance(d, dict):
                return d
            
            choices = chunk.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                d = choices[0].get("delta")
                if isinstance(d, dict):
                    return d

        return None

    def _add_content(self, content: str) -> str:
        incoming_len = len(content)

        if self._content_len + incoming_len > self._max_content_chars:
            logger.warning(
                "Content exceeds limit, truncating",
                current_len=self._content_len,
                incoming_len=incoming_len,
                limit=self._max_content_chars,
            )
            allowed = max(0, self._max_content_chars - self._content_len)
            if allowed <= 0:
                return ""
            truncated = content[:allowed]
            self.content_parts.append(truncated)
            self._content_len += len(truncated)
            return truncated

        self.content_parts.append(content)
        self._content_len += incoming_len
        return content

    def _add_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> None:
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue

            idx = tc.get("index")
            if not isinstance(idx, int):
                continue

            if idx < 0:
                logger.warning("Received negative tool index, skipping", idx=idx)
                continue
            if idx > self._max_tool_index_limit:
                logger.warning("Tool index exceeds limit, skipping", idx=idx, limit=self._max_tool_index_limit)
                continue

            if self._max_tool_index >= 0:
                gap = idx - self._max_tool_index
                if gap > 500:
                    logger.warning(
                        "Detected abnormally large tool index gap",
                        prev_max=self._max_tool_index,
                        new_idx=idx,
                        gap=gap,
                    )

            if idx > self._max_tool_index:
                self._max_tool_index = idx

            if idx not in self.tool_calls_map:
                self.tool_calls_map[idx] = {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
                self._args_len_map[idx] = 0

            target = self.tool_calls_map[idx]

            tc_id = tc.get("id")
            if isinstance(tc_id, str) and tc_id:
                if not target.get("id"):
                    target["id"] = tc_id
                else:
                    if target["id"] != tc_id:
                        logger.warning(
                            "Detected tool_call id change for same index, using latest",
                            index=idx,
                            old_id=target["id"],
                            new_id=tc_id,
                        )
                        target["id"] = tc_id

            func = tc.get("function")
            if isinstance(func, dict):
                self._merge_function(idx, target["function"], func)

    def _merge_function(self, idx: int, target: Dict[str, str], incoming: Dict[str, Any]) -> None:
        inc_name = incoming.get("name")
        if isinstance(inc_name, str) and inc_name:
            target["name"] = inc_name

        inc_args = incoming.get("arguments")
        if not (isinstance(inc_args, str) and inc_args):
            return

        current_len = self._args_len_map.get(idx, 0)
        incoming_len = len(inc_args)

        if current_len + incoming_len > self._max_argument_chars:
            logger.warning(
                "Tool arguments exceed limit, truncating",
                index=idx,
                current_len=current_len,
                incoming_len=incoming_len,
                limit=self._max_argument_chars,
            )
            allowed = max(0, self._max_argument_chars - current_len)
            if allowed <= 0:
                return
            to_add = inc_args[:allowed]
        else:
            to_add = inc_args

        target["arguments"] += to_add
        self._args_len_map[idx] = current_len + len(to_add)

    def build_message(self) -> Message:
        with self._lock:
            full_content = "".join(self.content_parts)

            tool_calls_list: List[Dict[str, Any]] = []

            for idx in sorted(self.tool_calls_map.keys()):
                raw_tc = self.tool_calls_map[idx]
                func = raw_tc.get("function", {})
                func_name = (func.get("name") or "").strip() if isinstance(func, dict) else ""

                if not func_name:
                    logger.warning("Tool call missing name, skipping", index=idx)
                    continue

                raw_args = ""
                if isinstance(func, dict):
                    raw_args = (func.get("arguments") or "").strip()

                if not raw_args:
                    raw_args = "{}"

                cleaned_args = self._clean_arguments(raw_args)

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

            return Message.assistant(
                content=full_content,
                tool_calls=tool_calls_list if tool_calls_list else None,
            )

    def _clean_arguments(self, raw_json: str) -> str:
        if not raw_json:
            return "{}"

        s = raw_json.strip()

        try:
            json.loads(s)
            return s
        except json.JSONDecodeError:
            pass

        if s.startswith("```"):
            m = _RE_MARKDOWN_JSON.search(s)
            if m:
                s = m.group(1).strip()

        if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
            inner = s[1:-1].strip()
            if inner.startswith("{") or inner.startswith("["):
                s = inner

        s = _RE_TRAILING_COMMA_OBJ.sub("}", s)
        s = _RE_TRAILING_COMMA_ARR.sub("]", s)

        if "'" in s and '"' not in s:
            try:
                parsed = self._safe_literal_eval(s)
                if isinstance(parsed, (dict, list)):
                    s = json.dumps(parsed, ensure_ascii=False)
            except (ValueError, SyntaxError):
                pass

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
        import ast
        
        try:
            node = ast.parse(expr, mode='eval')
        except SyntaxError as e:
            raise ValueError(f"Invalid expression: {e}")
        
        def _validate_node(n: ast.AST) -> Any:
            if isinstance(n, ast.Expression):
                return _validate_node(n.body)
            elif isinstance(n, ast.Constant):
                if type(n.value) not in _ALLOWED_LITERAL_TYPES:
                    raise ValueError(f"Unsupported type: {type(n.value)}")
                return n.value
            elif isinstance(n, (ast.List, ast.Tuple)):
                return [_validate_node(el) for el in n.elts]
            elif isinstance(n, ast.Dict):
                keys = []
                for k in n.keys:
                    if k is None:
                        raise ValueError("Dict unpacking not supported")
                    keys.append(_validate_node(k))
                values = [_validate_node(v) for v in n.values]
                return dict(zip(keys, values))
            elif isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
                operand = _validate_node(n.operand)
                if not isinstance(operand, (int, float)):
                    raise ValueError("Unary operators only allowed on numbers")
                return -operand if isinstance(n.op, ast.USub) else operand
            else:
                raise ValueError(f"Unsupported node type: {type(n).__name__}")
        
        return _validate_node(node)

    def reset(self) -> None:
        with self._lock:
            self.content_parts.clear()
            self.tool_calls_map.clear()
            self._args_len_map.clear()
            self._max_tool_index = -1
            self._content_len = 0

    def get_current_content(self) -> str:
        with self._lock:
            return "".join(self.content_parts)

    def get_tool_call_count(self) -> int:
        with self._lock:
            return len(self.tool_calls_map)

    def __repr__(self) -> str:
        return (
            f"StreamBuffer("
            f"content_length={self._content_len}, "
            f"tool_calls={self.get_tool_call_count()}"
            f")"
        )


__all__ = ["StreamBuffer"]