# gecko/core/engine/buffer.py
"""
流式缓冲区模块（Production Grade / 生产级）

核心职责：
1) 解决 OpenAI 协议中流式 Tool Call 分片传输、乱序到达的问题
2) 解决 LLM 输出不规范 JSON（如 Markdown 包裹 / 尾逗号 / 误加引号）导致的解析崩溃问题
3) 提供线程安全的流式数据聚合，并具备资源保护（长度限制）

✅ 本版本在“保持原有对外接口不变”的前提下，修复与增强：
------------------------------------------------------------
[P0] 性能修复：避免每个 chunk 都 sum(content_parts) 导致 O(n²)
     - 引入 _content_len 累加计数
     - 引入 _args_len_map 累加计数（按 tool index 追踪 arguments 长度）

[P1] 工具调用 ID 合并策略修复：不再做字符串拼接（避免 call_1call_1...）
     - 改为“首次非空赋值/最新覆盖”策略（更符合各厂商实现）

[P1] build_message 阶段不再“就地修改”内部缓存，避免污染 raw 数据
     - 输出时创建深拷贝结构，内部保留原始 arguments（便于 debug/复用）

[P1] 更严格的索引与数据防御
     - tool_calls 片段格式异常时防御性跳过
     - 保护 max_tool_index_limit，避免稀疏数组攻击

[P2] JSON 清洗增强：更稳健处理 ```json / ``` 包裹、误加引号、尾逗号、单引号 dict
     - 保持“尽量修复，修复失败返回 {}”的容错策略

设计原则：
- 防御性编程：假设模型输出可能不规范
- 资源保护：限制缓冲区长度，防止内存溢出
- 容错处理：解析失败时优雅降级（返回 {}）
"""

from __future__ import annotations

import copy
import json
import re
import threading
from typing import Any, Dict, List, Optional

from gecko.core.logging import get_logger
from gecko.core.message import Message
from gecko.core.protocols import StreamChunk

logger = get_logger(__name__)


class StreamBuffer:
    """
    流式响应聚合缓冲区

    使用场景：
    - Engine 在“思考阶段（Thinking）”使用 model.astream() 获取 chunk 流
    - 该类负责实时聚合 content 与 tool_calls 分片
    - 最终 build_message() 产出一个结构完整的 Message.assistant

    线程安全：
    - 内部使用 threading.RLock 保护并发访问
    - 在多数 asyncio 单线程场景中并不必须，但保留可增强健壮性（允许多线程回调/桥接）

    资源保护：
    - max_content_chars：content 总字符数上限
    - max_argument_chars：单个工具调用 arguments 总字符数上限
    - max_tool_index：工具调用 index 上限，防稀疏数组攻击
    """

    def __init__(
        self,
        max_content_chars: int = 200_000,
        max_argument_chars: int = 100_000,
        max_tool_index: int = 1000,
    ):
        # content 片段缓存
        self.content_parts: List[str] = []
        # ✅ 性能优化：累计内容长度，避免每次 sum(len(part)) O(n)
        self._content_len: int = 0

        # tool_calls_map: index -> tool_call_dict（用于乱序分片合并）
        self.tool_calls_map: Dict[int, Dict[str, Any]] = {}
        # 追踪最大工具索引（用于 gap 监控与防御）
        self._max_tool_index: int = -1

        # ✅ 性能优化：记录每个 index 的 arguments 当前长度，避免频繁 len(target["arguments"])
        self._args_len_map: Dict[int, int] = {}

        # 可配置限制
        self._max_content_chars: int = int(max_content_chars)
        self._max_argument_chars: int = int(max_argument_chars)
        self._max_tool_index_limit: int = int(max_tool_index)

        # 可重入锁
        self._lock: threading.RLock = threading.RLock()

    # ---------------------------------------------------------------------
    # 公共入口：接收 chunk
    # ---------------------------------------------------------------------

    def add_chunk(self, chunk: StreamChunk) -> Optional[str]:
        """
        接收一个流式块，更新内部状态

        ✅ P1：delta 提取改为多协议兼容
        """
        with self._lock:
            delta = self._extract_delta(chunk)
            if delta is None:
                logger.debug("StreamChunk 无法提取 delta，跳过", chunk_type=type(chunk).__name__)
                return None

            new_content: Optional[str] = None

            # 1) 聚合文本内容
            content = delta.get("content")
            if isinstance(content, str) and content:
                added = self._add_content(content)
                if added:
                    new_content = added

            # 2) 聚合工具调用
            tool_calls = delta.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                self._add_tool_calls(tool_calls)

            return new_content

    def _extract_delta(self, chunk: Any) -> Optional[Dict[str, Any]]:
        """
        ✅ P1：从不同实现的 chunk 结构中提取 delta（防兼容性坑）

        兼容顺序：
        1) chunk.delta（你当前实现）
        2) chunk.choices[0].delta（部分 SDK）
        3) chunk.choices[0]["delta"]（dict 风格）
        4) chunk["choices"][0]["delta"]（纯 dict）
        """
        # 1) chunk.delta
        delta = getattr(chunk, "delta", None)
        if isinstance(delta, dict):
            return delta

        # 2/3) chunk.choices[0].delta or ["delta"]
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

        # 4) chunk is dict
        if isinstance(chunk, dict):
            choices = chunk.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                d = choices[0].get("delta")
                if isinstance(d, dict):
                    return d

        return None


    # ---------------------------------------------------------------------
    # 内容聚合（修复 O(n²)）
    # ---------------------------------------------------------------------

    def _add_content(self, content: str) -> str:
        """
        添加文本内容到缓冲区（带总长度限制）

        性能点：
            - 使用 self._content_len 累计计数，避免每次 sum(content_parts) O(n)

        返回：
            实际添加的内容（可能被截断）
        """
        incoming_len = len(content)

        # 超限则截断
        if self._content_len + incoming_len > self._max_content_chars:
            logger.warning(
                "内容超出限制，将截断",
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

    # ---------------------------------------------------------------------
    # 工具调用聚合（乱序 / 分片）
    # ---------------------------------------------------------------------

    def _add_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> None:
        """
        添加工具调用片段到缓冲区

        处理能力：
            - 分片传输：arguments 可能被多次增量吐出
            - 乱序到达：index 可能不是单调递增
            - 防御性限制：index 范围、异常 gap 监控
        """
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue

            idx = tc.get("index")
            if not isinstance(idx, int):
                continue

            # 索引有效性
            if idx < 0:
                logger.warning("收到负数工具索引，跳过", idx=idx)
                continue
            if idx > self._max_tool_index_limit:
                logger.warning("工具索引超出限制，跳过", idx=idx, limit=self._max_tool_index_limit)
                continue

            # gap 监控（不是强拦截，只告警）
            if self._max_tool_index >= 0:
                gap = idx - self._max_tool_index
                if gap > 500:
                    logger.warning(
                        "检测到异常大的工具索引间隙",
                        prev_max=self._max_tool_index,
                        new_idx=idx,
                        gap=gap,
                    )

            if idx > self._max_tool_index:
                self._max_tool_index = idx

            # 初始化结构（OpenAI tool_calls 协议结构）
            if idx not in self.tool_calls_map:
                self.tool_calls_map[idx] = {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
                self._args_len_map[idx] = 0

            target = self.tool_calls_map[idx]

            # ✅ 修复：tool_call_id 不做增量拼接，使用“首次非空赋值 / 最新覆盖”
            # 原因：多数模型实现会在首个分片给完整 id，后续分片可能重复携带相同 id
            tc_id = tc.get("id")
            if isinstance(tc_id, str) and tc_id:
                if not target.get("id"):
                    target["id"] = tc_id
                else:
                    # 如果 id 变化（极少见），以最新为准并告警
                    if target["id"] != tc_id:
                        logger.warning(
                            "检测到同一 index 的 tool_call id 发生变化，使用最新覆盖",
                            index=idx,
                            old_id=target["id"],
                            new_id=tc_id,
                        )
                        target["id"] = tc_id

            # 合并 function 片段
            func = tc.get("function")
            if isinstance(func, dict):
                self._merge_function(idx, target["function"], func)

    def _merge_function(self, idx: int, target: Dict[str, str], incoming: Dict[str, Any]) -> None:
        """
        合并函数调用信息

        策略：
            - name：最新非空覆盖（防止 "search"+"search" 拼接成 "searchsearch"）
            - arguments：增量拼接（流式分片），但受 max_argument_chars 限制
        """
        # name：覆盖策略
        inc_name = incoming.get("name")
        if isinstance(inc_name, str) and inc_name:
            target["name"] = inc_name

        # arguments：拼接策略 + 长度限制
        inc_args = incoming.get("arguments")
        if not (isinstance(inc_args, str) and inc_args):
            return

        current_len = self._args_len_map.get(idx, 0)
        incoming_len = len(inc_args)

        if current_len + incoming_len > self._max_argument_chars:
            logger.warning(
                "工具参数超出限制，将截断",
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

    # ---------------------------------------------------------------------
    # 构建 Message（输出不污染内部 raw）
    # ---------------------------------------------------------------------

    def build_message(self) -> Message:
        """
        构建最终的 Message.assistant

        行为：
            - 聚合 content
            - tool_calls 按 index 排序
            - arguments 进行“深度清洗”
            - 输出阶段使用拷贝结构，避免污染内部缓存（保留 raw 便于 debug）

        容错策略：
            - 缺少 function.name：跳过该 tool_call
            - arguments 为空：填 {}
            - 清洗失败：替换为 "{}"
        """
        with self._lock:
            full_content = "".join(self.content_parts)

            tool_calls_list: List[Dict[str, Any]] = []

            for idx in sorted(self.tool_calls_map.keys()):
                raw_tc = self.tool_calls_map[idx]
                func = raw_tc.get("function", {})
                func_name = (func.get("name") or "").strip() if isinstance(func, dict) else ""

                if not func_name:
                    logger.warning("工具调用缺少名称，跳过", index=idx)
                    continue

                raw_args = ""
                if isinstance(func, dict):
                    raw_args = (func.get("arguments") or "").strip()

                if not raw_args:
                    raw_args = "{}"

                cleaned_args = self._clean_arguments(raw_args)

                # ✅ 输出时创建拷贝，避免污染内部 raw_tc
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
                logger.debug("构建消息完成", tool_calls=len(tool_calls_list))

            return Message.assistant(
                content=full_content,
                # OpenAI 规范：tool_calls 为空时传 None
                tool_calls=tool_calls_list if tool_calls_list else None,
            )

    # ---------------------------------------------------------------------
    # JSON 清洗（防御性）
    # ---------------------------------------------------------------------

    def _clean_arguments(self, raw_json: str) -> str:
        """
        进阶 JSON 清洗：处理 LLM 的常见输出格式问题

        修复项：
        1) Markdown 代码块包裹：```json ... ``` / ``` ... ```
        2) 首尾误加引号：'{"a":1}' 或 "\"{...}\""
        3) 尾部逗号：{"a":1,} / [1,]
        4) 仅在“无双引号”场景将单引号替换为双引号（尽量不误伤）
        5) 最终以 json.loads 验证；失败则返回 "{}"

        返回：
            清洗后的 JSON 字符串（保证 json.loads 可解析），或 "{}"
        """
        if not raw_json:
            return "{}"

        s = raw_json.strip()

        # 0) 快速路径：本来就是合法 JSON
        try:
            json.loads(s)
            return s
        except json.JSONDecodeError:
            pass

        # 1) 去掉 Markdown 代码块
        if s.startswith("```"):
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
            if m:
                s = m.group(1).strip()

        # 2) 去除首尾误加引号（仅当内部看起来像对象/数组）
        if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
            inner = s[1:-1].strip()
            if inner.startswith("{") or inner.startswith("["):
                s = inner

        # 3) 修复尾逗号
        s = re.sub(r",\s*}", "}", s)
        s = re.sub(r",\s*\]", "]", s)

        # 4) 尝试修复单引号 dict（仅当完全没有双引号时）
        #    例如：{'a': 1} -> {"a": 1}
        if "'" in s and '"' not in s:
            s = s.replace("'", '"')

        # 5) 最终校验
        try:
            json.loads(s)
            return s
        except json.JSONDecodeError as e:
            logger.warning(
                "JSON 参数清洗失败，返回空对象",
                original=(raw_json[:200] + "...") if len(raw_json) > 200 else raw_json,
                parse_error=str(e),
            )
            return "{}"

    # ---------------------------------------------------------------------
    # 复用/状态查询
    # ---------------------------------------------------------------------

    def reset(self) -> None:
        """重置缓冲区（可复用实例）"""
        with self._lock:
            self.content_parts.clear()
            self.tool_calls_map.clear()
            self._args_len_map.clear()
            self._max_tool_index = -1
            self._content_len = 0

    def get_current_content(self) -> str:
        """获取当前累积文本（不构建 Message）"""
        with self._lock:
            return "".join(self.content_parts)

    def get_tool_call_count(self) -> int:
        """获取当前累积工具调用数量（按 index 计数）"""
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
