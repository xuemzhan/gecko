# gecko/core/engine/buffer.py
"""
流式缓冲区模块

核心职责：
1. 解决 OpenAI 协议中流式 Tool Call 分片传输、乱序到达的问题
2. 解决 LLM 输出不规范 JSON（如 Markdown 包裹）导致的解析崩溃问题
3. 提供线程安全的流式数据聚合

设计原则：
- 防御性编程：假设 LLM 输出可能不规范
- 资源保护：限制缓冲区大小，防止内存溢出
- 容错处理：解析失败时优雅降级

修复记录：
- [P2] 增强完整性检查和验证机制
- [P1] 进阶 JSON 清洗，处理更多边缘情况
"""
from __future__ import annotations

import json
import re
import threading
from typing import Any, Dict, List, Optional

from gecko.core.message import Message
from gecko.core.protocols import StreamChunk
from gecko.core.logging import get_logger

logger = get_logger(__name__)


class StreamBuffer:
    """
    流式响应聚合缓冲区
    
    使用场景：
    在 Engine 的 Thinking 阶段，随着 LLM 流式吐出 token，此类负责实时聚合，
    并在最后产出一个结构完整的 Message 对象。
    
    主要功能：
    1. 聚合文本内容片段
    2. 处理分片传输的工具调用
    3. 修复不规范的 JSON 参数
    
    线程安全：
    使用可重入锁保护内部状态，支持并发调用。
    
    使用示例：
        ```python
        buffer = StreamBuffer()
        
        async for chunk in model.astream(messages):
            text_delta = buffer.add_chunk(chunk)
            if text_delta:
                print(text_delta, end="", flush=True)
        
        message = buffer.build_message()
        ```
    
    属性：
        max_content_chars: 最大内容字符数限制
        max_argument_chars: 最大参数字符数限制
        max_tool_index: 最大工具索引限制
    """
    
    def __init__(
        self,
        max_content_chars: int = 200_000,
        max_argument_chars: int = 100_000,
        max_tool_index: int = 1000,
    ):
        """
        初始化流式缓冲区
        
        参数：
            max_content_chars: 最大内容字符数（防止内存溢出），默认 200k
            max_argument_chars: 工具参数最大字符数，默认 100k
            max_tool_index: 最大工具索引值（防止稀疏数组攻击），默认 1000
        """
        # 文本内容片段列表
        self.content_parts: List[str] = []
        
        # 工具调用映射：index -> tool_call_dict
        # 用于处理并发或分片传输的工具调用
        self.tool_calls_map: Dict[int, Dict[str, Any]] = {}
        
        # 跟踪最大工具索引
        self._max_tool_index: int = -1
        
        # 可配置的限制参数
        self._max_content_chars: int = max_content_chars
        self._max_argument_chars: int = max_argument_chars
        self._max_tool_index_limit: int = max_tool_index
        
        # 可重入锁，保护并发访问
        self._lock: threading.RLock = threading.RLock()
        
    def add_chunk(self, chunk: StreamChunk) -> Optional[str]:
        """
        接收一个流式块，更新内部状态
        
        参数：
            chunk: StreamChunk 对象，包含 delta 字段
        
        返回：
            Optional[str]: 本次 chunk 中新增的文本内容（用于流式回显）。
                          如果是纯工具调用的 chunk，返回 None。
        
        注意：
            - 方法是线程安全的
            - 自动处理内容截断和工具索引验证
            - 无效的 chunk 会被静默跳过
        """
        with self._lock:
            # 验证 chunk 格式
            if not hasattr(chunk, 'delta') or chunk.delta is None:
                logger.debug("StreamChunk 缺少 delta 字段，跳过")
                return None
            
            delta = chunk.delta
            new_content: Optional[str] = None

            # 1. 聚合文本内容
            content = delta.get("content")
            if content:
                new_content = self._add_content(content)

            # 2. 聚合工具调用
            tool_calls = delta.get("tool_calls")
            if tool_calls:
                self._add_tool_calls(tool_calls)
        
        return new_content
    
    def _add_content(self, content: str) -> str:
        """
        添加文本内容到缓冲区
        
        参数：
            content: 新的文本片段
        
        返回：
            实际添加的文本（可能被截断）
        """
        # 计算当前累计长度
        current_len = sum(len(p) for p in self.content_parts)
        
        # 检查是否超出限制
        if current_len + len(content) > self._max_content_chars:
            logger.warning(
                "内容超出限制，将截断",
                current_len=current_len,
                incoming_len=len(content),
                limit=self._max_content_chars
            )
            # 只追加能容纳的部分
            allowed = max(0, self._max_content_chars - current_len)
            if allowed > 0:
                truncated = content[:allowed]
                self.content_parts.append(truncated)
                return truncated
            return ""
        
        self.content_parts.append(content)
        return content
    
    def _add_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> None:
        """
        添加工具调用到缓冲区
        
        处理分片传输和乱序到达的工具调用。
        
        参数：
            tool_calls: 工具调用片段列表
        """
        for tc in tool_calls:
            idx = tc.get("index")
            if idx is None:
                continue

            # 验证索引有效性
            if idx < 0:
                logger.warning(f"收到负数工具索引: {idx}，跳过")
                continue

            if idx > self._max_tool_index_limit:
                logger.warning(
                    f"工具索引超出限制: {idx} > {self._max_tool_index_limit}，跳过"
                )
                continue

            # 检测异常大的索引间隙（可能是恶意数据或配置错误）
            if self._max_tool_index >= 0:
                gap = idx - self._max_tool_index
                if gap > 500:
                    logger.warning(
                        f"检测到异常大的工具索引间隙",
                        prev_max=self._max_tool_index,
                        new_idx=idx,
                        gap=gap,
                        action="继续处理但需监控"
                    )

            # 更新最大索引
            if idx > self._max_tool_index:
                self._max_tool_index = idx

            # 初始化该索引的工具调用结构
            if idx not in self.tool_calls_map:
                self.tool_calls_map[idx] = {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""}
                }

            target = self.tool_calls_map[idx]

            # 增量拼接 ID
            if tc.get("id"):
                target["id"] += tc["id"]

            # 增量合并函数信息
            func = tc.get("function", {})
            if func:
                self._merge_function(target["function"], func)
    
    def _merge_function(
        self, 
        target: Dict[str, str], 
        incoming: Dict[str, Any]
    ) -> None:
        """
        合并函数调用信息
        
        参数：
            target: 目标函数字典
            incoming: 新到达的函数片段
        """
        # 函数名使用"最新非空片段覆盖"策略
        # 避免在流式分片时产生重复拼接（如 "search" + "search" = "searchsearch"）
        if incoming.get("name"):
            target["name"] = incoming["name"]
        
        # 参数使用增量拼接
        if incoming.get("arguments"):
            new_args = incoming["arguments"]
            current_len = len(target["arguments"])
            
            # 检查参数长度限制
            if current_len + len(new_args) > self._max_argument_chars:
                logger.warning(
                    f"工具参数超出限制 ({current_len + len(new_args)} > {self._max_argument_chars})，截断"
                )
                allowed = max(0, self._max_argument_chars - current_len)
                if allowed > 0:
                    target["arguments"] += new_args[:allowed]
            else:
                target["arguments"] += new_args

    def build_message(self) -> Message:
        """
        构建最终的 Message 对象
        
        在此阶段会对收集到的 JSON 参数进行"清洗"和"修复"，
        防止因 Markdown 符号或引号问题导致后续工具执行失败。
        
        返回：
            Message: 完整的助手消息，包含聚合后的内容和工具调用
        
        注意：
            - 空名称的工具调用会被跳过
            - 无法解析的 JSON 参数会被替换为空对象 {}
            - 返回的消息符合 OpenAI API 规范
        """
        # 聚合所有文本内容
        full_content = "".join(self.content_parts)
        
        # 处理工具调用
        tool_calls_list: List[Dict[str, Any]] = []
        
        # 按索引排序，确保顺序一致性
        for idx in sorted(self.tool_calls_map.keys()):
            raw_tc = self.tool_calls_map[idx]
            
            # 验证工具调用完整性
            func_name = raw_tc["function"].get("name", "").strip()
            raw_args = raw_tc["function"].get("arguments", "").strip()
            
            # 跳过空名称的工具调用
            if not func_name:
                logger.warning(f"索引 {idx} 的工具调用缺少名称，跳过")
                continue
            
            # 处理空参数
            if not raw_args:
                logger.debug(f"工具 '{func_name}' (索引 {idx}) 参数为空，使用默认 {{}}")
                raw_args = "{}"
            
            # 深度清洗参数 JSON
            cleaned_args = self._clean_arguments(raw_args)
            
            # 更新清洗后的参数
            raw_tc["function"]["arguments"] = cleaned_args
            
            tool_calls_list.append(raw_tc)
        
        if tool_calls_list:
            logger.debug(f"构建消息完成，包含 {len(tool_calls_list)} 个工具调用")
            
        return Message.assistant(
            content=full_content,
            # 只有当列表非空时才设值，符合 OpenAI 规范
            tool_calls=tool_calls_list if tool_calls_list else None
        )

    def _clean_arguments(self, raw_json: str) -> str:
        """
        进阶 JSON 清洗：处理 LLM 的常见输出格式问题
        
        修复项：
        1. Markdown 代码块包裹 (```json ... ```)
        2. 首尾多余的误加引号 ('{...}')
        3. 尾部逗号 ({"key": "value",})
        4. 未转义的特殊字符
        5. 单引号替换为双引号
        
        参数：
            raw_json: 原始 JSON 字符串
        
        返回：
            清洗后的 JSON 字符串，如果无法修复则返回 "{}"
        """
        if not raw_json:
            return "{}"
        
        # 1. 快速路径：尝试直接解析（如果模型输出规范）
        try:
            json.loads(raw_json)
            return raw_json
        except json.JSONDecodeError:
            pass
        
        cleaned = raw_json.strip()
        
        # 2. 去除 Markdown 代码块
        # 匹配 ```json {...} ``` 或 ``` {...} ```
        if cleaned.startswith("```"):
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
            if match:
                cleaned = match.group(1).strip()
        
        # 3. 去除首尾多余的误加引号
        # 例如模型输出了: '{"arg": "val"}' (带单引号的字符串)
        if (
            (cleaned.startswith("'") and cleaned.endswith("'")) or
            (cleaned.startswith('"') and cleaned.endswith('"'))
        ):
            # 确保不是合法的 JSON 字符串
            inner = cleaned[1:-1]
            if inner.startswith("{") or inner.startswith("["):
                cleaned = inner
        
        # 4. 修复尾部逗号
        # {"key": "value",} -> {"key": "value"}
        cleaned = re.sub(r',\s*}', '}', cleaned)
        # ["item",] -> ["item"]
        cleaned = re.sub(r',\s*\]', ']', cleaned)
        
        # 5. 尝试修复单引号（某些模型可能输出 Python dict 格式）
        # 只在确定不会破坏数据的情况下替换
        if "'" in cleaned and '"' not in cleaned:
            cleaned = cleaned.replace("'", '"')
        
        # 6. 再次尝试解析验证
        try:
            json.loads(cleaned)
            return cleaned
        except json.JSONDecodeError as e:
            # 如果还无法解析，记录警告并返回空对象
            logger.warning(
                "JSON 参数清洗失败，返回空对象",
                original=raw_json[:100] if len(raw_json) > 100 else raw_json,
                parse_error=str(e)
            )
            # 返回空对象而非脏数据，这样工具调用会因参数缺失而失败，
            # LLM 可以在错误消息中看到并尝试修正
            return "{}"
    
    def reset(self) -> None:
        """
        重置缓冲区状态
        
        清空所有累积的内容和工具调用，可用于复用缓冲区实例。
        """
        with self._lock:
            self.content_parts.clear()
            self.tool_calls_map.clear()
            self._max_tool_index = -1
    
    def get_current_content(self) -> str:
        """
        获取当前累积的文本内容（不构建完整消息）
        
        返回：
            当前累积的所有文本内容
        """
        with self._lock:
            return "".join(self.content_parts)
    
    def get_tool_call_count(self) -> int:
        """
        获取当前累积的工具调用数量
        
        返回：
            工具调用数量
        """
        with self._lock:
            return len(self.tool_calls_map)
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"StreamBuffer("
            f"content_length={len(self.get_current_content())}, "
            f"tool_calls={self.get_tool_call_count()}"
            f")"
        )


# ====================== 模块导出 ======================

__all__ = ["StreamBuffer"]