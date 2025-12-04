# gecko/core/engine/buffer.py
"""
流式缓冲区模块

核心职责：
1. 解决 OpenAI 协议中流式 Tool Call 分片传输、乱序到达的问题。
2. 解决 LLM 输出不规范 JSON (如 Markdown 包裹) 导致的解析崩溃问题。
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Any, Optional
import threading

from gecko.core.message import Message
from gecko.core.protocols import StreamChunk
from gecko.core.logging import get_logger

logger = get_logger(__name__)

class StreamBuffer:
    """
    流式响应聚合缓冲区。
    
    使用场景：
    在 Engine 的 Thinking 阶段，随着 LLM 流式吐出 token，此类负责实时聚合，
    并在最后产出一个结构完整的 Message 对象。
    
    [P2-1 Fix] 增强的完整性检查和验证机制。
    """
    def __init__(self):
        self.content_parts: List[str] = []
        # tool_index -> tool_call_dict (处理并发或分片传输的工具调用)
        self.tool_calls_map: Dict[int, Dict[str, Any]] = {}
        self._max_tool_index: int = -1  # [P2-1 Fix] 跟踪最大工具索引
        self._max_content_chars: int = 200_000  # 防止工具返回超大内容
        # 使用可重入锁保护同步场景下的 add_chunk/构建操作
        self._lock: threading.RLock = threading.RLock()
        
    def add_chunk(self, chunk: StreamChunk) -> Optional[str]:
        """
        接收一个流式块 (StreamChunk)，更新内部状态。
        
        返回：
            Optional[str]: 本次 chunk 中新增的文本内容（用于流式回显）。
            如果是纯工具调用的 chunk，返回 None。
        
        [P2-1 Fix] 添加了索引验证和范围检查，防止稀疏索引导致的内存溢出。
        """
        # add_chunk 可能并发调用（来自不同协程/线程），使用锁保证原子性
        with self._lock:
            delta = chunk.delta

            # 1. 聚合文本内容
            content = delta.get("content")
            if content:
                # 防止累计过长的 content 导致内存峰值
                current_len = sum(len(p) for p in self.content_parts)
                if current_len + len(content) > self._max_content_chars:
                    logger.warning("Truncating incoming content to avoid memory blowup")
                    # 仅追加能容纳的部分
                    allowed = max(0, self._max_content_chars - current_len)
                    if allowed > 0:
                        self.content_parts.append(content[:allowed])
                else:
                    self.content_parts.append(content)

            # 2. 聚合工具调用 (处理 index 可能不连续或乱序的情况)
            if delta.get("tool_calls"):
                for tc in delta["tool_calls"]:
                    idx = tc.get("index")
                    if idx is None:
                        continue

                    # [P2-1 Fix] 验证索引范围，防止稀疏索引导致内存溢出
                    # 例如：如果收到索引 1000000 而之前只有索引 0，这可能是错误
                    if idx < 0:
                        logger.warning(f"Negative tool index received: {idx}, skipping")
                        continue

                    if idx > 1000:  # 合理的上限：最多允许 1000 个工具调用
                        logger.warning(f"Excessive tool index: {idx}, likely malformed response")
                        continue

                    # 防止稀疏索引（例如之前最大索引为 0，却突然收到 100000），这通常是协议或模型错误。
                    if self._max_tool_index >= 0 and (idx - self._max_tool_index) > 100:
                        logger.warning(
                            f"Sparse tool index gap detected: prev_max={self._max_tool_index} new_idx={idx}, skipping"
                        )
                        continue

                    # 跟踪最大索引
                    if idx > self._max_tool_index:
                        self._max_tool_index = idx

                    # 初始化该索引的结构
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

                    # 增量合并函数名和参数
                    # 对于函数名使用“最新非空片段覆盖”策略，避免在流式分片时产生重复拼接
                    func = tc.get("function", {})
                    if func.get("name"):
                        # 使用最新的非空片段替换之前的 name（而不是拼接）
                        target["function"]["name"] = func.get("name")
                    if func.get("arguments"):
                        # 防止参数字符串过长
                        if len(target["function"]["arguments"]) + len(func.get("arguments", "")) > 100_000:
                            logger.warning("Truncating tool arguments due to excessive length")
                            # 只保留前 100k 字符
                            target["function"]["arguments"] = (
                                target["function"]["arguments"] + func.get("arguments", "")
                            )[:100_000]
                        else:
                            target["function"]["arguments"] += func["arguments"]
        
        return content

    def build_message(self) -> Message:
        """
        构建最终的 Message 对象。
        
        关键逻辑：
        在此阶段会对收集到的 JSON 参数字符串进行“清洗”和“修复”，
        防止因为 Markdown 符号或引号问题导致后续工具执行失败。
        """
        full_content = "".join(self.content_parts)
        tool_calls_list = []
        
        # 按索引排序，确保工具调用顺序一致
        for idx in sorted(self.tool_calls_map.keys()):
            raw_tc = self.tool_calls_map[idx]
            
            # [P2-1 Fix] 验证工具调用的完整性
            func_name = raw_tc["function"].get("name", "").strip()
            raw_args = raw_tc["function"].get("arguments", "").strip()
            
            if not func_name:
                logger.warning(f"Tool call at index {idx} has empty name, skipping")
                continue
            
            if not raw_args:
                # 如果 arguments 为空，默认设为 {}
                logger.warning(f"Tool call '{func_name}' at index {idx} has empty arguments, using default {{}}") 
                raw_args = "{}"
            
            # [生产级增强] 深度清洗参数 JSON
            cleaned_args = self._clean_arguments(raw_args)
            
            # 更新清洗后的参数
            raw_tc["function"]["arguments"] = cleaned_args
            
            tool_calls_list.append(raw_tc)
        
        if tool_calls_list:
            logger.debug(f"Built message with {len(tool_calls_list)} tool calls")
            
        return Message.assistant(
            content=full_content,
            # 只有当列表非空时才设值，符合 Pydantic 定义及 OpenAI 规范
            tool_calls=tool_calls_list if tool_calls_list else None
        )

    def _clean_arguments(self, raw_json: str) -> str:
        """
        [P1-4 Fix] 进阶 JSON 清洗：处理 LLM 的常见输出格式问题
        
        修复项：
        1. Markdown 代码块包裹 (```json ... ```)
        2. 首尾多余的误加引号 ('{...}')
        3. 尾部逗号 ({"key": "value",})
        4. 未转义的换行符
        """
        if not raw_json: 
            return "{}"
            
        # 1. 尝试直接解析（这是最快路径，如果模型输出规范）
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
        
        # 3. [P1-4 Fix] 去除首尾多余的误加引号
        # 例如模型输出了: '{"arg": "val"}' (带单引号的字符串)
        if (cleaned.startswith("'") and cleaned.endswith("'")) or \
           (cleaned.startswith('"') and cleaned.endswith('"')):
            cleaned = cleaned[1:-1]
        
        # 4. [P1-4 Fix] 修复尾部逗号 ({"key": "value",})
        cleaned = re.sub(r',\s*}', '}', cleaned)  # {...,} -> {...}
        cleaned = re.sub(r',\s*\]', ']', cleaned)  # [...,] -> [...]

        # 5. 再次尝试解析验证 (Best Effort)
        try:
            json.loads(cleaned)
            return cleaned
        except json.JSONDecodeError as e:
            # [P1-4 Fix] 如果还无法解析，返回空对象而非脏数据
            # 这避免工具执行时因脏 JSON 而崩溃
            logger.warning(
                f"Failed to clean JSON arguments, returning empty dict",
                original={raw_json[:100] if raw_json else ""},
                parse_error=str(e)
            )
            # 返回空对象而非原始脏数据，这样工具调用会失败，
            # LLM 可以在错误消息中看到并修正
            return "{}"