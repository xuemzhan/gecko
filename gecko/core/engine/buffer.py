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
    """
    def __init__(self):
        self.content_parts: List[str] = []
        # tool_index -> tool_call_dict (处理并发或分片传输的工具调用)
        self.tool_calls_map: Dict[int, Dict[str, Any]] = {}
        
    def add_chunk(self, chunk: StreamChunk) -> Optional[str]:
        """
        接收一个流式块 (StreamChunk)，更新内部状态。
        
        返回：
            Optional[str]: 本次 chunk 中新增的文本内容（用于流式回显）。
            如果是纯工具调用的 chunk，返回 None。
        """
        delta = chunk.delta
        
        # 1. 聚合文本内容
        content = delta.get("content")
        if content:
            self.content_parts.append(content)
            
        # 2. 聚合工具调用 (处理 index 可能不连续或乱序的情况)
        if delta.get("tool_calls"):
            for tc in delta["tool_calls"]:
                idx = tc.get("index")
                if idx is None: 
                    continue
                
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
                
                # 增量拼接函数名和参数
                func = tc.get("function", {})
                if func.get("name"): 
                    target["function"]["name"] += func["name"]
                if func.get("arguments"): 
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
            
            # [生产级增强] 深度清洗参数 JSON
            raw_args = raw_tc["function"]["arguments"]
            cleaned_args = self._clean_arguments(raw_args)
            
            # 更新清洗后的参数
            raw_tc["function"]["arguments"] = cleaned_args
            
            tool_calls_list.append(raw_tc)
            
        return Message.assistant(
            content=full_content,
            # 只有当列表非空时才设值，符合 Pydantic 定义及 OpenAI 规范
            tool_calls=tool_calls_list if tool_calls_list else None
        )

    def _clean_arguments(self, raw_json: str) -> str:
        """
        清洗 LLM 输出的脏 JSON 字符串。
        
        常见问题修复：
        1. Markdown 代码块包裹 (```json ... ```)
        2. 首尾多余的误加引号 ('{...}')
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
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if match:
            cleaned = match.group(1)
            
        # 3. 简单修复：去除首尾多余的误加引号
        # 例如模型输出了: '{"arg": "val"}' (带单引号的字符串)
        if cleaned.startswith("'") and cleaned.endswith("'"):
            cleaned = cleaned[1:-1]
        elif cleaned.startswith('"') and cleaned.endswith('"'):
            # 只有当看起来是误加的外部引号时才去除
            cleaned = cleaned[1:-1]

        # 4. 再次尝试解析验证 (Best Effort)
        try:
            json.loads(cleaned)
            return cleaned
        except Exception:
            # 如果还无法解析，记录警告并返回原始内容，让上层 Engine/ToolBox 报错
            # 这样可以在 Tool Output 中反馈给 LLM，让它自己修正
            logger.warning(f"Failed to clean JSON arguments: {raw_json[:50]}...")
            return raw_json