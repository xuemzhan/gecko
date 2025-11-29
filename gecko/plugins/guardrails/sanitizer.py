# gecko/plugins/guardrails/sanitizer.py
"""
输入安全清洗器

检测和处理潜在的 Prompt Injection 攻击。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Pattern, Tuple

from gecko.core.logging import get_logger

logger = get_logger(__name__)


class ThreatLevel(str, Enum):
    """威胁等级"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class ScanResult:
    """扫描结果"""
    text: str
    threat_level: ThreatLevel = ThreatLevel.NONE
    detections: List[str] = field(default_factory=list)
    sanitized_text: Optional[str] = None

    @property
    def is_clean(self) -> bool:
        return self.threat_level == ThreatLevel.NONE

    @property
    def was_modified(self) -> bool:
        return self.sanitized_text is not None and self.sanitized_text != self.text


class InputSanitizer:
    """
    输入清洗器
    
    示例:
        ```python
        sanitizer = InputSanitizer()
        result = sanitizer.scan("Ignore previous instructions...")
        
        if result.threat_level == ThreatLevel.HIGH:
            raise SecurityError("Suspicious input detected")
        ```
    """

    # 检测模式: (pattern, name, threat_level)
    DEFAULT_PATTERNS: List[Tuple[str, str, ThreatLevel]] = [
        # 高危: 指令覆盖
        (r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?)",
         "instruction_override", ThreatLevel.HIGH),
        (r"disregard\s+(everything|all|what)",
         "instruction_override", ThreatLevel.HIGH),
        (r"forget\s+(everything|all)\s+(you|i)\s+(know|said)",
         "instruction_override", ThreatLevel.HIGH),

        # 高危: 角色劫持
        (r"you\s+are\s+now\s+(a|an|in)\s+\w+\s+mode",
         "role_hijack", ThreatLevel.HIGH),
        (r"pretend\s+(to\s+be|you\s+are)",
         "role_hijack", ThreatLevel.HIGH),

        # 高危: 提示泄露
        (r"(show|reveal|print)\s+(your|the)\s+(system|initial)\s+prompt",
         "prompt_leak", ThreatLevel.HIGH),

        # 中危: 特殊标记
        (r"\[/?INST\]", "special_token", ThreatLevel.MEDIUM),
        (r"<\|?(system|user|assistant)\|?>", "special_token", ThreatLevel.MEDIUM),
        (r"###\s*(System|Human|Assistant):", "role_marker", ThreatLevel.MEDIUM),

        # 低危: 可疑前缀
        (r"^system:\s*", "system_prefix", ThreatLevel.LOW),
    ]

    def __init__(
        self,
        patterns: Optional[List[Tuple[str, str, ThreatLevel]]] = None,
        max_length: int = 50000,
    ):
        self.max_length = max_length

        # 编译模式
        pattern_list = patterns if patterns is not None else self.DEFAULT_PATTERNS
        self._patterns: List[Tuple[Pattern, str, ThreatLevel]] = []

        for pattern_str, name, level in pattern_list:
            try:
                compiled = re.compile(pattern_str, re.IGNORECASE)
                self._patterns.append((compiled, name, level))
            except re.error as e:
                logger.warning(f"Invalid pattern '{name}': {e}")

    def scan(self, text: str) -> ScanResult:
        """
        扫描文本中的威胁
        
        参数:
            text: 待扫描文本
            
        返回:
            ScanResult 扫描结果
        """
        result = ScanResult(text=text)

        # 长度检查
        if len(text) > self.max_length:
            result.detections.append("excessive_length")
            result.threat_level = ThreatLevel.HIGH

        # 模式匹配
        max_level = ThreatLevel.NONE

        for pattern, name, level in self._patterns:
            if pattern.search(text):
                result.detections.append(name)
                if self._level_value(level) > self._level_value(max_level):
                    max_level = level

        if self._level_value(max_level) > self._level_value(result.threat_level):
            result.threat_level = max_level

        return result

    def sanitize(self, text: str) -> ScanResult:
        """
        扫描并清洗文本
        
        参数:
            text: 待处理文本
            
        返回:
            ScanResult 包含清洗后的文本
        """
        result = self.scan(text)

        if result.is_clean:
            return result

        sanitized = text

        # 长度截断
        if len(sanitized) > self.max_length:
            sanitized = sanitized[:self.max_length] + "\n[truncated]"

        # 移除特殊标记
        if any(d in ("special_token", "role_marker") for d in result.detections):
            sanitized = re.sub(r"\[/?INST\]", "", sanitized)
            sanitized = re.sub(r"<\|?(system|user|assistant)\|?>", "", sanitized)
            sanitized = re.sub(r"###\s*(System|Human|Assistant):", "", sanitized)

        result.sanitized_text = sanitized

        if result.was_modified:
            logger.warning(
                "Input sanitized",
                threat_level=result.threat_level.value,
                detections=result.detections
            )

        return result

    @staticmethod
    def _level_value(level: ThreatLevel) -> int:
        """获取威胁等级数值（用于比较）"""
        return {"none": 0, "low": 1, "medium": 2, "high": 3}.get(level.value, 0)


__all__ = ["InputSanitizer", "ScanResult", "ThreatLevel"]