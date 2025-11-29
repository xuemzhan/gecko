# gecko/plugins/guardrails/input_sanitizer.py
"""
输入安全清洗中间件

提供基础的 Prompt Injection 检测和清洗能力。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set
from enum import Enum

from pydantic import BaseModel, Field

from gecko.core.events import BaseEvent
from gecko.core.logging import get_logger
from gecko.plugins.guardrails.sanitizer import InputSanitizer as PublicInputSanitizer

logger = get_logger(__name__)

class ThreatLevel(str, Enum):
    """威胁等级"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SanitizationResult(BaseModel):
    """清洗结果"""
    original_text: str
    sanitized_text: str
    threat_level: ThreatLevel = ThreatLevel.NONE
    detected_patterns: List[str] = Field(default_factory=list)
    was_modified: bool = False


class InputSanitizer:
    """
    输入清洗器
    
    检测并处理潜在的 Prompt Injection 攻击模式。
    
    使用示例:
        ```python
        sanitizer = InputSanitizer()
        result = sanitizer.sanitize("Ignore previous instructions and...")
        
        if result.threat_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL):
            logger.warning("Potential attack detected", patterns=result.detected_patterns)
        ```
    """
    
    # 高危模式（可能的指令覆盖尝试）
    HIGH_RISK_PATTERNS = [
        # 指令覆盖
        (r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions?|rules?|prompts?)", "instruction_override"),
        (r"disregard\s+(all\s+)?(previous|prior|above)", "instruction_override"),
        (r"forget\s+(everything|all|what)\s+(you|i)\s+(told|said)", "instruction_override"),
        
        # 角色劫持
        (r"you\s+are\s+now\s+(a|an|in)\s+\w+\s+mode", "role_hijack"),
        (r"pretend\s+(to\s+be|you\s+are)\s+(a|an)", "role_hijack"),
        (r"act\s+as\s+(if|though)\s+you", "role_hijack"),
        
        # 系统提示泄露
        (r"(show|reveal|display|print|output)\s+(your|the)\s+(system|initial)\s+(prompt|instructions?)", "prompt_leak"),
        (r"what\s+(are|is)\s+your\s+(system|initial|original)\s+(prompt|instructions?)", "prompt_leak"),
    ]
    
    # 中危模式（可疑但可能合法）
    MEDIUM_RISK_PATTERNS = [
        # 特殊标记
        (r"\[/?INST\]", "special_token"),
        (r"<\|?(system|user|assistant|im_start|im_end)\|?>", "special_token"),
        (r"###\s*(System|Human|Assistant|User):", "special_token"),
        
        # 编码绕过尝试
        (r"base64[:=]", "encoding_bypass"),
        (r"\\x[0-9a-fA-F]{2}", "hex_encoding"),
        (r"&#\d+;", "html_entity"),
    ]
    
    # 低危模式（需要上下文判断）
    LOW_RISK_PATTERNS = [
        (r"system:\s*\w+", "system_prefix"),
        (r"</?(s|b|i|u|code|pre)>", "html_tag"),
    ]

    
    def __init__(
        self,
        enable_sanitization: bool = True,
        log_detections: bool = True,
        block_high_risk: bool = False,
        custom_patterns: Optional[List[tuple]] = None,
    ):
        """
        初始化清洗器

        参数:
            enable_sanitization: 是否启用清洗（False 则仅检测）
            log_detections: 是否记录检测结果
            block_high_risk: 是否阻止高危输入（抛出异常）
            custom_patterns: 自定义检测模式列表

        自定义模式支持两种形式：
        - (pattern, name)                   -> 默认视为 HIGH
        - (pattern, name, ThreatLevel.xxx)  -> 使用指定威胁等级
        """
        self.enable_sanitization = enable_sanitization
        self.log_detections = log_detections
        self.block_high_risk = block_high_risk

        # 编译正则表达式（内置规则）
        self._high_patterns = [
            (re.compile(p, re.IGNORECASE), name)
            for p, name in self.HIGH_RISK_PATTERNS
        ]
        self._medium_patterns = [
            (re.compile(p, re.IGNORECASE), name)
            for p, name in self.MEDIUM_RISK_PATTERNS
        ]
        self._low_patterns = [
            (re.compile(p, re.IGNORECASE), name)
            for p, name in self.LOW_RISK_PATTERNS
        ]

        # 自定义模式的等级映射：
        # key: 模式名称 name, value: ThreatLevel
        # 仅用于覆盖高危检测中的默认 HIGH 等级
        self._custom_levels = {}

        # 添加自定义模式
        if custom_patterns:
            for item in custom_patterns:
                # 兼容两种形状：(pattern, name) / (pattern, name, level)
                if len(item) == 2:
                    pattern, name = item
                    level = ThreatLevel.HIGH  # 未指定等级时默认 HIGH
                elif len(item) == 3:
                    pattern, name, level = item
                else:
                    raise ValueError(
                        f"Invalid custom pattern tuple: {item!r}, "
                        "expected (pattern, name) or (pattern, name, level)"
                    )

                compiled = re.compile(pattern, re.IGNORECASE)
                # 统一归入高危模式集合，实际等级由 _custom_levels 控制
                self._high_patterns.append((compiled, name))
                self._custom_levels[name] = level


    @staticmethod
    def _max_level(current: ThreatLevel, new: ThreatLevel) -> ThreatLevel:
        """
        比较两个威胁等级，返回更高的那个。

        由于 ThreatLevel 的 value 是字符串，不能直接用字典序比较，
        在这里显式定义等级顺序。
        """
        order = {
            ThreatLevel.NONE: 0,
            ThreatLevel.LOW: 1,
            ThreatLevel.MEDIUM: 2,
            ThreatLevel.HIGH: 3,
            ThreatLevel.CRITICAL: 4,
        }
        return new if order[new] > order[current] else current

    def detect(self, text: str) -> SanitizationResult:
        """
        检测文本中的威胁模式
        """
        detected: List[str] = []
        threat_level = ThreatLevel.NONE
        
        # 高危检测
        for pattern, name in self._high_patterns:
            if pattern.search(text):
                detected.append(f"high:{name}")
                # 修复点：自定义模式使用自定义等级，内置模式默认 HIGH
                level = self._custom_levels.get(name, ThreatLevel.HIGH)
                threat_level = self._max_level(threat_level, level)
        
        # 中危检测
        if threat_level != ThreatLevel.HIGH:
            for pattern, name in self._medium_patterns:
                if pattern.search(text):
                    detected.append(f"medium:{name}")
                    threat_level = self._max_level(threat_level, ThreatLevel.MEDIUM)
        
        # 低危检测
        for pattern, name in self._low_patterns:
            if pattern.search(text):
                detected.append(f"low:{name}")
                threat_level = self._max_level(threat_level, ThreatLevel.LOW)
        
        # 特殊检测：异常长度
        if len(text) > 50000:
            detected.append("high:excessive_length")
            threat_level = self._max_level(threat_level, ThreatLevel.HIGH)
        
        # 特殊检测：重复字符（可能的 DoS）
        if self._has_excessive_repetition(text):
            detected.append("medium:repetition_attack")
            threat_level = self._max_level(threat_level, ThreatLevel.MEDIUM)
        
        return SanitizationResult(
            original_text=text,
            sanitized_text=text,
            threat_level=threat_level,
            detected_patterns=detected,
            was_modified=False
        )

    def sanitize(self, text: str) -> SanitizationResult:
        """
        检测并清洗文本
        
        参数:
            text: 待处理文本
            
        返回:
            SanitizationResult 包含清洗后的文本
            
        异常:
            ValueError: 当 block_high_risk=True 且检测到高危模式
        """
        result = self.detect(text)
        
        if self.log_detections and result.detected_patterns:
            logger.warning(
                "Input threat detected",
                threat_level=result.threat_level.value,
                patterns=result.detected_patterns
            )
        
        # 高危阻断
        if self.block_high_risk and result.threat_level in (ThreatLevel.HIGH, ThreatLevel.CRITICAL):
            raise ValueError(
                f"Input blocked due to security concerns: {result.detected_patterns}"
            )
        
        # 执行清洗
        if self.enable_sanitization and result.threat_level != ThreatLevel.NONE:
            sanitized = self._apply_sanitization(text, result.detected_patterns)
            result.sanitized_text = sanitized
            result.was_modified = (sanitized != text)
        
        return result

    def _apply_sanitization(self, text: str, detected: List[str]) -> str:
        """应用清洗规则"""
        sanitized = text
        
        # 移除特殊标记
        if any("special_token" in d for d in detected):
            sanitized = re.sub(r"\[/?INST\]", "", sanitized)
            sanitized = re.sub(r"<\|?(system|user|assistant|im_start|im_end)\|?>", "", sanitized)
            sanitized = re.sub(r"###\s*(System|Human|Assistant|User):", "", sanitized)
        
        # 转义可疑的系统前缀
        if any("system_prefix" in d for d in detected):
            sanitized = re.sub(r"(system:\s*)", r"[escaped] \1", sanitized, flags=re.IGNORECASE)
        
        # 截断过长文本
        if any("excessive_length" in d for d in detected):
            sanitized = sanitized[:50000] + "\n[Content truncated for security]"
        
        return sanitized

    def _has_excessive_repetition(self, text: str, threshold: float = 0.5) -> bool:
        """检测过度重复"""
        if len(text) < 100:
            return False
        
        # 检查字符分布
        char_counts: Dict[str, int] = {}
        for char in text[:1000]:  # 只检查前 1000 字符
            char_counts[char] = char_counts.get(char, 0) + 1
        
        if char_counts:
            max_count = max(char_counts.values())
            if max_count / min(len(text), 1000) > threshold:
                return True
        
        return False


class InputSanitizerMiddleware:
    """
    EventBus 中间件版本

    使用示例:
        ```python
        event_bus = EventBus()
        event_bus.add_middleware(InputSanitizerMiddleware())
        ```
    """

    def __init__(self, sanitizer: Optional[InputSanitizer] = None):
        """
        参数:
            sanitizer: 可选自定义清洗器。
                       - 为 None 时，默认使用 gecko.plugins.guardrails.sanitizer.InputSanitizer
                       - 若显式传入自定义实现，则按原样使用（保持向后兼容）
        """
        # 如果调用方显式传入 sanitizer，则保持兼容；
        # 否则使用公开的核心 InputSanitizer 实现，避免本模块和 sanitizer.py 两套规则长期分叉。
        self.sanitizer = sanitizer or PublicInputSanitizer()
    
    async def __call__(self, event: BaseEvent) -> Optional[BaseEvent]:
        """中间件入口"""
        if event.type in ("run_started", "stream_started"):
            input_data = event.data.get("input")
            
            if isinstance(input_data, str):
                result = self.sanitizer.sanitize(input_data)
                if result.was_modified:
                    event.data["input"] = result.sanitized_text
                    event.data["_security_modified"] = True
                    event.data["_threat_level"] = result.threat_level.value
        
        return event


# ==================== 导出 ====================

__all__ = [
    "InputSanitizer",
    "InputSanitizerMiddleware",
    "SanitizationResult",
    "ThreatLevel",
]