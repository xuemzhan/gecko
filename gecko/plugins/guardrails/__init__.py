# gecko/plugins/guardrails/__init__.py
"""
Guardrails 总入口

- InputSanitizer / ScanResult / ThreatLevel: 基础扫描与清洗（sanitizer.py）
- InputSanitizerMiddleware / SanitizationResult: 事件总线中间件版本（input_sanitizer.py）
"""
from gecko.plugins.guardrails.sanitizer import (
    InputSanitizer,
    ScanResult,
    ThreatLevel,
)
from gecko.plugins.guardrails.input_sanitizer import (
    InputSanitizerMiddleware,
    SanitizationResult,
)

__all__ = [
    "InputSanitizer",
    "ScanResult",
    "ThreatLevel",
    "InputSanitizerMiddleware",
    "SanitizationResult",
]