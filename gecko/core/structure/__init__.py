# gecko/core/structure/__init__.py
"""
结构化输出子模块入口

本文件用于对外暴露稳定的公共接口，并隐藏内部实现细节。
通过这样的方式，我们可以在不破坏外部调用方的前提下，
自由地重构内部模块结构和实现。

对外主要暴露以下对象：
- StructureEngine: 核心结构化输出引擎
- StructureParseError: 结构化解析失败异常
- parse_structured_output: 同步封装的便捷函数
- extract_json_from_text: 轻量级 JSON 提取工具函数
- ExtractionStrategy / register_extraction_strategy: 策略插件扩展接口
"""

from gecko.core.structure.errors import StructureParseError
from gecko.core.structure.engine import StructureEngine
from gecko.core.structure.sync import parse_structured_output, extract_json_from_text
from gecko.core.structure.json_extractor import ExtractionStrategy, register_extraction_strategy

__all__ = [
    "StructureEngine",
    "StructureParseError",
    "parse_structured_output",
    "extract_json_from_text",
    "ExtractionStrategy",
    "register_extraction_strategy",
]
