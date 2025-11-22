# gecko/plugins/tools/standard/__init__.py
"""
Gecko 标准工具库

包含一组经过安全审查和优化的内置工具。
导入此模块时，会自动将工具注册到 ToolRegistry。
"""
from gecko.plugins.tools.standard.calculator import CalculatorTool
from gecko.plugins.tools.standard.duckduckgo import DuckDuckGoSearchTool

# 可以在此定义 lazy_load 逻辑，目前为了简单直接导入以触发注册
__all__ = [
    "CalculatorTool",
    "DuckDuckGoSearchTool",
]