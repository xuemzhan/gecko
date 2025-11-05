# agno/models/__init__.py
"""
此模块提供了与各种大语言模型（LLM）交互的统一接口。

核心功能:
- `get_model`: 一个工厂函数，用于根据供应商和模型ID获取模型实例。
- `Model`: 所有模型实现的抽象基类。
"""

# 从 utils 模块导出核心工厂函数 get_model，使其可以直接从 agno.models 导入
from .utils import get_model

# 从 base 模块导出 Model 基类，以便外部代码可以进行类型提示或继承
from .base import Model

# 导出所有公开的符号
__all__ = [
    "get_model",
    "Model",
]