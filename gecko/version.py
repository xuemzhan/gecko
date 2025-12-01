# gecko/version.py
"""
版本管理模块

通过单一来源 (Single Source of Truth) 管理 Gecko 框架版本号：
- 避免 __init__.py、CLI 等多个地方手写版本号导致不一致
- 方便在打包/CI 时统一维护
"""

__version__: str = "0.3.1"  # 请根据真实发布版本号修改
__author__: str = "Xuemzhan"
__email__: str = "zxm0813@gmail.com"
__license__: str = "MIT"