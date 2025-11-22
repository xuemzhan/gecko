"""
协议基础工具
提供运行时检查和验证辅助函数。
"""
from __future__ import annotations
import inspect as insp
from typing import Any, List

def check_protocol(obj: Any, protocol: type) -> bool:
    """
    检查对象是否实现了指定协议
    """
    return isinstance(obj, protocol)

def get_missing_methods(obj: Any, protocol: type) -> List[str]:
    """
    获取对象未实现的协议方法
    """
    missing = []
    # 获取协议的所有成员
    for name, value in insp.getmembers(protocol):
        if name.startswith("_"): continue
        
        if insp.isfunction(value) or insp.ismethod(value):
            if not hasattr(obj, name):
                missing.append(name)
            elif not callable(getattr(obj, name)):
                missing.append(name)
        elif insp.isdatadescriptor(value):
            if not hasattr(obj, name):
                missing.append(name)
    return missing