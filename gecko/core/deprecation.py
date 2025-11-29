# gecko/core/deprecation.py

import functools
import warnings
from typing import Optional, Callable

def deprecated(
    version: str,
    alternative: Optional[str] = None,
    removal_version: Optional[str] = None
) -> Callable:
    """
    标记函数为废弃
    
    示例:
        @deprecated("0.3.0", alternative="new_function", removal_version="1.0.0")
        def old_function():
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            msg = f"{func.__name__} is deprecated since v{version}."
            if alternative:
                msg += f" Use {alternative} instead."
            if removal_version:
                msg += f" Will be removed in v{removal_version}."
            
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        
        # 更新文档
        deprecation_note = f"\n\n.. deprecated:: {version}\n"
        if alternative:
            deprecation_note += f"   Use :func:`{alternative}` instead.\n"
        
        wrapper.__doc__ = (func.__doc__ or "") + deprecation_note
        return wrapper
    
    return decorator