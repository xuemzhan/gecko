# gecko/plugins/storage/utils.py
"""
Storage 插件工具函数

提供统一的 URL 解析和验证逻辑。
"""
from __future__ import annotations

from urllib.parse import parse_qs, urlparse
from typing import Dict, Tuple, Optional


def parse_storage_url(url: str) -> Tuple[str, str, Dict[str, str]]:
    """
    解析存储 URL
    
    格式：scheme://path?param1=value1&param2=value2
    
    返回：(scheme, path, params)
    
    示例:
        "sqlite:///./data.db" -> ("sqlite", "./data.db", {})
        "sqlite://:memory:" -> ("sqlite", ":memory:", {})
    """
    if "://" not in url:
        raise ValueError(f"Invalid storage URL: '{url}'. Must include scheme.")
    
    parsed = urlparse(url)
    scheme = parsed.scheme
    
    # 解析路径
    if scheme == "sqlite":
        # 特殊处理 sqlite
        # sqlite:///foo.db -> path = /foo.db (urlparse behavior)
        # 我们需要去掉开头的 / 变成相对路径，或者保留绝对路径
        if parsed.netloc:
            # sqlite://:memory: -> netloc=':memory:'
            path = parsed.netloc
        else:
            # sqlite:///./data.db -> path='/./data.db'
            path = parsed.path
            if path.startswith("/") and len(path) > 1 and path[1] == ".":
                # 修正相对路径 /./data.db -> ./data.db
                path = path[1:]
            elif path.startswith("/"):
                 # 绝对路径保持不变，或者根据 OS 调整
                 # 这里简化处理，对于 Windows 可能需要更复杂的逻辑
                 pass
    else:
        path = f"{parsed.netloc}{parsed.path}"

    # 解析参数
    params: Dict[str, str] = {}
    if parsed.query:
        query_dict = parse_qs(parsed.query)
        params = {k: v[0] for k, v in query_dict.items()}
    
    return scheme, path, params