# gecko/plugins/storage/utils.py
"""
Storage 插件工具函数

提供：
1. 统一的 URL 解析
2. 连接参数验证
3. URL 构建工具
"""
from __future__ import annotations
from urllib.parse import urlparse, parse_qs, urlencode
from typing import Tuple, Dict, Any
from gecko.core.exceptions import StorageError

def parse_storage_url(url: str) -> Tuple[str, str, Dict[str, str]]:
    """
    解析存储 URL
    
    格式：scheme://path?param1=value1&param2=value2
    
    返回：(scheme, path, params)
    
    示例:
        parse_storage_url("sqlite://./data.db?timeout=30")
        => ("sqlite", "./data.db", {"timeout": "30"})
        
        parse_storage_url("redis://localhost:6379/0?password=secret")
        => ("redis", "localhost:6379/0", {"password": "secret"})
    """
    if "://" not in url:
        raise StorageError(
            f"Invalid storage URL: '{url}'. Must include scheme (e.g., sqlite://)",
            context={"url": url}
        )
    
    parsed = urlparse(url)
    
    # 1. 提取 scheme
    scheme = parsed.scheme
    if not scheme:
        raise StorageError(
            f"Missing scheme in URL: '{url}'",
            context={"url": url}
        )
    
    # 2. 提取 path
    # 对于 sqlite://./data.db，path 是 './data.db'
    # 对于 redis://localhost:6379，netloc 是 'localhost:6379'
    if parsed.path and parsed.path != "/":
        # 有路径，使用路径
        path = parsed.path
        if path.startswith("/"):
            path = path[1:]  # 移除开头的 /
        
        # 如果有 netloc，拼接
        if parsed.netloc:
            path = f"{parsed.netloc}/{path}"
    else:
        # 无路径或路径为 /，使用 netloc
        path = parsed.netloc or ""
    
    # 3. 解析查询参数
    params: Dict[str, str] = {}
    if parsed.query:
        query_dict = parse_qs(parsed.query)
        # 只取第一个值（简化处理）
        params = {k: v[0] for k, v in query_dict.items()}
    
    return scheme, path, params

def build_storage_url(
    scheme: str,
    path: str,
    **params
) -> str:
    """
    构建存储 URL
    
    示例:
        build_storage_url("sqlite", "./data.db", timeout=30)
        => "sqlite://./data.db?timeout=30"
    """
    url = f"{scheme}://{path}"
    
    if params:
        query = urlencode(params)
        url += f"?{query}"
    
    return url

def validate_storage_url(url: str, required_scheme: str | None = None):
    """
    验证存储 URL
    
    参数:
        url: URL 字符串
        required_scheme: 要求的 scheme（如 'sqlite'）
    
    抛出:
        StorageError: 如果验证失败
    """
    try:
        scheme, path, params = parse_storage_url(url)
    except Exception as e:
        raise StorageError(
            f"Invalid storage URL: {e}",
            context={"url": url}
        ) from e
    
    # 检查 scheme
    if required_scheme and scheme != required_scheme:
        raise StorageError(
            f"Invalid scheme '{scheme}', expected '{required_scheme}'",
            context={"url": url, "expected": required_scheme, "actual": scheme}
        )
    
    # 检查 path
    if not path and scheme not in ["memory"]:  # :memory: 可以没有 path
        raise StorageError(
            f"Missing path in URL: '{url}'",
            context={"url": url}
        )

# ========== 常用 URL 工厂 ==========

def make_sqlite_url(
    db_path: str = "./gecko_data.db",
    timeout: int | None = None
) -> str:
    """创建 SQLite URL"""
    params = {}
    if timeout:
        params["timeout"] = str(timeout)
    
    return build_storage_url("sqlite", db_path, **params)

def make_redis_url(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    password: str | None = None
) -> str:
    """创建 Redis URL"""
    path = f"{host}:{port}/{db}"
    params = {}
    if password:
        params["password"] = password
    
    return build_storage_url("redis", path, **params)

def make_postgres_url(
    user: str,
    password: str,
    host: str = "localhost",
    port: int = 5432,
    database: str = "gecko"
) -> str:
    """创建 PostgreSQL URL"""
    path = f"{user}:{password}@{host}:{port}/{database}"
    return build_storage_url("postgres+pgvector", path)