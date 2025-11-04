# agno/utils/network_utils.py

"""
网络工具模块

该模块提供了一系列与网络操作相关的辅助函数。
主要功能包括：
- 带有指数退避重试机制的同步和异步 HTTP GET 请求。
- 使用 IP 地理位置 API 获取用户的大致地理位置。
- 在用户的默认浏览器中打开本地 HTML 文件。
"""

import asyncio
import webbrowser
from pathlib import Path
from time import sleep
from typing import Any, Dict, Optional, TypedDict

import httpx
import logging

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 2  # 重试间隔：1s, 2s, 4s...


# --- 地理位置数据结构 ---
class LocationInfo(TypedDict, total=False):
    city: Optional[str]
    region: Optional[str]
    country: Optional[str]


# --- HTTP 请求与重试 ---

def fetch_with_retry(
    url: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    proxy: Optional[str] = None,
) -> httpx.Response:
    """
    执行带指数退避重试的同步 HTTP GET 请求。

    Args:
        url: 目标 URL（自动 strip 空格）。
        max_retries: 最大重试次数。
        backoff_factor: 退避因子（等待时间 = backoff_factor ** attempt 秒）。
        proxy: 代理 URL。

    Returns:
        成功的 HTTP 响应。

    Raises:
        httpx.RequestError: 所有重试失败。
        httpx.HTTPStatusError: HTTP 4xx/5xx 错误。
    """
    clean_url = url.strip()
    for attempt in range(max_retries):
        try:
            with httpx.Client(proxy=proxy, timeout=10.0) as client:
                response = client.get(clean_url, follow_redirects=True)
                response.raise_for_status()
                return response
        except httpx.RequestError as e:
            if attempt == max_retries - 1:
                logger.error(f"同步请求失败（{max_retries} 次重试后）: {clean_url} - {e}")
                raise
            wait_time = backoff_factor ** attempt
            logger.warning(f"同步请求失败（第 {attempt + 1}/{max_retries} 次），{wait_time:.1f} 秒后重试...")
            sleep(wait_time)
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP 错误 {e.response.status_code} for {clean_url}: {e.response.text[:100]}")
            raise

    raise httpx.RequestError(f"Unexpected: all retries exhausted for {clean_url}")


async def async_fetch_with_retry(
    url: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    proxy: Optional[str] = None,
) -> httpx.Response:
    """
    执行带指数退避重试的异步 HTTP GET 请求。
    """
    clean_url = url.strip()
    # 复用 AsyncClient 提升性能
    async with httpx.AsyncClient(proxy=proxy, timeout=10.0) as client:
        for attempt in range(max_retries):
            try:
                response = await client.get(clean_url, follow_redirects=True)
                response.raise_for_status()
                return response
            except httpx.RequestError as e:
                if attempt == max_retries - 1:
                    logger.error(f"异步请求失败（{max_retries} 次重试后）: {clean_url} - {e}")
                    raise
                wait_time = backoff_factor ** attempt
                logger.warning(f"异步请求失败（第 {attempt + 1}/{max_retries} 次），{wait_time:.1f} 秒后重试...")
                await asyncio.sleep(wait_time)
            except httpx.HTTPStatusError as e:
                logger.error(f"异步 HTTP 错误 {e.response.status_code} for {clean_url}: {e.response.text[:100]}")
                raise

    raise httpx.RequestError(f"Unexpected: all retries exhausted for {clean_url}")


# --- 地理位置 ---

def get_location() -> LocationInfo:
    """
    通过 IP 查询用户地理位置（城市、地区、国家）。

    使用服务：
    1. api.ipify.org 获取公网 IP
    2. ip-api.com 获取地理位置

    Returns:
        包含 'city', 'region', 'country' 的字典；失败时返回空字典。
    """
    try:
        # 1. 获取公网 IP
        ip_resp = httpx.get("https://api.ipify.org?format=json", timeout=5)
        ip_resp.raise_for_status()
        ip = ip_resp.json().get("ip")
        if not ip:
            raise ValueError("无法从 ipify 获取 IP")

        # 2. 查询地理位置
        loc_resp = httpx.get(f"http://ip-api.com/json/{ip}", timeout=5)
        loc_resp.raise_for_status()
        data = loc_resp.json()

        if data.get("status") == "success":
            return LocationInfo(
                city=data.get("city"),
                region=data.get("regionName"),
                country=data.get("country"),
            )
        else:
            logger.warning(f"ip-api.com 查询失败: {data.get('message', 'Unknown error')}")
            return {}

    except Exception as e:
        logger.warning(f"获取地理位置失败: {e}")
        return {}


# --- 浏览器操作 ---

def open_html_file_in_browser(file_path: Path) -> None:
    """
    在默认浏览器中打开本地 HTML 文件。

    Args:
        file_path: HTML 文件路径。

    Raises:
        FileNotFoundError: 文件不存在。
    """
    abs_path = file_path.resolve()
    if not abs_path.is_file():
        msg = f"文件不存在或不是普通文件: {abs_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    file_url = abs_path.as_uri()
    logger.info(f"在浏览器中打开: {file_url}")

    # 尝试打开浏览器
    try:
        opened = webbrowser.open(file_url)
        if not opened:
            logger.warning("webbrowser.open() 返回 False，可能无法打开浏览器")
    except Exception as e:
        logger.error(f"打开浏览器时出错: {e}")
        raise


# --- 测试代码 ---
if __name__ == "__main__":
    import shutil
    from pathlib import Path

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("--- 正在运行 agno/utils/network_utils.py 的测试代码 ---")

    # 1. 同步请求测试
    print("\n[1] 测试 fetch_with_retry (同步):")
    try:
        resp = fetch_with_retry("https://httpbin.org/get", max_retries=2)
        print(f"  同步请求成功，状态码: {resp.status_code}")
    except Exception as e:
        print(f"  同步请求失败: {e}")

    try:
        fetch_with_retry("http://invalid.example/fail", max_retries=2, backoff_factor=0.1)
    except httpx.RequestError:
        print("  同步失败请求按预期抛出异常")

    # 2. 异步请求测试
    async def _test_async():
        try:
            resp = await async_fetch_with_retry("https://httpbin.org/get", max_retries=2)
            print(f"  异步请求成功，状态码: {resp.status_code}")
        except Exception as e:
            print(f"  异步请求失败: {e}")

        try:
            await async_fetch_with_retry("http://invalid.example/fail", max_retries=2, backoff_factor=0.1)
        except httpx.RequestError:
            print("  异步失败请求按预期抛出异常")

    asyncio.run(_test_async())

    # 3. 地理位置测试
    print("\n[3] 测试 get_location:")
    loc = get_location()
    if loc:
        print(f"  位置: {loc}")
    else:
        print("  无法获取位置（正常，可能因网络或 API 限制）")

    # 4. 浏览器测试
    print("\n[4] 测试 open_html_file_in_browser:")
    tmp_dir = Path("./temp_network_test")
    tmp_dir.mkdir(exist_ok=True)
    html_file = tmp_dir / "test.html"
    html_file.write_text("<h1>Agno 测试页面</h1>", encoding="utf-8")
    print(f"  创建测试文件: {html_file}")

    try:
        input("  按 Enter 在浏览器中打开（或 Ctrl+C 跳过）...")
        open_html_file_in_browser(html_file)
        print("  已尝试打开浏览器")
    except KeyboardInterrupt:
        print("\n  跳过浏览器测试")
    except Exception as e:
        print(f"  浏览器打开失败: {e}")

    # 清理
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"\n临时目录已清理")

    print("\n--- 测试结束 ---")