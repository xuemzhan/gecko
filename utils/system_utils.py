# agno/utils/system_utils.py

"""
系统与环境工具模块

该模块提供了与操作系统、shell环境和环境变量交互的辅助函数。
主要功能包括：
- 安全地执行外部shell命令并捕获其输出
- 从环境变量中读取配置，支持默认值和必需项检查
- 下载和管理CA证书文件
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

import httpx

# 常量定义
DEFAULT_TAIL_LINES = 100
DEFAULT_CERT_FILENAME = "cert.pem"
DEFAULT_CERTS_DIR = Path("./certs")
DEFAULT_COMMAND_TIMEOUT = 30  # 秒
DEFAULT_DOWNLOAD_TIMEOUT = 60  # 秒
DOWNLOAD_CHUNK_SIZE = 8192

# 日志配置
logger = logging.getLogger(__name__)


def log_info(message: str) -> None:
    """记录信息级别日志"""
    logger.info(message)


# --- 环境变量处理 ---

def get_from_env(
    key: str, 
    default: Optional[str] = None, 
    required: bool = False
) -> Optional[str]:
    """
    从环境变量中获取值
    
    提供统一的接口来读取环境变量，支持设置默认值或强制要求。
    
    Args:
        key: 环境变量的名称
        default: 环境变量未找到时返回的默认值
        required: 为True且环境变量未找到时抛出异常
        
    Returns:
        环境变量的值、默认值或None
        
    Raises:
        ValueError: 当required为True且环境变量未设置时
        
    Examples:
        >>> os.environ["TEST_VAR"] = "value"
        >>> get_from_env("TEST_VAR")
        'value'
        >>> get_from_env("MISSING_VAR", default="default")
        'default'
    """
    value = os.getenv(key, default)
    
    if required and value is None:
        raise ValueError(f"必需的环境变量 '{key}' 未设置")
    
    return value


# --- Shell 命令执行 ---

def run_shell_command(
    args: list, 
    tail: int = DEFAULT_TAIL_LINES,
    timeout: int = DEFAULT_COMMAND_TIMEOUT
) -> str:
    """
    执行外部shell命令并返回其输出
    
    为了安全和可预测性，命令和参数以列表形式传递。
    命令执行失败时返回错误信息。
    
    Args:
        args: 要执行的命令及其参数列表，如 ['ls', '-l']
        tail: 仅返回标准输出的最后N行，避免输出过长
        timeout: 命令执行超时时间（秒）
        
    Returns:
        命令的标准输出（最后tail行）或错误信息
        
    Examples:
        >>> run_shell_command(["echo", "hello"])
        'hello'
    """
    if not args:
        error_msg = "错误: 未提供任何命令"
        logger.error(error_msg)
        return error_msg
    
    log_info(f"执行shell命令: {' '.join(args)}")
    
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            encoding='utf-8',
            errors='replace'  # 处理无法解码的字符
        )
        
        if result.returncode != 0:
            error_message = result.stderr.strip() or "命令执行失败"
            logger.error(
                f"Shell命令执行失败 (返回码 {result.returncode}): {error_message}"
            )
            return f"错误: {error_message}"
        
        # 返回标准输出的最后几行
        stdout = result.stdout.strip()
        if not stdout:
            return ""
        
        stdout_lines = stdout.split('\n')
        return "\n".join(stdout_lines[-tail:]) if tail > 0 else stdout
    
    except subprocess.TimeoutExpired:
        error_msg = f"错误: 命令执行超时（{timeout}秒）"
        logger.error(error_msg)
        return error_msg
    
    except FileNotFoundError:
        error_msg = (
            f"错误: 命令 '{args[0]}' 未找到。"
            f"请确保它已安装并在系统的PATH中"
        )
        logger.error(error_msg)
        return error_msg
    
    except Exception as e:
        error_msg = f"执行shell命令时发生异常: {e}"
        logger.error(error_msg)
        return f"异常: {e}"


# --- 证书管理 ---

def download_cert(
    cert_url: str,
    certs_dir: Path = DEFAULT_CERTS_DIR,
    filename: str = DEFAULT_CERT_FILENAME,
    timeout: int = DEFAULT_DOWNLOAD_TIMEOUT
) -> str:
    """
    从指定URL下载CA证书包（如果本地不存在）
    
    适用于需要自定义证书的应用场景。
    
    Args:
        cert_url: 证书包的下载URL
        certs_dir: 存储证书的本地目录
        filename: 保存证书的文件名
        timeout: 下载超时时间（秒）
        
    Returns:
        本地证书文件的绝对路径
        
    Raises:
        httpx.HTTPError: 下载失败时
        
    Examples:
        >>> cert_path = download_cert("https://example.com/cert.pem")
        >>> Path(cert_path).exists()
        True
    """
    cert_path = certs_dir / filename
    
    # 证书已存在，直接返回
    if cert_path.exists():
        log_info(f"证书文件已存在: {cert_path.absolute()}")
        return str(cert_path.absolute())
    
    log_info(f"从 {cert_url} 下载证书...")
    
    # 创建证书目录
    certs_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用临时文件，下载完成后再重命名，确保原子性
    temp_path = cert_path.with_suffix('.tmp')
    
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            with client.stream("GET", cert_url) as response:
                response.raise_for_status()
                
                with open(temp_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=DOWNLOAD_CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
        
        # 下载成功，重命名临时文件
        temp_path.rename(cert_path)
        log_info(f"证书下载成功: {cert_path.absolute()}")
        
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        error_msg = f"下载证书失败: {e}"
        logger.error(error_msg)
        # 清理临时文件
        if temp_path.exists():
            temp_path.unlink()
        raise
    
    except Exception as e:
        logger.error(f"下载证书时发生未知错误: {e}")
        # 清理临时文件
        if temp_path.exists():
            temp_path.unlink()
        raise
    
    return str(cert_path.absolute())


# --- 测试代码 ---

def _test_get_from_env() -> None:
    """测试环境变量获取功能"""
    print("\n[1] 测试 get_from_env:")
    
    test_key = "AGNO_TEST_VAR"
    test_value = "test_value"
    os.environ[test_key] = test_value
    
    try:
        # 测试读取存在的变量
        result = get_from_env(test_key)
        print(f"  ✓ 读取存在的变量 '{test_key}': '{result}'")
        
        # 测试读取不存在的变量（带默认值）
        result = get_from_env('AGNO_NON_EXISTENT', default='default_val')
        print(f"  ✓ 读取不存在的变量（带默认值）: '{result}'")
        
        # 测试读取不存在的变量（无默认值）
        result = get_from_env('AGNO_NON_EXISTENT')
        print(f"  ✓ 读取不存在的变量（无默认值）: {result}")
        
        # 测试必需变量
        try:
            get_from_env("AGNO_REQUIRED_VAR", required=True)
            print("  ✗ 应该抛出异常但没有")
        except ValueError as e:
            print(f"  ✓ 成功捕获必需变量缺失的异常: {e}")
    
    finally:
        os.environ.pop(test_key, None)


def _test_run_shell_command() -> None:
    """测试shell命令执行功能"""
    print("\n[2] 测试 run_shell_command:")
    
    # 测试成功的命令
    output = run_shell_command(["python", "--version"])
    print(f"  ✓ 'python --version' 输出:\n    {output}")
    
    # 测试不存在的命令
    output = run_shell_command(["non_existent_command_xyz"])
    print(f"  ✓ 无效命令输出: {output[:60]}...")
    
    # 测试空命令列表
    output = run_shell_command([])
    print(f"  ✓ 空命令输出: {output}")


def _test_download_cert() -> None:
    """测试证书下载功能"""
    print("\n[3] 测试 download_cert:")
    
    temp_certs_dir = Path("./temp_certs_test")
    cert_url = "https://curl.se/ca/cacert.pem"
    
    try:
        print(f"  正在下载证书到 '{temp_certs_dir}'...")
        cert_path_str = download_cert(cert_url, certs_dir=temp_certs_dir)
        cert_path = Path(cert_path_str)
        
        file_size = cert_path.stat().st_size
        print(f"  ✓ 下载成功，路径: {cert_path}")
        print(f"  ✓ 文件大小: {file_size:,} 字节")
        
        # 测试缓存
        print("  测试证书缓存...")
        cached_path = download_cert(cert_url, certs_dir=temp_certs_dir)
        print(f"  ✓ 使用已存在的证书: {cached_path}")
        
    except Exception as e:
        print(f"  ⚠ 下载证书时出错（可能是网络问题）: {e}")
    
    finally:
        # 清理
        import shutil
        if temp_certs_dir.exists():
            shutil.rmtree(temp_certs_dir)
            print(f"  ✓ 临时目录已清理")


def _run_tests() -> None:
    """运行所有测试"""
    print("=" * 60)
    print("运行 agno/utils/system_utils.py 测试")
    print("=" * 60)
    
    _test_get_from_env()
    _test_run_shell_command()
    _test_download_cert()
    
    print("\n" + "=" * 60)
    print("✓ 测试完成")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    _run_tests()