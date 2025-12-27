# gecko/cli/__init__.py
"""
Gecko CLI 工具包入口

职责：
1. 暴露 main 函数作为 entry_point (在 pyproject.toml 中配置)。
2. 处理全局异常，避免向用户展示不友好的 Traceback (除非开启调试模式)。
"""
import os
import sys
import click

# 延迟导入 main，避免在包导入阶段触发复杂的依赖加载
from gecko.cli.main import cli

def main():
    """CLI 应用程序入口点"""
    try:
        # 启动 Click 命令组
        cli()
    except Exception as e:
        # 检查是否开启了调试模式
        debug_mode = os.getenv("GECKO_DEBUG", "0").lower() in ("1", "true", "yes")
        
        if debug_mode:
            # 调试模式下，抛出完整堆栈
            raise e
        else:
            # 生产模式下，仅打印红色错误信息并以非零状态码退出
            click.secho(f"Critical Error: {e}", fg="red", err=True)
            click.echo("Hint: Set GECKO_DEBUG=1 to see full traceback.")
            sys.exit(1)

__all__ = ["main"]