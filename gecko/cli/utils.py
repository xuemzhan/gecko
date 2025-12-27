# gecko/cli/utils.py
"""
CLI 辅助工具模块

职责：
1. async_cmd: 将异步函数包装为 Click 可调用的同步函数。
2. UI 组件: 封装 rich 库，提供降级方案（当用户未安装 rich 时回退到 print）。
"""
from __future__ import annotations

import asyncio
import functools
import sys
from typing import Any, Callable, Coroutine, List, Optional

# 必须依赖
try:
    import click
except ImportError:
    print("CLI requires 'click'. Install with: pip install click")
    sys.exit(1)

# 可选依赖：Rich (用于美化终端输出)
try:
    from rich.console import Console
    from rich.table import Table
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.status import Status
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # 定义空类型以通过静态检查
    Console = Any # type: ignore
    Table = Any # type: ignore
    Markdown = Any # type: ignore
    Panel = Any # type: ignore
    Status = Any # type: ignore

# 全局 Console 实例 (单例)
console = Console() if RICH_AVAILABLE else None # type: ignore

def async_cmd(f: Callable[..., Coroutine]) -> Callable:
    """
    装饰器：将异步 Click 命令包装为同步执行。
    
    Click 本身不支持 async def，需要通过 asyncio.run() 桥接。
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        # Windows 平台下 ProactorEventLoop 的特殊处理（如果需要）
        # 这里使用标准 run
        return asyncio.run(f(*args, **kwargs))
    return wrapper

# ========== UI 辅助函数 (支持降级) ==========

def print_markdown(text: str):
    """渲染 Markdown 文本"""
    if RICH_AVAILABLE:
        console.print(Markdown(text)) # type: ignore
    else:
        # 降级：直接打印文本
        click.echo(text)

def print_panel(text: str, title: str = "", style: str = "blue"):
    """渲染带边框的面板"""
    if RICH_AVAILABLE:
        console.print(Panel(text, title=title, border_style=style)) # type: ignore
    else:
        # 降级：简单的分隔线
        if title:
            click.echo(f"--- {title} ---")
        click.echo(text)
        if title:
            click.echo("-" * (len(title) + 8))

def print_table(title: str, columns: List[str], rows: List[List[str]]):
    """渲染表格"""
    if RICH_AVAILABLE:
        table = Table(title=title) # type: ignore
        for col in columns:
            table.add_column(col)
        for row in rows:
            table.add_row(*row)
        console.print(table) # type: ignore
    else:
        # 降级：简单的竖线分割
        click.echo(f"\n{title}:")
        header = " | ".join(columns)
        click.echo(header)
        click.echo("-" * len(header))
        for row in rows:
            click.echo(" | ".join(str(r) for r in row))

def print_error(msg: str):
    """打印错误信息（红色）"""
    if RICH_AVAILABLE:
        console.print(f"[bold red]Error:[/bold red] {msg}") # type: ignore
    else:
        click.secho(f"Error: {msg}", fg="red", err=True)

def print_info(msg: str):
    """打印提示信息（蓝色）"""
    if RICH_AVAILABLE:
        console.print(f"[bold blue]Info:[/bold blue] {msg}") # type: ignore
    else:
        click.secho(f"Info: {msg}", fg="blue")

class SpinnerContext:
    """加载动画上下文管理器 (兼容层)"""
    def __init__(self, msg: str):
        self.msg = msg
        self.status = None
    
    def __enter__(self):
        if RICH_AVAILABLE:
            self.status = console.status(f"[bold green]{self.msg}[/bold green]", spinner="dots") # type: ignore
            self.status.start()
        else:
            click.echo(f"{self.msg}...", nl=False)
            sys.stdout.flush()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if RICH_AVAILABLE and self.status:
            self.status.stop()
        else:
            click.echo(" Done.")