# gecko/cli/main.py
"""
Click 主命令组定义
"""
from __future__ import annotations
import click
from gecko.version import __version__

# 导入子命令
from gecko.cli.commands.chat import chat
from gecko.cli.commands.config import config
from gecko.cli.commands.run import run
from gecko.cli.commands.tools import tools

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(version=__version__, prog_name="Gecko AI")
def cli():
    """
    Gecko AI Framework CLI.
    
    用于构建、调试和运行智能体工作流的命令行工具。
    """
    pass

# 注册子命令
cli.add_command(chat)
cli.add_command(config)
cli.add_command(run)
cli.add_command(tools)

if __name__ == "__main__":
    cli()