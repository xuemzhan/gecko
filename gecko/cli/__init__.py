# gecko/cli/__init__.py
"""
Gecko CLI 工具

提供命令行交互能力，支持：
- 快速测试 Agent
- 工作流调试
- 配置验证
"""
from __future__ import annotations

import asyncio
import sys
from typing import Optional

# 尝试导入 click
try:
    import click
    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False


def main():
    """CLI 入口点"""
    if not CLICK_AVAILABLE:
        print("CLI requires 'click' package. Install with: pip install click")
        sys.exit(1)
    
    cli()

from gecko.version import __version__

if CLICK_AVAILABLE:
    @click.group() # type: ignore
    @click.version_option(version=__version__) # type: ignore
    def cli():
        """Gecko AI Agent Framework CLI"""
        pass
    
    @cli.command()
    @click.option("--model", "-m", default="gpt-3.5-turbo", help="Model name") # type: ignore
    @click.option("--api-key", "-k", envvar="OPENAI_API_KEY", help="API Key") # type: ignore
    @click.option("--system", "-s", default=None, help="System prompt") # type: ignore
    def chat(model: str, api_key: Optional[str], system: Optional[str]):
        """Interactive chat with an agent"""
        if not api_key:
            click.echo("Error: API key required. Use --api-key or set OPENAI_API_KEY") # type: ignore
            return
        
        asyncio.run(_run_chat(model, api_key, system))
    
    @cli.command()
    @click.argument("workflow_file") # type: ignore
    @click.option("--input", "-i", default=None, help="Input data (JSON)") # type: ignore
    @click.option("--visualize", "-v", is_flag=True, help="Print workflow graph") # type: ignore
    def run(workflow_file: str, input: Optional[str], visualize: bool):
        """Execute a workflow from file"""
        click.echo(f"Running workflow: {workflow_file}") # type: ignore
        # TODO: 实现工作流加载和执行
    
    @cli.command()
    def config():
        """Show current configuration"""
        from gecko.config import get_settings
        
        settings = get_settings()
        click.echo("Current Gecko Configuration:") # type: ignore
        click.echo(f"  Default Model: {settings.default_model}") # type: ignore
        click.echo(f"  Max Turns: {settings.max_turns}") # type: ignore
        click.echo(f"  Log Level: {settings.log_level}") # type: ignore
        click.echo(f"  Storage URL: {settings.default_storage_url}") # type: ignore
    
    @cli.command()
    def tools():
        """List available tools"""
        from gecko.plugins.tools.registry import ToolRegistry
        
        # 导入标准工具以触发注册
        try:
            import gecko.plugins.tools.standard  # noqa
        except ImportError:
            pass
        
        tool_list = ToolRegistry.list_tools()
        
        if not tool_list:
            click.echo("No tools registered") # type: ignore
            return
        
        click.echo("Available Tools:") # type: ignore
        for name in tool_list:
            click.echo(f"  - {name}") # type: ignore


async def _run_chat(model: str, api_key: str, system: Optional[str]):
    """运行交互式聊天"""
    from gecko import AgentBuilder
    from gecko.plugins.models import OpenAIChat
    
    click.echo(f"Starting chat with {model}...") # type: ignore
    click.echo("Type 'exit' or 'quit' to end the conversation.\n") # type: ignore
    
    # 创建模型和 Agent
    llm = OpenAIChat(api_key=api_key, model=model)
    
    builder = AgentBuilder().with_model(llm)
    if system:
        builder = builder.with_system_prompt(system)
    
    agent = builder.build()
    
    # 交互循环
    while True:
        try:
            user_input = click.prompt("You", type=str) # type: ignore
            
            if user_input.lower() in ("exit", "quit"): # type: ignore
                click.echo("Goodbye!") # type: ignore
                break
            
            # 执行推理
            result = await agent.run(user_input)
            click.echo(f"\nAssistant: {result.content}\n") # type: ignore
            
        except KeyboardInterrupt:
            click.echo("\nGoodbye!") # type: ignore
            break
        except Exception as e:
            click.echo(f"\nError: {e}\n") # type: ignore


# ==================== 导出 ====================

__all__ = ["main", "cli"]