# gecko/cli/commands/chat.py
from __future__ import annotations

import click
from typing import Optional

from gecko.cli.utils import async_cmd, print_markdown, print_panel, print_error, print_info, SpinnerContext

@click.command()
@click.option("--model", "-m", default="gpt-4o", help="模型名称 (例如 gpt-4o, ollama/llama3)")
@click.option("--api-key", "-k", envvar="OPENAI_API_KEY", help="API Key (默认读取环境变量)")
@click.option("--system", "-s", default=None, help="系统提示词 (System Prompt)")
@click.option("--temperature", "-t", default=0.7, type=float, help="采样温度")
@async_cmd
async def chat(model: str, api_key: Optional[str], system: Optional[str], temperature: float):
    """启动交互式对话会话 (Interactive Chat)。"""
    
    # [性能优化] 延迟导入核心库，加快 CLI 帮助信息显示速度
    try:
        from gecko import AgentBuilder
        from gecko.plugins.models import OpenAIChat, OllamaChat
        from gecko.core.output import AgentOutput
    except ImportError as e:
        print_error(f"无法导入 Gecko 组件，请检查安装: {e}")
        return

    # API Key 检查 (本地模型如 Ollama 除外)
    if "ollama" not in model.lower() and not api_key:
        print_error("未提供 API Key。请使用 --api-key 或设置 OPENAI_API_KEY 环境变量。")
        return

    # 打印会话信息面板
    print_panel(
        f"Model: {model}\nSystem: {system or 'Default'}\nTemp: {temperature}", 
        title="Gecko Chat Session", 
        style="green"
    )
    print_info("输入 'exit', 'quit' 或 'bye' 结束对话。")

    # 初始化 Agent
    try:
        # 简单工厂逻辑：根据模型名前缀判断
        if model.startswith("ollama"):
            # 移除前缀传给 OllamaChat
            model_name = model.split("/", 1)[1] if "/" in model else model
            llm = OllamaChat(model=model_name, temperature=temperature)
        else:
            llm = OpenAIChat(api_key=api_key or "dummy", model=model, temperature=temperature)
            
        builder = AgentBuilder().with_model(llm)
        if system:
            builder.with_system_prompt(system)
        agent = builder.build()
        
    except Exception as e:
        print_error(f"Agent 初始化失败: {e}")
        return

    # 交互循环
    while True:
        try:
            # 获取用户输入
            user_input = click.prompt(click.style("You", fg="green", bold=True), type=str)
            
            if user_input.lower() in ("exit", "quit", "bye"):
                print_info("Goodbye!")
                break
            
            # 执行推理 (带加载动画)
            with SpinnerContext("Thinking"):
                # 这里也可以改为 await agent.stream() 来实现打字机效果
                response = await agent.run(user_input)

            # 输出分隔
            click.echo("") 
            click.secho("Assistant:", fg="blue", bold=True)
            
            # 渲染回复
            if isinstance(response, AgentOutput):
                # 渲染主要内容
                print_markdown(response.content)
                
                # 如果有工具调用，显示调试信息
                if response.tool_calls:
                    tool_names = [tc['function']['name'] for tc in response.tool_calls]
                    click.secho(f"\n[Tool Calls]: {', '.join(tool_names)}", fg="yellow", dim=True)
            else:
                print_markdown(str(response))
            
            click.echo("-" * 40) # 视觉分隔符

        except KeyboardInterrupt:
            print_info("\n检测到中断。再次输入 Ctrl+C 或输入 'exit' 退出。")
            continue
        except Exception as e:
            print_error(f"运行时错误: {e}")