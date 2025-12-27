# cli_demo.py
"""
Gecko CLI 功能演示脚本

本脚本演示如何通过命令行接口与 Gecko 交互。
它使用 click.testing.CliRunner 模拟终端操作，
但在实际使用中，你通常直接在终端输入 `gecko <command> ...`。
"""
import os
import json
import sys
from click.testing import CliRunner
from gecko.cli.main import cli

# 设置演示用的 API Key (来自你的要求)
# 在实际生产环境中，请不要将 Key 硬编码在代码中
DEMO_API_KEY = "3bd5e6fdc377489c80dbb435b84d7560.izN8bDXCVR1FNSCVR1FNSYS" # Masked mostly for safety in display, assume it matches provided
# 修正为提供的真实 Key (仅用于演示环境)
REAL_API_KEY = "3bd5e6fdc377489c80dbb435b84d7560.izN8bDXCVR1FNSYS" 
# 注意：你提供的 Key 格式似乎包含后缀，这里直接使用原值
os.environ["ZHIPU_API_KEY"] = "3bd5e6fdc377489c80dbb435b84d7560.izN8bDXCVR1FNSYS"

def print_separator(title):
    print(f"\n{'='*20} {title} {'='*20}")

def main():
    runner = CliRunner()
    
    print(f"🚀 Gecko CLI Demo Started (PID: {os.getpid()})")
    
    # ---------------------------------------------------------
    # 1. 演示 `gecko config`
    # 查看当前框架的全局配置
    # ---------------------------------------------------------
    print_separator("演示: gecko config")
    result = runner.invoke(cli, ["config"])
    print(result.output)

    # ---------------------------------------------------------
    # 2. 演示 `gecko tools`
    # 列出当前系统中注册的所有工具
    # ---------------------------------------------------------
    print_separator("演示: gecko tools --verbose")
    result = runner.invoke(cli, ["tools", "-v"])
    print(result.output)

    # ---------------------------------------------------------
    # 3. 演示 `gecko run`
    # 运行上面定义的 my_workflow.py 文件
    # 这是生产环境中最常用的方式：定义复杂的 Python 逻辑，通过 CLI 触发
    # ---------------------------------------------------------
    # [核心修改] 获取当前脚本所在的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 拼接出 workflow 文件的完整路径
    workflow_path = os.path.join(current_dir, "my_workflow.py")
    
    print_separator(f"演示: gecko run {os.path.basename(workflow_path)}")
    
    # 确保 my_workflow.py 存在
    if not os.path.exists(workflow_path):
        print(f"❌ 错误: 找不到文件: {workflow_path}")
        print("请确保你已经创建了 'my_workflow.py' 并将其放在 'examples/cli/' 目录下。")
        return

    # 模拟用户输入 JSON 数据
    input_payload = json.dumps("请简要介绍一下 Gecko 框架的设计理念。")
    
    # 调用 CLI (传入绝对路径)
    result = runner.invoke(cli, ["run", workflow_path, "--input", input_payload])
    
    # 打印结果
    print(result.output)
    
    if result.exit_code != 0:
        print("❌ 演示运行失败，请检查 API Key 是否有效或网络连接。")
    else:
        print("✅ 演示运行成功！")

    # ---------------------------------------------------------
    # 4. 关于 `gecko chat` 的说明
    # ---------------------------------------------------------
    print_separator("关于 gecko chat")
    print("`gecko chat` 命令提供交互式终端对话。")
    print("由于它是交互式的，不便在此脚本中自动演示。")
    print("你可以直接在终端运行以下命令来体验（默认使用 OpenAI 协议）：")
    print(f"\n  export OPENAI_API_KEY='your-key'")
    print(f"  gecko chat --model gpt-4o\n")
    print("或者如果想使用 Ollama 本地模型：")
    print(f"\n  gecko chat --model ollama/llama3\n")

if __name__ == "__main__":
    main()