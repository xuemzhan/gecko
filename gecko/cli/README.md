# Gecko CLI (Command Line Interface)

Gecko CLI 是 Gecko AI 框架的命令行入口，旨在为开发者提供快速调试 Agent、运行工作流、管理配置和查看工具的能力。

## ✨ 特性

*   **交互式对话 (`chat`)**：支持 OpenAI 协议及 Ollama 本地模型，提供流式响应体验。
*   **工作流执行 (`run`)**：直接运行定义在 Python 脚本中的复杂 `Workflow`。
*   **工具管理 (`tools`)**：查看当前环境已注册的 Tool 及其参数 Schema。
*   **配置查看 (`config`)**：查看全局配置，自动脱敏敏感信息。
*   **优雅的 UI**：集成 `rich` 库，提供 Markdown 渲染、加载动画和格式化表格（可选依赖，自动降级）。

## 📦 安装

Gecko CLI 包含在 Gecko 核心包中，但建议安装额外依赖以获得最佳体验：

```bash
# 安装基础包 (CLI 功能可用，但无富文本效果)
pip install gecko-ai

# [推荐] 安装增强 UI 支持 (Rich)
pip install "gecko-ai[cli]" 
# 或者手动安装
pip install rich
```

## 🚀 快速开始

在终端中输入 `gecko` 即可查看帮助信息：

```bash
gecko --help
```

### 全局环境变量

*   `OPENAI_API_KEY`: 默认的 API Key（`chat` 命令会自动读取）。
*   `GECKO_DEBUG`: 设置为 `1` 或 `true` 以开启详细的错误堆栈追踪。

---

## 🛠 命令详解

### 1. `chat` - 交互式对话

启动一个临时的 Agent 进行对话测试。

**用法：**
```bash
gecko chat [OPTIONS]
```

**参数：**
*   `-m, --model <str>`: 模型名称 (默认: `gpt-4o`)。支持 `ollama/llama3` 等格式。
*   `-k, --api-key <str>`: API Key。
*   `-s, --system <str>`: 系统提示词 (System Prompt)。
*   `-t, --temperature <float>`: 采样温度 (默认: 0.7)。

**示例：**

```bash
# 使用 OpenAI
export OPENAI_API_KEY="sk-..."
gecko chat -m gpt-4o -s "你是一个翻译助手"

# 使用智谱 AI (兼容 OpenAI 协议)
gecko chat -m glm-4-flash --api-key "your-zhipu-key" --base-url "https://open.bigmodel.cn/api/paas/v4/"

# 使用 Ollama (本地)
gecko chat -m ollama/llama3
```

### 2. `run` - 运行工作流

加载并执行一个定义了 `Workflow` 的 Python 脚本。这是生产环境中最常用的功能。

**用法：**
```bash
gecko run [OPTIONS] WORKFLOW_FILE
```

**参数：**
*   `WORKFLOW_FILE`: 包含 `workflow` 变量的 Python 脚本路径。
*   `-i, --input <str>`: 输入数据。可以是 JSON 字符串，也可以是 JSON 文件的路径。

**示例：**

```bash
# 运行当前目录下的 my_workflow.py
gecko run my_workflow.py --input "介绍一下量子计算"

# 从文件读取输入
gecko run workflows/rag_flow.py --input ./data/query.json
```

#### 📝 如何编写 Workflow 文件

`gecko run` 要求目标 Python 文件中必须导出一个名为 **`workflow`** 的变量，该变量必须是 `gecko.Workflow` 的实例。

**`my_workflow.py` 示例：**

```python
import os
from gecko import AgentBuilder, Workflow, step
from gecko.plugins.models import ZhipuChat

# 定义节点
@step(name="process_request")
async def my_node(user_input: str):
    api_key = os.getenv("ZHIPU_API_KEY")
    agent = AgentBuilder()\
        .with_model(ZhipuChat(api_key=api_key, model="glm-4-flash"))\
        .build()
    result = await agent.run(user_input)
    return result.content

# 定义工作流
wf = Workflow(name="DemoFlow")
wf.add_node("node1", my_node)
wf.set_entry_point("node1")

# [关键] 导出变量
workflow = wf
```

### 3. `tools` - 工具列表

列出当前系统中已注册的所有工具。这有助于调试插件是否正确加载。

**用法：**
```bash
gecko tools [OPTIONS]
```

**参数：**
*   `-v, --verbose`: 显示工具的详细参数 Schema。

### 4. `config` - 查看配置

显示 Gecko 框架当前加载的全局配置（`gecko.config.settings`）。

**用法：**
```bash
gecko config
```

> **注意**：API Key 等敏感字段会自动脱敏显示为 `********`。

---

## 🏗️ 模块架构 (开发指南)

如果你想为 CLI 添加新命令，请遵循以下目录结构：

```text
gecko/cli/
├── __init__.py          # 异常处理与入口封装
├── main.py              # Click Group 主入口，注册子命令
├── utils.py             # UI 渲染 (Rich) 与 async_cmd 装饰器
└── commands/            # 子命令目录
    ├── __init__.py
    ├── chat.py          # chat 命令实现
    ├── run.py           # run 命令实现
    ├── config.py        # config 命令实现
    └── tools.py         # tools 命令实现
```

**添加新命令步骤：**

1.  在 `gecko/cli/commands/` 下创建新文件（例如 `deploy.py`）。
2.  使用 `@click.command()` 定义命令。
3.  如果需要异步，使用 `@async_cmd` 装饰器。
4.  在 `gecko/cli/main.py` 中导入并使用 `cli.add_command(deploy)` 注册。

### 异步命令开发示例

```python
# gecko/cli/commands/demo.py
import click
import asyncio
from gecko.cli.utils import async_cmd, print_info

@click.command()
@async_cmd  # 自动处理 asyncio.run
async def demo():
    """这是一个异步命令示例"""
    print_info("开始异步任务...")
    await asyncio.sleep(1)
    print_info("任务完成！")
```