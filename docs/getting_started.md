# 安装与快速开始

## 环境要求

*   Python 3.9+
*   (可选) Redis, ChromaDB, LanceDB 等外部服务

## 安装

Gecko 尚未发布到 PyPI，目前建议通过源码安装：

```bash
git clone https://github.com/your-repo/gecko.git
cd gecko
pip install -e .

# 安装所有可选依赖 (Redis, Vector DBs)
pip install ".[all]"
```

## 配置 API Key

Gecko 使用 `.env` 文件或环境变量管理密钥。在项目根目录创建 `.env`：

```bash
# .env
ZHIPU_API_KEY="your_api_key_here"
OPENAI_API_KEY="your_api_key_here"
# 可选配置
GECKO_LOG_LEVEL="INFO"
```

## Hello World: 你的第一个 Agent

创建一个简单的 Agent，使用智谱 AI 模型并挂载计算器工具。

```python
import asyncio
import os
from gecko.core.builder import AgentBuilder
from gecko.plugins.models import ZhipuChat
from gecko.plugins.tools.standard import CalculatorTool

async def main():
    # 1. 初始化模型
    api_key = os.getenv("ZHIPU_API_KEY")
    model = ZhipuChat(api_key=api_key, model="glm-4-flash")
    
    # 2. 构建 Agent
    agent = (AgentBuilder()
             .with_model(model)
             .with_tools([CalculatorTool()])
             .with_session_id("demo_session")
             .build())

    # 3. 运行 (自动调用计算器工具)
    response = await agent.run("计算 (123 * 45) + 99 的结果")
    print(f"🤖 Agent: {response.content}")

if __name__ == "__main__":
    asyncio.run(main())
```