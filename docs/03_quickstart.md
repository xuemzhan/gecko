# 快速开始

本教程将带你构建一个具备 **工具调用** 和 **RAG 检索能力** 的智能体。

## 1. 准备工作

确保已安装 RAG 依赖：
```bash
pip install "gecko-ai[rag]"
```

在项目根目录创建 `.env` 文件并填入 API Key：
```ini
ZHIPU_API_KEY="your_api_key_here"
GECKO_LOG_LEVEL="INFO"
```

## 2. 构建 RAG Agent

```python
import asyncio
import os
from gecko.core.builder import AgentBuilder
from gecko.plugins.models import ZhipuChat
from gecko.plugins.models.embedding import LiteLLMEmbedder, ModelConfig
from gecko.plugins.storage.factory import create_storage
from gecko.plugins.knowledge import IngestionPipeline, RetrievalTool

async def main():
    # 1. 初始化向量存储 (ChromaDB)
    # create_storage 是异步工厂，会自动处理连接和初始化
    vector_store = await create_storage("chroma://./quickstart_db")
    
    # 2. 初始化 Embedding 模型
    # 使用智谱的 Embedding 服务
    embedder = LiteLLMEmbedder(
        config=ModelConfig(
            model_name="zhipu/embedding-2", 
            api_key=os.getenv("ZHIPU_API_KEY")
        ),
        dimension=1024
    )

    # 3. 知识入库 (模拟数据)
    # 创建一个临时文件作为知识源
    with open("gecko_intro.txt", "w", encoding="utf-8") as f:
        f.write("Gecko 是一个高性能的 Python AI 框架，支持异步编排、断点恢复和全链路监控。")
    
    # 执行入库流水线：加载 -> 切分 -> 向量化 -> 存储
    pipeline = IngestionPipeline(vector_store, embedder)
    await pipeline.run(["gecko_intro.txt"])

    # 4. 创建检索工具
    # Agent 将通过此工具自主决定何时查阅知识库
    rag_tool = RetrievalTool(
        vector_store=vector_store, 
        embedder=embedder, 
        top_k=1
    )

    # 5. 构建 Agent
    llm = ZhipuChat(api_key=os.getenv("ZHIPU_API_KEY"), model="glm-4-flash")
    
    agent = (AgentBuilder()
             .with_model(llm)
             .with_tools([rag_tool])  # 挂载 RAG 工具
             .with_system_prompt("你是一个技术专家，遇到不清楚的问题请优先查阅知识库。")
             .build())

    # 6. 运行
    query = "Gecko 框架有什么核心特点？"
    print(f"User: {query}")
    
    response = await agent.run(query)
    print(f"Agent: {response.content}")

    # 清理资源
    await vector_store.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

**预期输出：**
Agent 会先调用 `knowledge_search` 工具，获取到 "Gecko 是一个高性能..." 的内容，然后基于此回答用户。