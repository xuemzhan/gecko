# examples/knowledge_rag_demo.py
"""
RAG 知识库演示

验证功能：
1. 文档入库 (IngestionPipeline)
2. 向量检索 (RetrievalTool) -> 验证 Bug #9 修复 (PrivateAttr)
3. 结合 Agent 进行问答
"""
import asyncio
import os
import shutil
from gecko.core.agent import Agent
from gecko.core.builder import AgentBuilder
from gecko.core.message import Message
from gecko.plugins.storage.factory import create_storage
from gecko.plugins.knowledge import IngestionPipeline, Document
from gecko.plugins.knowledge.tool import RetrievalTool
from gecko.plugins.models import ZhipuChat
# 假设我们使用 Zhipu 的 Embedding (通过 LiteLLMEmbedder)
from gecko.plugins.models.presets.zhipu import ZhipuChat # Zhipu SDK 暂无单独 Embedder Preset，需手动配置
from gecko.plugins.models.embedding import LiteLLMEmbedder
from gecko.plugins.models.config import ModelConfig

async def main():
    api_key = os.getenv("ZHIPU_API_KEY")
    if not api_key:
        print("Please set ZHIPU_API_KEY")
        return

    # 1. 准备向量存储 (Chroma)
    persist_path = "./demo_rag_db"
    if os.path.exists(persist_path): shutil.rmtree(persist_path)
    
    vector_store = await create_storage(f"chroma://{persist_path}")
    
    # 2. 准备 Embedding 模型
    embedder = LiteLLMEmbedder(
        config=ModelConfig(model_name="zhipu/embedding-2", api_key=api_key),
        dimension=1024
    )

    # 3. 数据入库
    print("🚀 Ingesting documents...")
    pipeline = IngestionPipeline(vector_store, embedder)
    
    # 创建临时文件用于测试
    with open("gecko_intro.txt", "w") as f:
        f.write("Gecko 是一个高性能的 AI Agent 框架，支持异步编排和插件化设计。它由 Python 编写。")
    
    await pipeline.run(["gecko_intro.txt"])

    # 4. 初始化检索工具
    # [Verification] Bug #9: 如果 RetrievalTool 没有使用 PrivateAttr，这里会抛出 Pydantic ValidationError
    print("\n🔧 Initializing RetrievalTool...")
    rag_tool = RetrievalTool(
        vector_store=vector_store,
        embedder=embedder,
        top_k=1
    )
    print("✅ Tool initialized successfully (PrivateAttr fix works).")

    # 5. 构建 Agent
    model = ZhipuChat(api_key=api_key, model="glm-4-flash")
    agent = AgentBuilder()\
        .with_model(model)\
        .with_tools([rag_tool])\
        .build()

    # 6. 提问
    query = "Gecko 框架是用什么语言编写的？"
    print(f"\n👤 User: {query}")
    response = await agent.run(query)
    print(f"🤖 Agent: {response.content}") # type: ignore

    # 清理
    await vector_store.shutdown()
    if os.path.exists("gecko_intro.txt"): os.remove("gecko_intro.txt")
    if os.path.exists(persist_path): shutil.rmtree(persist_path)

if __name__ == "__main__":
    asyncio.run(main())