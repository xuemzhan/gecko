# examples/storage/vector_storage_demo.py
import asyncio
import os
import shutil
import random
from typing import List

# 导入 Gecko 组件
from gecko.plugins.storage.factory import create_storage
from gecko.core.logging import setup_logging

# 尝试导入 Zhipu (可选)
try:
    from gecko.plugins.models import ZhipuChat # Zhipu SDK 通常包含 embedding 能力
    ZHIPU_AVAILABLE = True
except ImportError:
    ZHIPU_AVAILABLE = False

setup_logging(level="INFO")

async def get_embeddings(texts: List[str], dim: int) -> List[List[float]]:
    """
    获取向量：优先使用 Zhipu API，否则使用随机向量（仅用于演示存储功能）
    """
    api_key = os.getenv("ZHIPU_API_KEY")
    if ZHIPU_AVAILABLE and api_key:
        try:
            # 注意：这里简单模拟调用 Zhipu Embedding，实际应使用 BaseEmbedder 接口
            # 为了演示 Storage 模块，这里简化处理
            import litellm
            resp = await litellm.aembedding(
                model="zhipu/embedding-2", # 假设使用智谱 Embedding
                input=texts,
                api_key=api_key
            )
            return [d["embedding"] for d in resp.data]
        except Exception as e:
            print(f"⚠️ Zhipu Embedding failed: {e}, falling back to random.")
    
    # Fallback: Random vectors
    return [[random.random() for _ in range(dim)] for _ in texts]

async def run_vector_demo(url: str, name: str):
    print(f"\n{'='*20} Running {name} Demo {'='*20}")
    
    # 1. 初始化存储
    # 工厂模式会自动识别 scheme (chroma/lancedb)
    try:
        store = await create_storage(url)
    except ImportError as e:
        print(f"❌ Skipping {name}: {e}")
        return

    try:
        # 2. 准备数据
        # 包含不同类别的文档，以及一个无 metadata 的文档
        docs_data = [
            {"id": "doc_1", "text": "The apple is a fruit.", "metadata": {"category": "fruit", "year": 2023}},
            {"id": "doc_2", "text": "Bananas are yellow.", "metadata": {"category": "fruit", "year": 2024}},
            {"id": "doc_3", "text": "Python is a programming language.", "metadata": {"category": "tech"}},
            {"id": "doc_4", "text": "Gecko is an AI framework.", "metadata": None}, # 测试 None Metadata 健壮性
        ]
        
        # 生成向量 (假设维度 1024)
        dim = 1024
        texts = [d["text"] for d in docs_data]
        embeddings = await get_embeddings(texts, dim)
        
        for i, doc in enumerate(docs_data):
            doc["embedding"] = embeddings[i]

        # 3. 写入数据 (Upsert)
        print(f"💾 Upserting {len(docs_data)} documents...")
        await store.upsert(docs_data) # type: ignore
        print("   Done.")

        # 4. 基础搜索 (无过滤)
        query_text = "Tell me about fruits"
        query_vec = (await get_embeddings([query_text], dim))[0]
        
        print(f"\n🔍 Basic Search: '{query_text}'")
        results = await store.search(query_vec, top_k=2) # type: ignore
        for res in results:
            print(f"   - [{res['score']:.4f}] {res['text']} (Meta: {res['metadata']})")

        # 5. [新特性] 带过滤搜索 (Metadata Filtering)
        print(f"\n🔍 Filtered Search (category='fruit')")
        # 即使 "Python" 可能在向量空间上偶遇（随机模式下），也会被过滤掉
        results_filtered = await store.search( # type: ignore
            query_vec, 
            top_k=5, 
            filters={"category": "fruit"}
        )
        for res in results_filtered:
            print(f"   - [{res['score']:.4f}] {res['text']} (Meta: {res['metadata']})")
            
        # 验证过滤正确性
        assert all(r['metadata'].get('category') == 'fruit' for r in results_filtered)
        print("   ✅ Filtering verified.")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        await store.shutdown()
        # 清理文件
        path = url.split("://")[1].split("?")[0]
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except:
                pass

async def main():
    # 测试 Chroma
    await run_vector_demo("chroma://./demo_chroma_db", "ChromaDB")
    
    # 测试 LanceDB (指定维度)
    await run_vector_demo("lancedb://./demo_lance_db?dim=1024", "LanceDB")

if __name__ == "__main__":
    asyncio.run(main())