# examples/vector_storage_demo.py
import asyncio
import os
import shutil
import random
from gecko.plugins.storage.backends.chroma import ChromaStorage
from gecko.plugins.storage.backends.lancedb import LanceDBStorage

async def run_demo(storage_cls, url, name):
    print(f"\n--- Demo: {name} ---")
    
    # 清理旧数据
    path = url.split("://")[1]
    if os.path.exists(path):
        shutil.rmtree(path)
        
    store = storage_cls(url)
    
    try:
        print(f"1. Initializing {name}...")
        await store.initialize()
        
        # 生成假数据 (100 个向量，维度 128)
        dim = 128
        count = 100
        print(f"2. Generating {count} vectors (dim={dim})...")
        
        documents = []
        for i in range(count):
            vec = [random.random() for _ in range(dim)]
            documents.append({
                "id": f"doc_{i}",
                "embedding": vec,
                "text": f"This is document number {i}",
                "metadata": {"index": i, "group": "A" if i < 50 else "B"}
            })
            
        print("3. Upserting documents (Thread Offloaded)...")
        start_t = asyncio.get_running_loop().time()
        await store.upsert(documents)
        end_t = asyncio.get_running_loop().time()
        print(f"   Done in {end_t - start_t:.4f}s")
        
        print("4. Searching...")
        query_vec = [random.random() for _ in range(dim)]
        results = await store.search(query_vec, top_k=3)
        
        for i, res in enumerate(results):
            print(f"   Rank {i+1}: {res['id']} (Score: {res['score']:.4f}) - {res['text']}")
            
    except ImportError:
        print(f"⚠️  Skipping {name}: Library not installed.")
    except Exception as e:
        print(f"❌ Error in {name}: {e}")
    finally:
        await store.shutdown()
        if os.path.exists(path):
            shutil.rmtree(path)

async def main():
    # 演示 Chroma
    try:
        import chromadb
        await run_demo(ChromaStorage, "chroma://./demo_chroma", "ChromaDB")
    except ImportError:
        print("\n[SKIP] ChromaDB not installed")

    # 演示 LanceDB
    try:
        import lancedb
        # LanceDB Demo 需要设置正确的维度，我们在 demo 类里没有动态改，
        # 但 LanceDBStorage 实现里是动态读 params 的。
        # 这里我们在代码里 hardcode 了 demo 数据的维度是 128
        # LanceDB Storage 默认读 dim=1536，我们需要通过 URL 传参覆盖
        await run_demo(LanceDBStorage, "lancedb://./demo_lance?dim=128", "LanceDB")
    except ImportError:
        print("\n[SKIP] LanceDB not installed")

if __name__ == "__main__":
    asyncio.run(main())