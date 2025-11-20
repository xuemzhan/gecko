# gecko/plugins/knowledge/pipeline.py
from typing import List, Optional
from gecko.plugins.knowledge.interfaces import EmbedderProtocol
from gecko.plugins.storage.interfaces import VectorInterface
from gecko.plugins.knowledge.splitters import RecursiveCharacterTextSplitter
from gecko.plugins.knowledge.readers import AutoReader
from gecko.core.utils import ensure_awaitable

class IngestionPipeline:
    """
    RAG 数据入库流水线
    Load -> Split -> Embed -> Store
    """
    def __init__(
        self,
        vector_store: VectorInterface,
        embedder: EmbedderProtocol,
        splitter = None
    ):
        self.vector_store = vector_store
        self.embedder = embedder
        self.splitter = splitter or RecursiveCharacterTextSplitter()

    async def run(self, file_paths: List[str], batch_size: int = 100):
        """
        执行入库流程
        :param file_paths: 文件路径列表
        :param batch_size: 向量库写入批次大小
        """
        print(f"🚀 开始处理 {len(file_paths)} 个文件...")
        
        # 1. Load
        raw_docs = []
        for path in file_paths:
            try:
                docs = AutoReader.read(path)
                raw_docs.extend(docs)
            except Exception as e:
                print(f"⚠️ 读取失败 {path}: {e}")

        # 2. Split
        chunks = self.splitter.split_documents(raw_docs)
        print(f"✂️ 切分为 {len(chunks)} 个片段")

        # 3. Embed & Store (Batch Processing)
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [doc.text for doc in batch]
            
            # 生成向量
            embeddings = await ensure_awaitable(self.embedder.embed_documents, texts)
            
            # 注入向量到文档对象
            docs_to_upsert = []
            for doc, emb in zip(batch, embeddings):
                doc.embedding = emb
                docs_to_upsert.append(doc.to_dict())
            
            # 写入数据库
            await self.vector_store.upsert(docs_to_upsert)
            print(f"💾 已存储批次 {i} - {i+len(batch)}")
            
        print("✅ 入库完成")