# gecko/plugins/knowledge/splitters.py
from typing import List
from gecko.plugins.knowledge.document import Document

class RecursiveCharacterTextSplitter:
    """
    递归字符切分器 (参考 LangChain 逻辑)
    尝试按顺序使用分隔符切分文本，直到块大小符合要求。
    """
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        separators: List[str] | None = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """切分文档列表"""
        final_docs = []
        for doc in documents:
            chunks = self.split_text(doc.text)
            for i, chunk in enumerate(chunks):
                # 继承元数据，并增加切片信息
                new_meta = doc.metadata.copy()
                new_meta.update({"chunk_index": i, "source_id": doc.id})
                final_docs.append(Document(text=chunk, metadata=new_meta))
        return final_docs

    def split_text(self, text: str) -> List[str]:
        """切分单文本核心逻辑"""
        final_chunks = []
        if self._length(text) <= self.chunk_size:
            return [text]
            
        # 找到最优分隔符
        separator = self.separators[-1]
        for sep in self.separators:
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                break
                
        # 切分
        splits = text.split(separator) if separator else list(text)
        
        # 合并碎片
        good_splits = []
        current_chunk = ""
        
        for s in splits:
            if self._length(current_chunk) + self._length(s) < self.chunk_size:
                current_chunk += (separator if current_chunk else "") + s
            else:
                if current_chunk:
                    good_splits.append(current_chunk)
                current_chunk = s
        
        if current_chunk:
            good_splits.append(current_chunk)
            
        return good_splits

    def _length(self, text: str) -> int:
        return len(text)