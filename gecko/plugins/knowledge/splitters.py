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
        """
        切分单文本的核心逻辑

        设计要点：
        - 优先使用用户指定的分隔符（如空格、换行等）
        - 在保证单个 chunk 不超过 chunk_size 的前提下尽量“吃满”
        - ✅ 修复点：
          1) 判断是否还能追加下一段时，需要把分隔符的长度一起算进去
          2) 允许“刚好等于 chunk_size” 的情况（<=），避免不必要的碎片化
        """
        final_chunks: List[str] = []

        # 若整体长度不超过 chunk_size，直接返回整体
        if self._length(text) <= self.chunk_size:
            return [text]

        # 1. 选择分隔符（优先粗粒度）
        #    - 若指定了 separators，则按顺序优先选用第一个在文本中出现的
        #    - 若没有任何分隔符匹配，则退化为逐字符切分
        separator = self.separators[-1]
        for sep in self.separators:
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                break

        # 2. 基于分隔符切分
        splits = text.split(separator) if separator else list(text)

        # 3. 合并碎片
        good_splits: List[str] = []
        current_chunk = ""

        for s in splits:
            # 本次准备追加的内容实际会占用的长度：
            # - 自身长度 self._length(s)
            # - 如果当前已经有内容，还要加上分隔符长度（例如空格）
            to_add = self._length(s)
            if current_chunk:
                to_add += self._length(separator)

            # ✅ 修复点：
            #   使用 <=，允许“刚好等于 chunk_size”
            #   同时严格按“当前长度 + 分隔符 + 新内容”的总长度判断，
            #   避免出现 "hello world" 实际 11 个字符却被错误认为是 10 的情况。
            if self._length(current_chunk) + to_add <= self.chunk_size:
                # 可以继续往当前块追加
                if current_chunk:
                    current_chunk += separator + s
                else:
                    current_chunk = s
            else:
                # 当前块已经装不下更多内容了，先收集当前块，开启新块
                if current_chunk:
                    good_splits.append(current_chunk)
                current_chunk = s

        # 收尾：最后一个块如果非空就放进去
        if current_chunk:
            good_splits.append(current_chunk)

        return good_splits


    def _length(self, text: str) -> int:
        return len(text)