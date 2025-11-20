# gecko/plugins/knowledge/readers.py
import os
from pathlib import Path
from typing import List
from gecko.plugins.knowledge.document import Document
from gecko.plugins.knowledge.interfaces import ReaderProtocol

class TextReader(ReaderProtocol):
    """简单文本读取器 (.txt, .md, .py, etc)"""
    def load(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

class PDFReader(ReaderProtocol):
    """PDF 读取器 (依赖 pypdf)"""
    def load(self, file_path: str) -> str:
        try:
            import pypdf
        except ImportError:
            raise ImportError("请安装 pypdf 以支持 PDF 读取: pip install pypdf")
            
        text = ""
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text

class AutoReader:
    """自动分发读取器"""
    _READERS = {
        ".txt": TextReader,
        ".md": TextReader,
        ".py": TextReader,
        ".json": TextReader,
        ".pdf": PDFReader
    }

    @classmethod
    def read(cls, file_path: str) -> List[Document]:
        path = Path(file_path)
        ext = path.suffix.lower()
        
        reader_cls = cls._READERS.get(ext)
        if not reader_cls:
            raise ValueError(f"不支持的文件类型: {ext}")
        
        content = reader_cls().load(str(path))
        return [Document(text=content, metadata={"source": str(path), "filename": path.name})]