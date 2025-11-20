# gecko/plugins/knowledge/__init__.py
from gecko.plugins.knowledge.document import Document
from gecko.plugins.knowledge.interfaces import EmbedderProtocol
from gecko.plugins.knowledge.embedders import OpenAIEmbedder, OllamaEmbedder
from gecko.plugins.knowledge.splitters import RecursiveCharacterTextSplitter
from gecko.plugins.knowledge.pipeline import IngestionPipeline
from gecko.plugins.knowledge.tool import RetrievalTool

__all__ = [
    "Document", 
    "EmbedderProtocol", 
    "OpenAIEmbedder", 
    "OllamaEmbedder",
    "RecursiveCharacterTextSplitter",
    "IngestionPipeline",
    "RetrievalTool"
]