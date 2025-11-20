# tests/plugins/test_rag_components.py
from gecko.plugins.knowledge.document import Document
from gecko.plugins.knowledge.splitters import RecursiveCharacterTextSplitter

def test_document_model():
    doc = Document(text="hello", metadata={"source": "test"})
    d = doc.to_dict()
    assert d["text"] == "hello"
    assert d["metadata"]["source"] == "test"
    assert d["id"] is not None

def test_splitter():
    splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    text = "helloworld1234567890"
    # 预期切分： "helloworld", "1234567890" (大致)
    
    doc = Document(text=text)
    chunks = splitter.split_documents([doc])
    
    assert len(chunks) >= 2
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["source_id"] == doc.id