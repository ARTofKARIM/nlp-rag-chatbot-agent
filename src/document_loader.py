"""Document loading and chunking for RAG pipeline."""
import os
from typing import List

class Document:
    def __init__(self, content: str, metadata: dict = None):
        self.content = content
        self.metadata = metadata or {}

class DocumentLoader:
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_text(self, filepath: str) -> List[Document]:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        return [Document(content=text, metadata={"source": filepath, "type": "text"})]

    def load_pdf(self, filepath: str) -> List[Document]:
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(filepath)
            docs = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text.strip():
                    docs.append(Document(content=text, metadata={"source": filepath, "page": i + 1}))
            return docs
        except ImportError:
            print("PyPDF2 not installed")
            return []

    def load_directory(self, dir_path: str) -> List[Document]:
        docs = []
        for fname in os.listdir(dir_path):
            fpath = os.path.join(dir_path, fname)
            if fname.endswith(".txt"):
                docs.extend(self.load_text(fpath))
            elif fname.endswith(".pdf"):
                docs.extend(self.load_pdf(fpath))
        print(f"Loaded {len(docs)} documents from {dir_path}")
        return docs

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        chunks = []
        for doc in documents:
            text = doc.content
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk_text = text[i:i + self.chunk_size]
                if chunk_text.strip():
                    metadata = {**doc.metadata, "chunk_start": i}
                    chunks.append(Document(content=chunk_text, metadata=metadata))
        print(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
