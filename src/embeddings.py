"""Embedding generation for document chunks."""
import numpy as np
from typing import List

class EmbeddingEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.dimension = None

    def load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.dimension = self.model.get_sentence_embedding_dimension()
            print(f"Embedding model loaded: {self.model_name} (dim={self.dimension})")
        except ImportError:
            print("sentence-transformers not installed, using random embeddings")
            self.dimension = 384

    def embed(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            return np.random.randn(len(texts), self.dimension or 384).astype(np.float32)
        return self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed([query])[0]
