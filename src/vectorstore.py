"""Vector store for document retrieval."""
import numpy as np
from typing import List, Tuple

class SimpleVectorStore:
    def __init__(self):
        self.embeddings = []
        self.documents = []
        self.metadatas = []

    def add(self, texts: List[str], embeddings: np.ndarray, metadatas: List[dict] = None):
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            self.documents.append(text)
            self.embeddings.append(emb)
            self.metadatas.append(metadatas[i] if metadatas else {})
        print(f"Added {len(texts)} documents. Total: {len(self.documents)}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5, threshold: float = 0.0) -> List[Tuple[str, float, dict]]:
        if not self.embeddings:
            return []
        scores = []
        for emb in self.embeddings:
            score = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb) + 1e-8)
            scores.append(float(score))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked[:top_k]:
            if score >= threshold:
                results.append((self.documents[idx], score, self.metadatas[idx]))
        return results

    def size(self):
        return len(self.documents)
