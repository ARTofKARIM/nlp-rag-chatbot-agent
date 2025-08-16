"""RAG retrieval chain."""
from typing import List, Tuple

class RAGRetriever:
    def __init__(self, vectorstore, embedding_engine, top_k=5, threshold=0.7):
        self.vectorstore = vectorstore
        self.embedding_engine = embedding_engine
        self.top_k = top_k
        self.threshold = threshold

    def retrieve(self, query: str) -> List[Tuple[str, float]]:
        query_emb = self.embedding_engine.embed_query(query)
        results = self.vectorstore.search(query_emb, self.top_k, self.threshold)
        return [(text, score) for text, score, _ in results]

    def build_context(self, query: str) -> str:
        results = self.retrieve(query)
        if not results:
            return "No relevant context found."
        context_parts = []
        for i, (text, score) in enumerate(results):
            context_parts.append(f"[Source {i+1}, relevance: {score:.2f}]\n{text}")
        return "\n\n".join(context_parts)
