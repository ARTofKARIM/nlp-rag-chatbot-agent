"""Tests for RAG retrieval."""
import unittest
import numpy as np
from src.vectorstore import SimpleVectorStore
from src.embeddings import EmbeddingEngine
from src.retriever import RAGRetriever

class TestVectorStore(unittest.TestCase):
    def test_add_and_search(self):
        store = SimpleVectorStore()
        texts = ["hello world", "machine learning", "deep learning"]
        embs = np.random.randn(3, 384).astype(np.float32)
        store.add(texts, embs)
        self.assertEqual(store.size(), 3)
        results = store.search(embs[0], top_k=2)
        self.assertGreater(len(results), 0)

    def test_empty_store(self):
        store = SimpleVectorStore()
        results = store.search(np.random.randn(384), top_k=5)
        self.assertEqual(len(results), 0)

class TestRetriever(unittest.TestCase):
    def test_build_context(self):
        store = SimpleVectorStore()
        texts = ["Python is great", "ML is fun"]
        embs = np.random.randn(2, 384).astype(np.float32)
        store.add(texts, embs)
        engine = EmbeddingEngine()
        engine.dimension = 384
        retriever = RAGRetriever(store, engine, top_k=2, threshold=0.0)
        context = retriever.build_context("test query")
        self.assertIsInstance(context, str)

if __name__ == "__main__":
    unittest.main()
