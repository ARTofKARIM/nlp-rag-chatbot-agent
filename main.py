"""Main entry point for RAG chatbot."""
import argparse
import yaml
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingEngine
from src.vectorstore import SimpleVectorStore
from src.retriever import RAGRetriever
from src.agent import RAGAgent

def main():
    parser = argparse.ArgumentParser(description="RAG Chatbot Agent")
    parser.add_argument("--docs", help="Documents directory")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--query", help="Single query mode")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    emb_engine = EmbeddingEngine(config["embeddings"]["model"])
    emb_engine.load_model()
    store = SimpleVectorStore()

    if args.docs:
        loader = DocumentLoader(config["embeddings"]["chunk_size"], config["embeddings"]["chunk_overlap"])
        docs = loader.load_directory(args.docs)
        chunks = loader.chunk_documents(docs)
        texts = [c.content for c in chunks]
        metas = [c.metadata for c in chunks]
        embeddings = emb_engine.embed(texts)
        store.add(texts, embeddings, metas)

    retriever = RAGRetriever(store, emb_engine, config["retrieval"]["top_k"], config["retrieval"]["score_threshold"])
    agent = RAGAgent(retriever, config["agent"]["system_prompt"], config["agent"]["memory_k"])

    if args.query:
        print(agent.chat(args.query))
    elif args.interactive:
        print("RAG Chatbot (type 'quit' to exit)")
        while True:
            query = input("\nYou: ").strip()
            if query.lower() in ["quit", "exit"]:
                break
            print(f"Bot: {agent.chat(query)}")

if __name__ == "__main__":
    main()
