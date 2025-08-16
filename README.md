# RAG Chatbot Agent

A Retrieval-Augmented Generation (RAG) chatbot with document ingestion, semantic search, conversational memory, and a REST API interface.

## Architecture
```
nlp-rag-chatbot-agent/
├── src/
│   ├── document_loader.py  # PDF/TXT loading and chunking
│   ├── embeddings.py       # Sentence-transformer embeddings
│   ├── vectorstore.py      # Cosine similarity vector search
│   ├── retriever.py        # RAG retrieval chain
│   ├── agent.py            # Conversational agent with memory
│   └── api.py              # FastAPI REST endpoints
├── config/config.yaml
├── tests/
│   ├── test_retriever.py
│   └── test_agent.py
└── main.py
```

## RAG Pipeline
```
Documents → Chunking → Embeddings → Vector Store
Query → Embedding → Retrieval → Context + History → LLM → Response
```

## Installation
```bash
git clone https://github.com/mouachiqab/nlp-rag-chatbot-agent.git
cd nlp-rag-chatbot-agent
pip install -r requirements.txt
```

## Usage
```bash
# Ingest documents and query
python main.py --docs data/ --query "What is machine learning?"

# Interactive mode
python main.py --docs data/ --interactive

# REST API
uvicorn src.api:app --reload
```

## Technologies
- Python 3.9+, LangChain, sentence-transformers, ChromaDB, FastAPI



