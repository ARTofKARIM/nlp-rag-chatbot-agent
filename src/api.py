"""FastAPI REST API for the RAG chatbot."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="RAG Chatbot API", version="1.0.0")

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list
    query: str

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    return QueryResponse(answer="RAG response placeholder", sources=[], query=request.query)

@app.post("/ingest")
async def ingest_documents(directory: str):
    return {"status": "ok", "message": f"Documents ingested from {directory}"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
