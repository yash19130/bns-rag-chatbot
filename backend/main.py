"""
main.py — FastAPI backend.

Endpoints:
  GET  /health        — liveness check
  POST /ingest        — trigger ingestion pipeline
  POST /query         — RAG query → answer + citations
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.retriever import retriever
from backend.generator import generate_answer
from backend.config import FAISS_INDEX_DIR

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Startup: load retriever once
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    if FAISS_INDEX_DIR.exists():
        try:
            retriever.load()
            log.info("Retriever loaded on startup.")
        except Exception as e:
            log.warning(f"Could not load index on startup: {e}. Run /ingest first.")
    else:
        log.warning("No FAISS index found. Run POST /ingest to build it.")
    yield


app = FastAPI(
    title="BNS Legal Research RAG",
    description="RAG chatbot over Bharatiya Nyaya Sanhita and Bharatiya Nagarik Suraksha Sanhita",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    citations: str
    has_answer: bool
    chunks_used: int

class IngestResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "index_loaded": retriever._loaded,
        "total_chunks": retriever.index.ntotal if retriever._loaded else 0,
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """
    Trigger the full ingestion pipeline:
    parse PDFs → clean → chunk → embed → build FAISS index.
    This takes ~2-5 minutes on first run.
    """
    try:
        from backend.ingest import run_ingestion
        run_ingestion()
        retriever.load()  # reload with fresh index
        return IngestResponse(
            status="success",
            message=f"Ingestion complete. {retriever.index.ntotal} chunks indexed."
        )
    except Exception as e:
        log.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    RAG query endpoint.
    1. Retrieve relevant chunks from FAISS
    2. Generate answer via Groq (Llama 3.1 70B)
    3. Return answer + citations
    """
    if not retriever._loaded:
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. POST to /ingest first."
        )

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks = retriever.retrieve(req.question, top_k=req.top_k)
    result = generate_answer(req.question, chunks)

    return QueryResponse(
        answer=result["answer"],
        citations=result["citations"],
        has_answer=result["has_answer"],
        chunks_used=len(chunks),
    )