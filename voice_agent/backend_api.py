"""Thin FastAPI wrapper — exposes the current RAG system as a REST API.

This is the bridge between the voice agent (HTTP adapter) and the existing
RAG pipeline. Run it alongside the voice agent:

    uvicorn voice_agent.backend_api:app --host 0.0.0.0 --port 8000

The voice agent's HTTPBackendAdapter will POST to /query.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

# These will be set during lifespan
_engine = None


class QueryRequest(BaseModel):
    query: str
    conversation_id: str = ""
    user_id: str = "default"
    image_base64: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: list = []
    metadata: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the RAG engine on startup."""
    global _engine

    from config.settings import Settings
    from core.llm_factory import LLMFactory
    from core.embedding_factory import EmbeddingFactory
    from core.vector_store import VectorStoreManager
    from ingest.indexer import Indexer
    from pipeline.rag_engine import RAGEngine

    settings = Settings.from_env()
    settings.apply_env()

    llm = LLMFactory.create(
        provider=os.getenv("RAG_LLM_PROVIDER", "openai"),
        model=os.getenv("RAG_LLM_MODEL", ""),
        temperature=0.7,
        max_tokens=int(os.getenv("RAG_MAX_TOKENS", "1024")),
    )
    embed_model = EmbeddingFactory.create(
        provider=os.getenv("RAG_EMBEDDING_PROVIDER", "openai")
    )

    vsm = VectorStoreManager(storage_path="./vector_db/qdrant_storage")
    indexer = Indexer(vsm, embed_model, chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "700")))

    collection = os.getenv("RAG_COLLECTION", "trca_documents")
    if not vsm.collection_exists(collection):
        pdf_dir = os.getenv("RAG_PDF_DIR", "data")
        index = indexer.index_directory(pdf_dir, collection)
    else:
        index = indexer.load_index(collection)

    _engine = RAGEngine(
        index=index,
        llm=llm,
        system_prompt=settings.system_prompt_chat,
        top_k=int(os.getenv("RAG_TOP_K", "22")),
        conversational=True,
    )

    yield

    _engine = None


app = FastAPI(title="RAG Backend API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "engine_loaded": _engine is not None}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    import asyncio

    if _engine is None:
        return QueryResponse(answer="Backend not ready — engine not initialized.")

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _engine.query, req.query)

    sources = [
        {"text": s.text[:500], "metadata": s.metadata, "score": s.score}
        for s in result.sources
    ]
    return QueryResponse(answer=result.answer, sources=sources)
