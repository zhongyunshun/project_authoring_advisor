# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agency-specific RAG (Retrieval-Augmented Generation) system for project authoring, validated with the Toronto and Region Conservation Authority (TRCA). Helps engineers draft project scopes, design notes, and retrieve past project information from technical documents.

Built on **LlamaIndex + Qdrant** (migrated from LangChain + FAISS).

## Common Commands

### Setup
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Application
```bash
# Interactive chat (default)
python main.py --mode chat --model openai --embedding openai

# Agentic mode with ReAct reasoning + web search
python main.py --mode agent --model openai --web_search

# Batch CSV processing
python main.py --mode csv --input_csv_file questions.csv --output_csv_path output/

# Streamlit web UI
streamlit run app.py

# Force re-index PDFs
python main.py --mode chat --reindex --pdf_dir data
```

### Evaluation
```bash
# Prompt engineering evaluation (all stages)
python prompt_engineer_ragas/prompt_eval.py --stages all

# Individual stages
python prompt_engineer_ragas/prompt_eval.py --stages generate
python prompt_engineer_ragas/prompt_eval.py --stages custom_eval
python prompt_engineer_ragas/prompt_eval.py --stages ragas_eval
python prompt_engineer_ragas/prompt_eval.py --stages summarize

# Specific patterns only
python prompt_engineer_ragas/prompt_eval.py --patterns rag-only persona+cot+format --stages generate

# BLEU/ROUGE evaluation
python evaluation/folder_eval.py
```

### Voice Agent (Docker Compose)
```bash
docker compose -f docker-compose.voice.yml up
```

### No formal test suite or linting tools are configured.

## Architecture

### Layer Structure
```
Interfaces (CLI main.py / Streamlit ui/ / Voice voice_agent/)
    ↓
Orchestration (pipeline/rag_engine.py: RAGEngine, PromptingRAGEngine | agents/rag_agent.py: RAGAgent)
    ↓
Core Abstractions (core/llm_factory.py | core/embedding_factory.py | core/vector_store.py)
    ↓
Data Pipeline (ingest/indexer.py → ingest/pdf_loader.py → ingest/chunker.py → Qdrant)
```

### Key Design Patterns

- **Factory pattern**: `LLMFactory.create(provider=)` supports `openai`, `gemini`, `claude`, `llama_cpp`. `EmbeddingFactory.create(provider=)` supports `openai`, `huggingface`.
- **Unified response**: All engines return `RAGResponse(answer, sources)`.
- **Three RAG strategies**: `RAGEngine` (standard conversational/stateless), `PromptingRAGEngine` (composable prompt patterns: persona+CoT+format), `RAGAgent` (ReAct with tool selection).
- **Adapter pattern in voice agent**: `BackendAdapter` interface with `HTTPBackendAdapter` (calls FastAPI) and `LocalBackendAdapter` (in-process RAGEngine).

### Critical Paths

- **Config/API keys**: `config/settings.py` → `Settings` dataclass loaded via `Settings.from_env()`. Keys come from `.env` or `config/keys.py`.
- **Vector storage**: Qdrant file-based at `./vector_db/qdrant_storage`. Collections must use matching embedding dimensions.
- **Document ingestion**: `Indexer` orchestrates PDF loading → chunking (SentenceSplitter) → embedding → Qdrant upsert. Supports incremental indexing and on-the-fly upload via Streamlit.
- **Prompt patterns**: Defined in `prompt_engineer_ragas/templates/`. Selectable via `--pattern` flag (persona+cot+format, cot+format, etc.).

### Multi-Provider LLM Support

Default models per provider (configured in `config/settings.py`):
- OpenAI: gpt-4o-mini
- Gemini: models/gemini-2.0-flash
- Claude: claude-sonnet-4-20250514
- Local: LlamaCPP with GGUF files (requires `--model_path`)

Default embeddings:
- OpenAI: text-embedding-3-small
- HuggingFace: all-MiniLM-L6-v2 (local, no API key needed)

## Important Constraints

- Embeddings used for querying **must match** those used for indexing (same model, same dimensions).
- The `data/` directory contains TRCA project PDFs — these are the source documents for the RAG system.
- Voice agent has its own `requirements.txt` in `voice_agent/`.
- The README references the old LangChain/FAISS architecture in some places; the actual codebase now uses LlamaIndex/Qdrant.
