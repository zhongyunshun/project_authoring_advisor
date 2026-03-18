# Agency-Specific Project Authoring Advisor (TRCA-LLM)

> Retrieval-Augmented Generation (RAG) system for agency-specific project authoring, validated with the Toronto and Region Conservation Authority (TRCA).

## Overview

This repository implements an **agency-specific project authoring advisor**: a deployable RAG system that helps engineers draft **project scopes**, **design notes**, and retrieve **past project information** directly from an agency's technical documents (reports, workplans, design briefs, geomorphic assessments).

The system combines:
- **Semantic retrieval** over an agency-curated Technical Document Database (TDD) using LlamaIndex + Qdrant
- **Domain-aware prompting** (persona, structured format, chain-of-thought, few-shot)
- **Multi-provider LLM support** (OpenAI, Gemini, Claude, LlamaCPP)
- **Agentic mode** with ReAct reasoning and web search tool selection
- **LLM-assisted, adversarial prompt evaluation** (AMLLM-Auto-EVAL)
- A **chat-style Streamlit GUI** with on-the-fly document ingestion
- A **voice agent** for real-time speech-to-text, text-to-speech, and voice-powered RAG (see `voice_agent/README.md`)

> **Practical outcome:** Faster, more consistent authoring grounded in institutional knowledge, with transparent citations to source passages.

---

## System Architecture

```
Interfaces
├── CLI (main.py)              — chat, CSV batch, agentic modes
├── Streamlit (app.py)         — web UI with chat, PDF upload, source display
└── Voice (voice_agent/)       — mic transcription → RAG → spoken answer

        ↓

Orchestration
├── RAGEngine                  — conversational or stateless RAG
├── PromptingRAGEngine         — composable prompt patterns (persona+CoT+format)
└── RAGAgent                   — ReAct agent with tool selection

        ↓

Core Abstractions
├── LLMFactory                 — OpenAI, Gemini, Claude, LlamaCPP
├── EmbeddingFactory           — OpenAI, HuggingFace
└── VectorStoreManager         — Qdrant (file-based, no server)

        ↓

Data Pipeline
└── Indexer → PDFLoader → DocumentChunker → Qdrant
```

### Data Flow

**Ingestion:**
```
PDFs (data/)
  → PDFLoader.load_directory()        [PyMuPDF reader]
  → DocumentChunker.chunk()           [SentenceSplitter, 700 chars, 50 overlap]
  → EmbeddingFactory.create()         [OpenAI or HuggingFace embeddings]
  → VectorStoreManager.create_index() [Qdrant file storage]
```

**Query:**
```
User Question
  → Retriever.retrieve()              [top_k=22 semantic search in Qdrant]
  → LLM generation                    [with retrieved context + system prompt]
  → RAGResponse(answer, sources)      [answer text + source chunks with metadata]
```

**Agentic Query:**
```
User Question
  → ReActAgent reasoning              [should I search documents? search the web?]
  → Tool selection & execution         [document_query_tool | web_search_tool]
  → RAGResponse(answer, sources)
```

---

## Repository Structure

```
project_authoring_advisor/
├── config/                        # Configuration & API key management
│   ├── settings.py                # Settings dataclass, model registry, env loading
│   └── keys.py                    # API keys (not committed)
├── core/                          # Factory patterns for providers
│   ├── llm_factory.py             # LLMFactory — multi-provider LLM creation
│   ├── embedding_factory.py       # EmbeddingFactory — OpenAI or HuggingFace
│   └── vector_store.py            # VectorStoreManager — Qdrant wrapper
├── ingest/                        # Document ingestion pipeline
│   ├── pdf_loader.py              # PDFLoader — PyMuPDF-based PDF reading
│   ├── chunker.py                 # DocumentChunker — sentence-aware splitting
│   └── indexer.py                 # Indexer — orchestrates load → chunk → embed → store
├── pipeline/                      # RAG engine implementations
│   ├── base.py                    # SourceNode, RAGResponse, BaseRAGPipeline
│   ├── rag_engine.py              # RAGEngine, PromptingRAGEngine
│   └── prompt_templates.py        # Prompt patterns & build_prompt()
├── agents/                        # Agentic RAG with ReAct
│   ├── rag_agent.py               # RAGAgent — ReAct agent with tools
│   └── tools.py                   # Tool factories (document query, web search)
├── ui/                            # Streamlit web interface
│   ├── streamlit_app.py           # Main chat UI (sidebar, engine init, chat)
│   └── pages/
│       └── upload_pdf.py          # PDF upload & on-the-fly indexing page
├── streamlit_class/               # Streamlit state management
│   └── conversations.py           # Conversation class (session_id, title, history)
├── evaluation/                    # Evaluation scripts (organised by dataset)
│   └── generation_quality/
│       ├── 200_factual_qa_eval/
│       │   └── factual_qa_eval.py # Combined BLEU/ROUGE + RAGAS eval over a folder of CSVs
│       └── thrity_open_ended_questions_eval/
│           └── prompt_eval.py     # Prompt engineering evaluation pipeline (generate, judge, RAGAS, summarize)
├── prompt_engineer_ragas/         # Prompt engineering artefacts
│   ├── thrity_open_ended_questions.csv  # 30 open-ended test questions (3 projects × 10)
│   ├── templates/                 # Prompt pattern building blocks
│   ├── prompts/                   # Generated prompt logs per pattern
│   ├── prompting_results/         # Generated answers per pattern
│   └── data/                      # Cached evaluation CSVs (RAGAS input/output)
├── data/                          # Source TRCA PDFs
│   ├── German Mills/              # German Mills Settlers Park documents
│   ├── Humber Bay Park East/      # Humber Bay Park East Shoreline documents
│   └── Peacham Cr/                # Peacham Crescent documents
├── vector_db/                     # Qdrant vector storage (file-based)
│   └── qdrant_storage/
├── voice_agent/                   # Voice STT/TTS/RAG (see voice_agent/README.md)
├── scripts/                       # Migration & utility scripts
│   └── migrate_faiss_to_qdrant.py # One-time FAISS → Qdrant migration
├── preprocessing/                 # Text processing utilities (compute_complexity.py)
├── QA_pair/                       # QA logs and evaluation data
├── utils/                         # Misc utilities
├── main.py                        # CLI entry point
├── app.py                         # Streamlit entry point
├── requirements.txt               # Python dependencies
└── CLAUDE.md                      # Development guidance
```

---

## Module Reference

### `config/settings.py` — Configuration

Central configuration loaded from `.env` files and environment variables.

| Class / Method | Description |
|----------------|-------------|
| `Settings` | Frozen dataclass holding all configuration. Fields: `openai_api_key`, `google_api_key`, `anthropic_api_key`, `tavily_api_key` (API keys); `llm_provider`, `llm_model`, `llm_temperature`, `llm_max_tokens` (LLM config); `embedding_provider`, `embedding_model` (embeddings); `qdrant_path`, `collection_name`, `top_k`, `chunk_size`, `chunk_overlap` (retrieval); `system_prompt_chat`, `system_prompt_csv` (system prompts); `AVAILABLE_MODELS` (dict of model names per provider). |
| `Settings.from_env()` | Class method that populates a `Settings` instance from environment variables and `.env` file. |
| `Settings.apply_env()` | Pushes API keys from the Settings object into `os.environ` so downstream libraries (OpenAI, Google) can find them. |

### `core/llm_factory.py` — LLM Provider Factory

Creates LlamaIndex LLM instances from a provider name.

| Class / Method | Description |
|----------------|-------------|
| `LLMFactory.create(provider, model, temperature, max_tokens, model_path, n_ctx) -> LLM` | Static factory method. Supported providers and defaults: `"openai"` → `gpt-4o-mini`, `"gemini"` → `models/gemini-2.5-flash`, `"claude"` → `claude-sonnet-4-6`, `"llama_cpp"` / `"llama"` / `"qwen"` → local GGUF model via LlamaCPP. |
| `LLMFactory._build_llama_cpp(model_path, target_ctx, max_tokens, temperature)` | Internal method for LlamaCPP with context-window fallback negotiation. Tries `target_ctx`, then falls back to `8192 → 4096 → 2048` if VRAM is insufficient. Uses `n_gpu_layers=-1` for full GPU offload. |

### `core/embedding_factory.py` — Embedding Provider Factory

Creates LlamaIndex embedding models.

| Class / Method | Description |
|----------------|-------------|
| `EmbeddingFactory.create(provider, model_name) -> BaseEmbedding` | Static factory method. `"openai"` → `OpenAIEmbedding(model_name="text-embedding-3-small")` (1536 dims). `"huggingface"` / `"sbert"` / `"sentencebert"` / `"ds"` → `HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")` (384 dims, local, no API key). |

**Important:** Embeddings used for querying must match those used during indexing (same model, same dimensions). Mismatched dimensions cause errors.

### `core/vector_store.py` — Vector Store Manager

Wraps Qdrant client for local file-based vector storage.

| Class / Method | Description |
|----------------|-------------|
| `VectorStoreManager(storage_path)` | Constructor. Creates a Qdrant client pointing to `storage_path` (default `./vector_db/qdrant_storage`). No server required. |
| `.get_vector_store(collection_name) -> QdrantVectorStore` | Returns a LlamaIndex-compatible vector store for a given collection. |
| `.get_index(collection_name, embed_model) -> VectorStoreIndex` | Loads an existing Qdrant collection as a queryable index. |
| `.create_index(collection_name, nodes, embed_model) -> VectorStoreIndex` | Creates a new collection from a list of `TextNode` objects. |
| `.add_nodes(collection_name, nodes, embed_model) -> VectorStoreIndex` | Incrementally upserts nodes into an existing collection. Used for on-the-fly PDF uploads. |
| `.collection_exists(collection_name) -> bool` | Checks if a collection already exists in Qdrant. |

### `ingest/pdf_loader.py` — PDF Loading

| Class / Method | Description |
|----------------|-------------|
| `PDFLoader.load_directory(path) -> List[Document]` | Static method. Recursively loads all PDFs from a directory using PyMuPDFReader. Adds `"source"` metadata (filename) to each document. Skips problematic PDFs gracefully. |
| `PDFLoader.load_uploaded_file(uploaded_file) -> List[Document]` | Static method. Handles Streamlit `UploadedFile` objects for on-the-fly ingestion. |

### `ingest/chunker.py` — Document Chunking

| Class / Method | Description |
|----------------|-------------|
| `DocumentChunker(chunk_size=700, chunk_overlap=50)` | Constructor. Uses LlamaIndex's `SentenceSplitter` for sentence-aware chunking. |
| `.chunk(documents) -> List[TextNode]` | Splits documents into text nodes. Preserves document metadata (source filename, page number) in each chunk. Default: 700 characters per chunk with 50-character overlap. |

### `ingest/indexer.py` — Ingestion Orchestrator

| Class / Method | Description |
|----------------|-------------|
| `Indexer(vector_store_manager, embed_model, chunk_size, chunk_overlap)` | Constructor. Composes PDFLoader, DocumentChunker, and VectorStoreManager. |
| `.index_directory(pdf_dir, collection_name) -> VectorStoreIndex` | Full pipeline: load all PDFs → chunk → embed → create Qdrant collection. |
| `.index_uploaded_file(uploaded_file, collection_name) -> VectorStoreIndex` | Incremental pipeline: load single PDF → chunk → embed → upsert into existing collection. |
| `.load_index(collection_name) -> VectorStoreIndex` | Loads an existing Qdrant collection as a queryable index. |

### `pipeline/base.py` — Base Abstractions

| Class | Description |
|-------|-------------|
| `SourceNode` | Dataclass representing a retrieved chunk: `text` (str), `metadata` (dict — source filename, page number), `score` (float — similarity). |
| `RAGResponse` | Dataclass returned by all engines: `answer` (str), `sources` (List[SourceNode]). Property `context_text` joins all source texts. |
| `BaseRAGPipeline` | Abstract base class defining the interface: `query(question) -> RAGResponse` and `reset()`. |

### `pipeline/rag_engine.py` — RAG Engines

| Class / Method | Description |
|----------------|-------------|
| `RAGEngine(index, llm, system_prompt, top_k, conversational, memory_token_limit)` | Main RAG engine. If `conversational=True`: uses LlamaIndex's `CondensePlusContextChatEngine` with `ChatMemoryBuffer` — maintains multi-turn conversation history, condenses prior turns into standalone queries. If `conversational=False`: uses `RetrieverQueryEngine` — stateless, single-turn, `response_mode="compact"`. |
| `RAGEngine.query(question) -> RAGResponse` | Retrieves top-k chunks from Qdrant, passes them with the question to the LLM, returns answer with source attribution. |
| `RAGEngine.reset()` | Clears conversation memory (conversational mode only). |
| `PromptingRAGEngine(index, llm, top_k, pattern)` | Stateless engine for prompt engineering research. Manually retrieves chunks, builds a structured prompt using `build_prompt()` with the selected pattern, and sends it to the LLM via `llm.complete()`. |
| `PromptingRAGEngine.query(question) -> RAGResponse` | Retrieves nodes, formats a prompt with the configured pattern, calls the LLM, returns the response. |

### `pipeline/prompt_templates.py` — Prompt Patterns

Building blocks that can be composed into named patterns:

| Component | Purpose |
|-----------|---------|
| `PERSONA` | Sets the LLM's role as a TRCA-specialized Q&A assistant |
| `COT` | Chain-of-thought: check TRCA docs → web search → LLM knowledge → cite |
| `FORMAT_TEMPLATE` | Professional tone, ask clarifications, cite sources |
| `FEW_SHOT` | Example Q&A pair (Humber Bay Park East) |

| Pattern Name | Components |
|-------------|------------|
| `persona+cot+format` | PERSONA + domain info + COT + FORMAT + user input + FEW_SHOT |
| `cot+format` | domain info + COT + FORMAT + user input + FEW_SHOT |
| `persona+format` | PERSONA + domain info + FORMAT + user input + FEW_SHOT |
| `persona+cot` | PERSONA + domain info + COT + user input + FEW_SHOT |
| `rag-only` | domain info + user input (minimal, no prompt engineering) |
| `gpt-4o-mini` | user input only (zero-shot baseline, no context) |

| Function | Description |
|----------|-------------|
| `build_prompt(pattern, query, retrieval_log) -> str` | Constructs a full prompt string by inserting retrieved context and the user's query into the selected pattern template. |

### `agents/rag_agent.py` — Agentic RAG

| Class / Method | Description |
|----------------|-------------|
| `RAGAgent(tools, llm, system_prompt, conversational, memory_token_limit, verbose)` | ReAct agent that autonomously selects tools based on the question. Uses LlamaIndex's `ReActAgent` with optional `ChatMemoryBuffer` for multi-turn conversations. |
| `RAGAgent.query(question) -> RAGResponse` | The agent reasons step-by-step: decides whether to search documents, search the web, or answer from knowledge, executes the selected tools, and synthesizes a final response. |
| `RAGAgent.reset()` | Resets conversation memory and agent state. |

### `agents/tools.py` — Tool Factories

| Function | Description |
|----------|-------------|
| `create_document_query_tool(index, llm, top_k) -> QueryEngineTool` | Wraps a Qdrant index as a LlamaIndex `QueryEngineTool` that the agent can invoke to search project documents. |
| `create_web_search_tool(provider, max_results) -> FunctionTool` | Creates a web search tool using Tavily or Serper API. The agent uses this when document context is insufficient. |

### `ui/streamlit_app.py` — Streamlit Chat UI

| Function | Description |
|----------|-------------|
| `init_sidebar()` | Renders the sidebar: OpenAI API key input, LLM provider dropdown (`openai`/`gemini`/`claude`), model selection dropdown (dynamically populated from `Settings.AVAILABLE_MODELS`), agentic mode checkbox, chat history list with conversation switching, "New Conversation" button. Changing the provider or model forces engine rebuild. |
| `init_engine()` | Lazy-initializes the RAG engine. Loads settings from `.env`, creates an LLM via `LLMFactory`, loads the HuggingFace embedding model, and builds either a `RAGEngine` (default) or `RAGAgent` (if agentic mode is enabled). Caches in `st.session_state.rag_engine`. |
| `render_chat()` | Displays the multi-turn chat interface. Shows conversation history with role-based message bubbles. Each assistant message has an expandable "Context Used" section showing the top 5 retrieved source chunks with metadata (document name, page number, similarity score). Handles user input, calls `engine.query()`, and appends results to conversation history. |
| `main()` | Entry point. Initializes session state (conversations list, current conversation), calls `init_sidebar()`, `init_engine()`, and `render_chat()`. |

### `ui/pages/upload_pdf.py` — PDF Upload Page

| Function | Description |
|----------|-------------|
| `upload_files_form()` | Renders a file uploader accepting multiple PDFs. For each uploaded file, calls `Indexer.index_uploaded_file()` to incrementally add it to the `pdf_uploads` Qdrant collection. After upload, rebuilds the RAG engine to include the new documents. Tracks uploaded filenames in session state to avoid re-indexing. |

### `streamlit_class/conversations.py` — Conversation State

| Class | Description |
|-------|-------------|
| `Conversation(session_id, title, chat_history)` | Container for a single chat session. `session_id` (int) — unique identifier. `title` (str) — editable display name. `chat_history` (list of dicts) — messages with `role` (`"user"` or `"assistant"`), `content` (str), and optional `sources` (list of source dicts for assistant messages). |

### `evaluation/generation_quality/200_factual_qa_eval/factual_qa_eval.py`

Batch evaluation of RAG CSV outputs using BLEU/ROUGE and RAGAS. RAGAS uses LlamaIndex-wrapped `gpt-4o-mini` and `text-embedding-3-small` — no LangChain required.

| Function | Description |
|----------|-------------|
| `calculate_bleu(csv_file, n) -> (float, list)` | BLEU-n score between `answer` and `generated_answer`. Supports `alternative_answer` as a second reference. |
| `calculate_rouge(csv_file) -> dict` | ROUGE-1/2/3 F-measure. Takes best score across primary and alternative references. |
| `evaluate_folder_bleu_rouge(input_folder, output_csv)` | Iterates all CSVs in a folder, computes BLEU-1/2/3 and ROUGE-1/2/3, prints a summary table, saves results CSV. |
| `evaluate_single_ragas(csv_file, ragas_llm, ragas_embeddings)` | Runs RAGAS (faithfulness, answer_relevancy, context_precision, context_recall) on one CSV. Renames `question`→`user_input`, `answer`→`reference`, `generated_answer`→`response` automatically. |
| `evaluate_folder_ragas(input_folder, output_csv)` | Iterates all CSVs in a folder, runs `evaluate_single_ragas` on each, prints a summary table, saves results CSV. |

### `evaluation/generation_quality/thrity_open_ended_questions_eval/prompt_eval.py`

Unified prompt engineering evaluation pipeline across 6 prompt patterns and 30 open-ended TRCA questions. Uses LlamaIndex backend for RAGAS; GPT-4o-mini for AMLLM judge. Reads questions from `prompt_engineer_ragas/thrity_open_ended_questions.csv`.

| Class | Description |
|-------|-------------|
| `EvalConfig` | Dataclass holding all paths and settings for an evaluation run (LLM provider, patterns, output dirs, etc.). |
| `ResponseGenerator` | Queries `PromptingRAGEngine` for every question × pattern, writes answer `.txt` and prompt-log `.txt` files. |
| `CustomEvaluator` | Calls GPT-4o-mini to score each answer on 5 metrics (0–20 each). Saves JSON results and prints per-question tables. |
| `RagasEvaluator` | Builds a RAGAS dataset from saved prompt/answer files, runs faithfulness/relevancy/precision/recall via `LlamaIndexLLMWrapper`, saves CSV. |
| `ResultSummarizer` | Pivots custom and RAGAS results into per-metric summary CSVs; prints ranked average-score tables. |
| `EvalPipeline` | Orchestrates the four stages: `generate → custom_eval → ragas_eval → summarize`. |

---

## Quick Start

### 1. Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root. All components load keys via `Settings.from_env()`:

```bash
# .env (in project root)
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...          # for Gemini
ANTHROPIC_API_KEY=...       # for Claude
TAVILY_API_KEY=...          # for web search in agent mode
```

### 3. Build / Load the Vector DB

The system uses **Qdrant** (file-based, no server required) to store document embeddings at `./vector_db/qdrant_storage`. Before you can query documents, the vector database must contain an indexed collection.

#### Build from PDFs

Use `--reindex` to process all PDFs in the specified directory. This runs the full ingestion pipeline: PDF loading (PyMuPDF) → sentence-aware chunking (default 700 tokens, 50 overlap) → embedding → Qdrant collection creation.

```bash
# Index PDFs using OpenAI embeddings (1536 dimensions, requires API key)
python main.py --mode chat --model openai --embedding openai --reindex --pdf_dir data

# Index using HuggingFace embeddings (384 dimensions, free, no API key needed)
python main.py --mode chat --model openai --embedding huggingface --reindex --pdf_dir data

# Custom chunk size and collection name
python main.py --mode chat --model openai --embedding openai --reindex --pdf_dir data --chunk_size 500 --collection my_docs
```

The `--reindex` flag forces a full rebuild — it creates a new Qdrant collection from scratch, replacing any existing one with the same name. Indexing only needs to be done once per document set (or when documents change).

You can also upload individual PDFs through the Streamlit UI without re-indexing the entire collection (see [Streamlit Usage](#streamlit-usage)).

#### Load existing

If a collection already exists in the vector database, the system loads it automatically — no `--reindex` needed:

```bash
# Loads the existing "trca_documents" collection (default)
python main.py --mode chat --model openai --embedding openai

# Load a specific collection
python main.py --mode chat --model openai --embedding openai --collection my_docs
```

On startup, `main.py` checks `VectorStoreManager.collection_exists()`. If the collection is found, it calls `Indexer.load_index()` to load it directly. If not found, it falls back to indexing from `--pdf_dir`.

> **Important:** The `--embedding` provider used at query time must match the one used during indexing. OpenAI embeddings produce 1536-dimensional vectors; HuggingFace produces 384-dimensional vectors. Mismatched dimensions will cause an error.

---

## CLI Usage

All CLI modes are run through `main.py`:

### Interactive Chat (Conversational)

```bash
# Default (OpenAI LLM + OpenAI embeddings)
python main.py --mode chat --model openai --embedding openai

# With Gemini LLM
python main.py --mode chat --model gemini --embedding openai

# With Claude LLM
python main.py --mode chat --model claude --embedding openai

# With HuggingFace embeddings (must match indexing)
python main.py --mode chat --model openai --embedding huggingface

# Custom retrieval settings
python main.py --mode chat --model openai --embedding openai --top_k 10 --chunk_size 500

# Specific model name
python main.py --mode chat --model openai --embedding openai --model_name gpt-4o
```

Type questions at the prompt. Type `exit` to quit. The engine maintains conversation memory across turns.

### Agentic Mode (ReAct + Web Search)

```bash
# Agent with document search only
python main.py --mode agent --model openai

# Agent with document + web search
python main.py --mode agent --model openai --web_search
```

The agent autonomously decides whether to search your documents, search the web, or combine both to answer each question.

### Batch CSV Processing

```bash
python main.py --mode csv --model openai --embedding openai \
    --input_csv_file questions.csv --output_csv_path output/
```

The input CSV must have a `question` column. Output CSV includes `generated_answer` and `retrieved_contexts` columns.

### CLI Options Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `chat` | `chat` (conversational), `csv` (batch), `agent` (agentic) |
| `--model` | `openai` | LLM provider: `openai`, `gemini`, `claude` |
| `--model_name` | _(provider default)_ | Specific model (e.g. `gpt-4o`, `claude-opus-4-6`) |
| `--embedding` | `openai` | Embedding provider: `openai`, `huggingface` |
| `--model_path` | | Path to local GGUF model file (for llama_cpp) |
| `--n_ctx` | `8192` | Context window for local LLMs |
| `--max_tokens` | `1024` | Max output tokens |
| `--top_k` | `22` | Number of chunks to retrieve |
| `--chunk_size` | `700` | Chunk size for text splitting (during indexing) |
| `--collection` | `trca_documents` | Qdrant collection name |
| `--pdf_dir` | `data` | Directory containing source PDFs |
| `--reindex` | off | Force re-indexing of PDFs |
| `--web_search` | off | Enable web search tool (agent mode only) |
| `--pattern` | `rag-only` | Prompt pattern for PromptingRAGEngine |

---

## Streamlit Usage

### Starting the App

```bash
streamlit run app.py
```

### Main Chat Page

1. **Enter API key** in the sidebar (or set via `.env` — it will auto-load)
2. **Select LLM provider** (OpenAI, Gemini, Claude) and specific model from the dropdowns
3. **Toggle agentic mode** if you want ReAct reasoning with web search
4. **Ask questions** in the chat input box
5. **View sources** — click "Context Used for This Response" to see the top 5 retrieved chunks with document name, page number, and similarity score
6. **Manage conversations** — switch between conversations in the sidebar, create new ones, edit titles

### PDF Upload Page

Navigate to the upload page in the Streamlit sidebar:

1. **Upload one or more PDFs** — they are immediately chunked, embedded, and indexed into a `pdf_uploads` Qdrant collection
2. **The RAG engine rebuilds** automatically to include the new documents
3. **Uploaded files are tracked** — re-uploading the same file is skipped

---

## Evaluation

Evaluation is organised under `evaluation/generation_quality/` by dataset type.

### 1. Factual QA Evaluation — 200 questions (`200_factual_qa_eval/`)

`factual_qa_eval.py` combines BLEU/ROUGE (lexical) and RAGAS (semantic) evaluation into a single script that runs over a folder of CSV result files. It uses the **LlamaIndex backend** for RAGAS — no LangChain dependency.

**Input CSV columns** (column renaming is handled automatically):

| CSV column | RAGAS field | Notes |
|---|---|---|
| `question` | `user_input` | The question posed to the RAG system |
| `answer` | `reference` | Ground-truth / reference answer |
| `generated_answer` | `response` | Model output to evaluate |
| `retrieved_contexts` | `retrieved_contexts` | List of retrieved context strings |
| `alternative_answer` | _(BLEU/ROUGE only)_ | Optional extra reference |

```bash
# BLEU + ROUGE only
python evaluation/generation_quality/200_factual_qa_eval/factual_qa_eval.py --mode bleu_rouge

# RAGAS only  (faithfulness, answer_relevancy, context_precision, context_recall)
python evaluation/generation_quality/200_factual_qa_eval/factual_qa_eval.py --mode ragas

# Both — saves separate _bleu_rouge.csv and _ragas.csv outputs
python evaluation/generation_quality/200_factual_qa_eval/factual_qa_eval.py --mode both \
    --input_folder QA_pair/qa_pair_200_0210/output \
    --output_csv evaluation/generation_quality/200_factual_qa_eval/results.csv
```

**BLEU / ROUGE metrics:**

| Metric | What it measures |
|---|---|
| **BLEU-1/2/3** | N-gram precision between generated and reference answers |
| **ROUGE-1/2/3** | N-gram recall/F-measure (best score across references) |

**RAGAS metrics:**

| Metric | What it measures | Required columns |
|---|---|---|
| **Faithfulness** | Answer grounded in retrieved context? | `user_input`, `retrieved_contexts`, `response` |
| **Answer Relevancy** | Answer addresses the question? | `response`, `user_input` |
| **Context Precision** | Retrieved chunks relevant to question? | `user_input`, `retrieved_contexts`, `reference` |
| **Context Recall** | All necessary info retrieved? | `retrieved_contexts`, `user_input`, `reference` |

---

### 2. Prompt Engineering Evaluation — 30 open-ended questions (`thrity_open_ended_questions_eval/`)

`prompt_eval.py` evaluates all 6 prompt patterns across 30 TRCA open-ended questions using both GPT-4o-mini as judge (AMLLM-Auto-EVAL) and RAGAS. Also uses the **LlamaIndex backend** for RAGAS.

**Test question set** (`prompt_engineer_ragas/thrity_open_ended_questions.csv`):

| Project | Questions | Topics |
|---|---|---|
| German Mills Settlers Park | 10 | Erosion control, stakeholder roles, biodiversity |
| Humber Bay Park East | 10 | Shoreline maintenance, phased design, permits |
| Peacham Crescent | 10 | Environmental assessment, slope stabilization, Indigenous engagement |

**AMLLM-Auto-EVAL** — GPT-4o-mini scores each answer on 5 metrics (0–20 pts each, 100 total):

| Metric | What it measures |
|---|---|
| **Comprehensiveness** | Covers all aspects of the question? |
| **Accuracy** | Factually correct per TRCA documents? |
| **Relevance** | Stays on topic? |
| **Clarity** | Well-structured and easy to understand? |
| **Conciseness** | Appropriately brief without losing substance? |

```bash
# Run all stages (generate → custom_eval → ragas_eval → summarize)
python evaluation/generation_quality/thrity_open_ended_questions_eval/prompt_eval.py --stages all

# Run individual stages
python evaluation/generation_quality/thrity_open_ended_questions_eval/prompt_eval.py --stages generate
python evaluation/generation_quality/thrity_open_ended_questions_eval/prompt_eval.py --stages custom_eval
python evaluation/generation_quality/thrity_open_ended_questions_eval/prompt_eval.py --stages ragas_eval
python evaluation/generation_quality/thrity_open_ended_questions_eval/prompt_eval.py --stages summarize

# Specific patterns only
python evaluation/generation_quality/thrity_open_ended_questions_eval/prompt_eval.py \
    --patterns rag-only persona+cot+format --stages generate
```

**Prompt patterns evaluated:**

| Pattern | Components |
|---|---|
| `persona+cot+format` | PERSONA + COT + FORMAT + FEW_SHOT (full engineering) |
| `cot+format` | COT + FORMAT + FEW_SHOT |
| `persona+format` | PERSONA + FORMAT + FEW_SHOT |
| `persona+cot` | PERSONA + COT + FEW_SHOT |
| `rag-only` | Minimal — retrieval context only |
| `gpt-4o-mini` | Zero-shot baseline (no RAG context) |

**Evaluation Results (Paper Summary):**

| Configuration | AMLLM Score (0–100) |
|---|---|
| GPT-4 without RAG | 53.4 |
| GPT-4 with RAG | 75.7 |
| GPT-4 with RAG + persona+CoT+format | **88.9** |

---

## Key Design Patterns

| Pattern | Where Used | Purpose |
|---------|-----------|---------|
| **Factory** | `LLMFactory`, `EmbeddingFactory` | Provider-agnostic component creation |
| **Strategy** | `RAGEngine`, `PromptingRAGEngine`, `RAGAgent` | Swappable query strategies |
| **Template Method** | `BaseRAGPipeline` | Common interface for all engines |
| **Dataclass/DTO** | `SourceNode`, `RAGResponse`, `Settings` | Typed data containers |

---

## Important Constraints

- **Embedding consistency** — queries must use the same embedding model and dimensions as indexing. Mixing OpenAI (1536d) and HuggingFace (384d) causes errors.
- **Qdrant file-based** — no server required. Storage at `./vector_db/qdrant_storage`. Collections are created on first index.
- **data/ directory** — contains TRCA project PDFs. These are the source documents for the RAG system.
- **No formal test suite** — no unit tests or linting tools are configured.

---

## Dependencies

### Core RAG Framework
- `llama-index-core`, `llama-index-llms-openai`, `llama-index-llms-gemini`, `llama-index-llms-llama-cpp`
- `llama-index-embeddings-openai`, `llama-index-embeddings-huggingface`
- `llama-index-vector-stores-qdrant`, `llama-index-readers-file`, `llama-index-agent-openai`

### Vector Database
- `qdrant-client` (local file storage, no server)

### LLM Providers
- `openai`, `google-genai`

### UI
- `streamlit`

### Data Processing
- `pandas`, `tqdm`, `PyPDF2`, `PyMuPDF`, `nltk`

### Embeddings
- `sentence-transformers` (for HuggingFace local embeddings)

### Evaluation
- `ragas`, `rouge-score`, `nltk`, `datasets`

### Web Search (Optional)
- `tavily-python`

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| **Embedding dimension mismatch** | Ensure `--embedding` matches the provider used during indexing. Check error message for dimensions (384 vs 1536). |
| **Collection not found** | Run `python main.py --mode chat --reindex --pdf_dir data` to index documents first. |
| **No API key** | Create a `.env` file in the project root with `OPENAI_API_KEY=sk-...`. All components load keys via `Settings.from_env()`. HuggingFace embeddings require no API key. |
| **PDF parsing fails** | Re-export the PDF with a text layer (ensure it has a text layer, not just scanned images). |
| **LlamaCPP VRAM error** | The factory auto-negotiates context size (8192 → 4096 → 2048). Use `--n_ctx` to set manually. |
| **RAGAS evaluation fails** | Ensure `OPENAI_API_KEY` is set in `.env`. Check that `retrieved_contexts` column contains valid Python lists (not raw strings). |
| **Streamlit session issues** | Clear browser cache or restart with `streamlit run app.py --server.port 8502`. |

---

## Citation

If you use this codebase, please cite the associated paper (preprint/manuscript).

---

## License

MIT (unless specified otherwise in submodules). Please review data licensing for agency documents.
