# Agency-Specific Project Authoring Advisor (TRCA-LLM)

> Retrieval-Augmented Generation (RAG) system for agency-specific project authoring, validated with the Toronto and Region Conservation Authority (TRCA).

---

## Overview

This repository implements an **agency-specific project authoring advisor**: a lightweight, deployable RAG system that helps engineers draft **project scopes**, **design notes**, and retrieve **past project information** directly from an agency’s technical documents (e.g., reports, plans, assessments).

The system combines:
- **Semantic retrieval** over an agency-curated Technical Document Database (TDD)
- **Domain-aware prompting** (persona, structured format, chain-of-thought, few-shot)
- **LLM-assisted, adversarial prompt evaluation** (AMLLM-Auto-EVAL)
- A **chat-style Streamlit GUI** with **on-the-fly document ingestion**

It is **model-agnostic for embeddings** (e.g., Sentence-BERT, E5, GTE, Instructor) and supports both **closed** (e.g., GPT-4/Gemini via API) and **open** (e.g., Llama) generators.

> **Practical outcome:** Faster, more consistent authoring grounded in institutional knowledge, with transparent citations to source passages.

---


## Key Capabilities

- **Ask in plain language**: “What are the erosion control measures for Site P-531?”  
- **Grounded answers**: The advisor retrieves top‑K passages from the agency TDD and synthesizes a concise, cited response.
- **Prompt patterns**: Persona + format template + chain‑of‑thought (+ few shot) for clearer, auditable outputs.
- **MapReduce QA**: Scales to long inputs by mapping answers per passage chunk and reducing to a final response.
- **On‑the‑fly ingestion**: Upload PDFs/Excels via the GUI; new files are embedded and searchable immediately.
- **Role-agnostic, phase-aware**: Useful across planning, design, and maintenance phases.

---

## System Architecture

```
           ┌────────────────────────────────────────────────┐
           │            Technical Document Database          │
           │ (PDF/Excel → text cleaning → chunking → meta)  │
           └────────────────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Embeddings (custom) │  ← Sentence-BERT / E5 / GTE / Instructor / etc.
                    └──────────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   FAISS Vector DB    │
                    └──────────────────────┘
                               │
                     retrieve top‑K passages
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ Prompted Generator (GPT‑4/Gemini/Llama)                                      │
│  • Persona + Format + CoT (+ few-shot)                                       │
│  • Map phase: generate per-passage answers                                   │
│  • Reduce phase: aggregate, de-duplicate, reconcile, cite                    │
└──────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
                      Streamlit Chat UI (multi‑turn, uploads)
```

---

## Evaluation (Paper Summary)

- **Question set**: 30 questions across **10 categories** mirroring real tasks (e.g., future projects, maintenance strategies, stakeholder responsibilities).  
- **Metrics**: **Comprehensiveness**, **Accuracy**, **Relevance**, **Clarity**, **Conciseness** (0–100 composite).  
- **Headline results**:  
  - GPT‑4 **with RAG**: **75.7/100** vs **53.4/100** without RAG.  
  - With persona+format+CoT prompting, best configuration reached **88.9/100**.  
  - Gemini and Llama showed **similar improvements** with RAG and prompting.

> We are working with **TRCA** to make the advisor available to staff in different roles. After a period of real‑world use, we will analyze interaction logs and outcomes to quantify role‑specific impacts.

---

## Repository Structure

```
config/                    # API keys, env config (do NOT commit secrets)
data/                      # Source PDFs/Excels (agency documents)
docs/                      # Module documentation, evaluation details
embeddings/                # Embedding helpers (provider-agnostic)
evaluation/                # BLEU/ROUGE and RAGAS evaluation scripts
output/                    # Extracted/cleaned text from PDFs
Pages/                     # Streamlit subpages
pipeline_preprocess_file_exec/    # PDF processing helpers
pipelines/                 # End-to-end pipelines
preprocessing/             # Cleaning, segmentation, metadata
prompt_engineer_ragas/     # Prompt templates, generation, custom eval
QA_pair/                   # QA logs/CSV outputs
streamlit_class/           # Streamlit utilities (conversation state)
trials/                    # Scratch experiments
tuning_scripts/            # Hyperparameter tuning
utils/                     # Misc utility scripts (e.g., pdf_extractor.py)
vector_db/                 # FAISS index storage
StreamLitUI.py             # Streamlit entry point
main.py                    # CLI entry point (chat/csv modes)
```

> We retained folder roles from the earlier project but updated usage to reflect the paper’s pipeline.

---

## Quick Start

### 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Configure Keys (optional)

If you plan to use hosted LLMs (OpenAI, Google, etc.), set keys via environment or `config/keys.py`:

```python
# config/keys.py
OPENAI_API_KEY = "sk-..."
GOOGLE_API_KEY = "..."   # if using Gemini
```

### 3) Choose an Embedding Model (local/custom)

We recommend local Sentence-Transformers by default (no external API needed):

```python
from embeddings import get_hf_embeddings
emb = get_hf_embeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda",                       # or "cpu"
    normalize_embeddings=True,
)
```

You can also pass any **LangChain-compatible Embeddings** implementation (e.g., E5/GTE/Instructor).

### 4) Build / Load the Vector DB

- **Option A — Build from PDFs**

```python
from embeddings import create_vector_db_from_pdfs
vector_db = create_vector_db_from_pdfs("./data/ALL_PDFS", embedding=emb)
vector_db.save_local("./vector_db/faiss_store")
```

- **Option B — Load existing**

```python
from database import load_vector_db
vector_db = load_vector_db("./vector_db/faiss_store", embedding=emb)
```

> ⚠️ Always load with the **same embedding** configuration used to build the index.

### 5) Run (CLI)

- **Interactive chat**:

```bash
python main.py --mode chat
```

- **Batch CSV** (evaluate a column of questions and save answers):

```bash
python main.py --mode csv --input_csv_file path/to/input.csv --output_csv_path path/to/output/
```

### 6) Run (Streamlit GUI)

```bash
streamlit run StreamLitUI.py
```

In the browser:
- Enter API key (if needed for your generator).
- Ask questions, upload PDFs/Excels; new documents become searchable immediately.

---

## Prompting & AMLLM-Auto-EVAL

- **Prompt patterns**: Persona, Format Template, Chain-of-Thought, Few-shot.  
- **LLM-assisted evaluation**: A **single-prompt** judge scores answers across five metrics, enabling **adversarial** and **repeatable** template selection without costly human loops.  
- **Tuning**: Use `tuning_scripts/` to sweep **top_k**, **chunk_size**, **search_type**.  

Example (custom evaluator):
```bash
python prompt_engineer_ragas/custom_eval.py
```

RAGAS metrics:
```bash
python -m evaluation.folder_eval_ragas
```

BLEU/ROUGE:
```bash
python evaluation/folder_eval.py
```

---

## Practical Applications

- **Engineers** author clearer scopes and design notes faster, with cited evidence.  
- **Managers** gain consistent, phase-aware summaries across projects.  
- **Agencies** formalize institutional knowledge in a reusable knowledge base, enabling repeatable adoption across teams.  

---

## Troubleshooting

- **Embeddings mismatch**: Queries must use the **same model and settings** as indexing.  
- **FAISS I/O**: Prefer `save_local` / `load_local` (see `database.py`).  
- **PDF parsing**: If a file fails, re-export with text layer; check `utils/pdf_extractor.py`.  
- **Token limits**: Long contexts automatically split; MapReduce aggregates final answers.  
- **No API keys**: Use local generators/embeddings to avoid external dependencies.

---

## Citation

If you use this codebase, please cite the associated paper (preprint/manuscript).

---

## License

MIT (unless specified otherwise in submodules). Please review data licensing for agency documents.
