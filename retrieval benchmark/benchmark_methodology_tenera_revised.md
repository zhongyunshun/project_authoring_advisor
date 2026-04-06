# Benchmark Dataset Curation & Retrieval Evaluation Methodology
## Tenera Building Maintenance Advisor (Agentic System)

---

## Overview

This document defines the complete methodology for:
1. **PDF ingestion** into chunk-agnostic, benchmark-compatible units
2. **Benchmark dataset curation** — how to produce QA pairs and label gold-relevant chunks
3. **Retrieval evaluation** — metrics, tooling, and interpretation
4. **System improvement** — how to close the feedback loop from benchmark results

The core design principle: **the benchmark must be independent of any particular chunking strategy**, so that you can sweep chunk sizes, overlap, and splitting methods without re-labeling.

---

## Part 1: PDF Ingestion — Chunk-Agnostic Design

### The Problem with Fixed-Size Chunking in Benchmarks

Don't use fixed-size chunking such as `SentenceSplitter(chunk_size=700, chunk_overlap=50)`. If we label gold chunks at this size, then switching to 384-token chunks invalidates all labels. The solution is to anchor the benchmark to **atomic passage units** — logical sections of a document that are semantically complete — and let the chunker produce chunks that map back to these units.

A second problem: a bare synthetic `passage_id` is not always sufficient for human review and audit. Labelers often need to know **which section** a passage belongs to, and sometimes **which paragraph** inside that passage contains the exact answer. Therefore, every passage must carry:
- a stable synthetic `passage_id`
- a required `section_id`
- zero or more `paragraph_ids`

### Step 1: Structural Parsing (Before Chunking)

Use PyMuPDF to extract **structural blocks**, not raw text runs. During parsing, assign:
- `section_id` from heading numbering where available (e.g. `3.2`, `4.1.3`)
- `section_title`
- `paragraph_id` for non-heading blocks, local to section
- `block_idx` for reproducibility

```python
import re
import fitz  # PyMuPDF

SECTION_ID_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.*)$")

def parse_heading(text: str) -> tuple[str | None, str]:
    """Return (section_id, section_title)."""
    m = SECTION_ID_RE.match(text.strip())
    if m:
        return m.group(1), m.group(2).strip()
    return None, text.strip()

def extract_structural_blocks(pdf_path: str) -> list[dict]:
    """
    Extract text blocks with structure metadata.
    Each block = one paragraph/heading/table-cell/caption.
    """
    doc = fitz.open(pdf_path)
    blocks = []
    current_section_id = "root"
    current_section_title = None
    paragraph_counter_by_section = {}

    for page_num, page in enumerate(doc):
        raw_blocks = page.get_text("dict")["blocks"]
        for block in raw_blocks:
            if block["type"] != 0:  # skip images
                continue

            lines = block.get("lines", [])
            text = " ".join(
                span["text"]
                for line in lines
                for span in line["spans"]
            ).strip()
            if not text:
                continue

            avg_size = sum(
                span["size"]
                for line in lines
                for span in line["spans"]
            ) / max(1, sum(len(line["spans"]) for line in lines))

            is_heading = avg_size > 13  # tune per corpus

            if is_heading:
                parsed_section_id, parsed_title = parse_heading(text)
                current_section_id = parsed_section_id or f"unnumbered::{len(blocks)}"
                current_section_title = parsed_title
                paragraph_id = None
            else:
                paragraph_counter_by_section.setdefault(current_section_id, 0)
                paragraph_counter_by_section[current_section_id] += 1
                paragraph_id = f"{current_section_id}¶{paragraph_counter_by_section[current_section_id]}"

            blocks.append({
                "doc_id": pdf_path,
                "page": page_num + 1,
                "block_idx": len(blocks),
                "text": text,
                "bbox": block["bbox"],
                "avg_font_size": round(avg_size, 1),
                "is_heading": is_heading,
                "section_id": current_section_id,
                "section_title": current_section_title,
                "paragraph_id": paragraph_id,
                "char_count": len(text),
            })

    return blocks
```

### Step 2: Logical Passage Assembly

Group consecutive blocks into **logical passages** anchored to headings. Each passage is your benchmark's atomic unit — it can be labelled once and remains stable even as chunk sizes change.



```python
def assemble_passages(blocks: list[dict], min_chars=200, max_chars=1500) -> list[dict]:
    """
    Group blocks under the same heading into passages.
    A passage = heading block + body blocks until next heading.
    """
    passages = []
    current_heading = None
    current_section_id = "root"
    current_blocks = []

    for block in blocks:
        if block["is_heading"]:
            if current_blocks:
                passages.append(_make_passage(
                    heading=current_heading,
                    section_id=current_section_id,
                    blocks=current_blocks,
                ))
            current_heading = block["section_title"] or block["text"]
            current_section_id = block["section_id"]
            current_blocks = []
        else:
            current_blocks.append(block)

    if current_blocks:
        passages.append(_make_passage(
            heading=current_heading,
            section_id=current_section_id,
            blocks=current_blocks,
        ))

    return passages

def _make_passage(heading, section_id, blocks):
    paragraph_ids = [b["paragraph_id"] for b in blocks if b["paragraph_id"]]
    return {
        "passage_id": f"{blocks[0]['doc_id']}::sec{section_id}::p{blocks[0]['block_idx']}",
        "doc_id": blocks[0]["doc_id"],
        "heading": heading,
        "section_id": section_id,
        "pages": sorted(set(b["page"] for b in blocks)),
        "text": "\n".join(b["text"] for b in blocks),
        "block_indices": [b["block_idx"] for b in blocks],
        "paragraph_ids": paragraph_ids,
    }
```

### Step 3: Chunk-to-Passage Mapping

When you chunk (at any size), annotate each chunk with:
- parent `passage_id`
- parent `section_id` (**required**)
- covered `paragraph_ids` (**optional but strongly recommended**)

```python
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

def chunk_passage(passage: dict, chunk_size: int, chunk_overlap: int) -> list[TextNode]:
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents([
        Document(text=passage["text"], metadata={
            "passage_id": passage["passage_id"],
            "doc_id": passage["doc_id"],
            "heading": passage["heading"],
            "section_id": passage["section_id"],
            "pages": passage["pages"],
            "paragraph_ids": passage["paragraph_ids"],
        })
    ])
    for i, node in enumerate(nodes):
        node.metadata["chunk_idx_in_passage"] = i
        node.metadata["chunk_total_in_passage"] = len(nodes)
    return nodes
```

If you later implement sentence-aware or span-aware chunking, you should compute the exact subset of `paragraph_ids` covered by each chunk. If you cannot compute exact paragraph coverage, store the full `paragraph_ids` list for the passage and mark chunk-level paragraph coverage as approximate.

**Why this matters:** your gold labels no longer point to a bare `passage_id` only. They point to a structured reference:

```json
{
  "passage_id": "data/Westfield Office Tower/WOT_maintenance_plan.pdf::sec3.2::p12",
  "section_id": "3.2",
  "paragraph_id": "3.2¶2"
}
```

At evaluation time, a retrieved chunk is:
- a **full hit** if `passage_id` and `section_id` match, and the gold item does not require a paragraph
- also a **full hit** if `paragraph_id` is present in the gold reference and that paragraph is covered by the chunk
- a **partial structural hit** if passage/section matches but required paragraph is absent

### Stored Format for Passages

Save parsed passages as JSONL — one passage per line:

```jsonl
{"passage_id": "data/Westfield Office Tower/WOT_maintenance_plan.pdf::sec3.2::p12", "doc_id": "data/Westfield Office Tower/WOT_maintenance_plan.pdf", "heading": "HVAC Preventive Maintenance Schedule", "section_id": "3.2", "pages": [8, 9], "paragraph_ids": ["3.2¶1", "3.2¶2", "3.2¶3"], "text": "Air handling units shall be inspected quarterly and filters replaced every 90 days...", "block_indices": [47, 48, 49, 50]}
{"passage_id": "data/Riverside Plaza/RP_cost_estimate.pdf::sec5.1::p5", "doc_id": "data/Riverside Plaza/RP_cost_estimate.pdf", "heading": "Roof Replacement Cost Breakdown", "section_id": "5.1", "pages": [3], "paragraph_ids": ["5.1¶1", "5.1¶2"], "text": "Unit costs for TPO membrane installation were derived from RS Means 2023...", "block_indices": [18, 19]}
```

---

## Part 2: Benchmark Dataset Curation

### 2.1 Dataset Structure

Each benchmark sample (one row in your dataset) contains structured gold passage references:

```json
{
  "qid": "q_0042",
  "question": "What is the recommended inspection frequency for rooftop HVAC units at Westfield Office Tower?",
  "question_type": "factual_lookup",
  "gold_passage_ids": [
    {
      "passage_id": "data/Westfield Office Tower/WOT_maintenance_plan.pdf::sec3.2::p12",
      "section_id": "3.2",
      "paragraph_id": "3.2¶2"
    }
  ],
  "gold_passage_ids_partial": [
    {
      "passage_id": "data/Westfield Office Tower/WOT_maintenance_plan.pdf::sec3.2::p14",
      "section_id": "3.2"
    }
  ],
  "gold_answer": "Rooftop HVAC units are to be inspected quarterly, with filters replaced every 90 days and coil cleaning performed annually.",
  "gold_answer_alt": "Quarterly inspections, 90-day filter replacement cycle, annual coil cleaning.",
  "source_doc": "data/Westfield Office Tower/WOT_maintenance_plan.pdf",
  "difficulty": "easy",
  "requires_multi_hop": false,
  "labeler": "human",
  "labeler_confidence": 3,
  "notes": ""
}
```

**Field definitions:**

| Field | Type | Description |
|---|---|---|
| `qid` | str | Unique question ID |
| `question` | str | The query as posed |
| `question_type` | enum | See taxonomy below |
| `gold_passage_ids` | list[object] | **Required** evidence refs; each object must contain `passage_id` and `section_id`, and may contain `paragraph_id` |
| `gold_passage_ids_partial` | list[object] | Helpful evidence refs; same object schema as above |
| `gold_answer` | str | Reference answer for generation eval |
| `gold_answer_alt` | str | Acceptable paraphrase |
| `source_doc` | str | Which PDF this came from |
| `difficulty` | easy/medium/hard | Subjective label |
| `requires_multi_hop` | bool | True if answer spans >1 passage |
| `labeler` | human/llm/hybrid | How was this generated |
| `labeler_confidence` | 1–3 | Labeler's confidence in gold labels |

**Reference object schema:**

```json
{
  "passage_id": "string, required",
  "section_id": "string, required",
  "paragraph_id": "string, optional"
}
```

Use `paragraph_id` only when:
- the answer is localized to one paragraph inside a longer passage
- you want stricter citation grounding
- you need paragraph-level auditability for human review

Otherwise, store only `passage_id` + `section_id`.

### 2.2 Question Type Taxonomy

Question type drive which metrics matter most for each question.

| Type | Description | Example |
|---|---|---|
| `factual_lookup` | Single fact, single passage | "What is the warranty period for the membrane roofing installed at Riverside Plaza?" |
| `definition` | Defines a term used in Tenera docs | "What does Tenera mean by 'preventive maintenance cycle'?" |
| `numeric_extraction` | Extract a specific number/unit | "What was the total labour cost for the elevator modernization at Westfield Tower?" |
| `procedural` | Step-by-step process | "What is the procedure for winterizing the chilled water system at Harbour Centre?" |
| `multi_hop` | Requires connecting 2+ passages | "Which facilities used both corrective and preventive maintenance for their boiler systems, and what were the recorded costs?" |
| `comparison` | Compare two projects or options | "How did the facade inspection scope differ between Riverside Plaza and Harbour Centre?" |
| `scope_authoring` | Technician drafting a procedure or estimate | "Draft the maintenance scope and preliminary cost estimate for a similar mid-rise HVAC overhaul." |

### 2.3 QA Pair Generation — Hybrid Labeling Pipeline

Use a **three-stage pipeline**: LLM draft → human review → confidence scoring.

#### Stage 1: LLM-Assisted Question Generation per Passage

We can use LLM-assisted question generation to generate `factual_lookup`, `definition`, and `numeric_extraction` questions. For each passage in the JSONL corpus, prompt an LLM to generate candidate QA pairs:

```python
GENERATION_PROMPT = """
You are a technical reviewer for Tenera building maintenance and cost estimation documents.

Given the following passage from a Tenera technical document, generate {n_questions} high-quality questions
that a facility manager or cost estimator might ask when drafting a maintenance procedure or cost estimate.

PASSAGE:
---
Heading: {heading}
Section ID: {section_id}
Source: {doc_id}, Pages {pages}
Paragraph IDs: {paragraph_ids}
Text:
{text}
---

Requirements:
- Each question must be answerable SOLELY from this passage (or note if it requires context from elsewhere)
- Vary question types across: factual_lookup, numeric_extraction, procedural, definition
- Do not generate trivially obvious questions ("What document is this from?")
- For each question, identify the minimum set of paragraphs in the passage that contain the answer

Output JSON array only, no commentary:
[
  {{
    "question": "...",
    "question_type": "...",
    "answer": "...",
    "answer_paragraph_ids": ["3.2¶2"],
    "requires_other_passages": false,
    "difficulty": "easy|medium|hard"
  }}
]
"""
```

Run prompts similar to this against every passage. For a corpus of ~100 Tenera documents with ~20 passages each, we get ~2000 candidates before filtering.

#### Stage 2: LLM Cross-Passage (Multi-Hop) Generation

But for `procedural`, `multi_hop`, `comparison`, `scope_authoring` type questions, we need domain experts.

Questions may include:
- Questions they would actually ask when starting a new maintenance procedure or cost estimate
- Questions about past maintenance decisions that need institutional knowledge
- Scope authoring prompts ("Draft the preventive maintenance schedule and cost estimate for a mid-rise building's electrical systems...")

We may select passage pairs strategically: same building (different maintenance systems), same trade across buildings (e.g. roofing costs at Riverside Plaza + Westfield Tower), or contrasting maintenance approaches (corrective vs. preventive for the same asset type).

When generating multi-hop samples, keep the same structured gold reference format. Example:

```json
{
  "qid": "q_0118",
  "question": "What preventive maintenance schedule and annual cost are associated with rooftop HVAC units at Westfield Office Tower?",
  "question_type": "multi_hop",
  "gold_passage_ids": [
    {
      "passage_id": "data/Westfield Office Tower/WOT_maintenance_plan.pdf::sec3.2::p12",
      "section_id": "3.2"
    },
    {
      "passage_id": "data/Westfield Office Tower/WOT_cost_estimate.pdf::sec7.4::p9",
      "section_id": "7.4",
      "paragraph_id": "7.4¶1"
    }
  ],
  "gold_passage_ids_partial": [],
  "gold_answer": "Rooftop HVAC units are inspected quarterly, with 90-day filter replacement and annual coil cleaning, and the annual allocated maintenance cost is $4,800."
}
```

#### Stage 3: Human Review

Have labelers review each LLM-generated QA pair using this rubric:

```
For a portion candidate QA pairs in Stage 1, a human labeler does:

1. READ the passage(s) identified as gold
2. VERIFY the question is sensible for an engineer's use case
3. VERIFY the answer is correct and complete given the passage
4. ADJUST gold_passage_ids / gold_passage_ids_partial if the LLM missed or added wrong ones
5. ENSURE every gold reference includes:
   - passage_id (required)
   - section_id (required)
   - paragraph_id (optional, only if needed)
6. SCORE confidence: 1 (unsure), 2 (likely correct), 3 (certain) (Optional)
7. MARK status: KEEP / REVISE / REJECT

Common rejection reasons:
- Answer not actually in passage (LLM hallucinated)
- Question is too trivial
- Question requires domain knowledge not in corpus
- Gold passage set is wrong
- section_id is missing or inconsistent with the source passage
- paragraph_id points to the wrong paragraph
```

Labelers can work in a simple spreadsheet (transformed to JSON after annotation) or even a raw JSON file.

### 2.4 Gold Passage Labeling

**Two-tier labeling** at the passage level:

```
Tier 1 — Required (gold_passage_ids):
  The query CANNOT be correctly answered without this passage.
  Required fields per item:
    - passage_id
    - section_id
  Optional field:
    - paragraph_id
  Label: 2 (highly relevant)

Tier 2 — Helpful (gold_passage_ids_partial):
  This passage provides supporting context but the core answer is elsewhere.
  Required fields per item:
    - passage_id
    - section_id
  Optional field:
    - paragraph_id
  Label: 1 (partially relevant)

Everything else:
  Label: 0 (irrelevant)
```

This three-level scheme enables NDCG@k as an evaluation metric.

**Labeling protocol for humans:**

```
1. Given: question Q
2. Search yourself: scan all passages (ctrl+F in the source PDF if needed)
3. For each passage you find relevant, ask:
   "If I removed ONLY this passage from the context,
    could the LLM still answer correctly?"
   → YES: it's Tier 2 (partial)
   → NO:  it's Tier 1 (required)
4. Record a structured reference for each relevant passage:
   - always include passage_id
   - always include section_id
   - include paragraph_id only if the answer is localized to a specific paragraph
5. Write the gold_answer using only what is in Tier 1 passages
6. For paragraph-specific labels, verify the paragraph text really contains the answer span
```

**Example:**

```json
{
  "qid": "q_0061",
  "question": "What is the total labour allowance for the elevator modernization at Westfield Tower?",
  "gold_passage_ids": [
    {
      "passage_id": "data/Westfield Tower/WOT_elevator_estimate.pdf::sec6.3::p21",
      "section_id": "6.3",
      "paragraph_id": "6.3¶1"
    }
  ],
  "gold_passage_ids_partial": [
    {
      "passage_id": "data/Westfield Tower/WOT_elevator_estimate.pdf::sec6.3::p22",
      "section_id": "6.3"
    }
  ],
  "gold_answer": "$38,500"
}
```

### 2.5 Target Dataset Composition

For your current Tenera document corpus (maintenance plans, cost estimates, inspection reports, procedure manuals), aim for:

| Split | Count | How Generated |
|---|---|---|
| Development set | 50–80 | LLM-gen + human review; use for tuning |
| Test set | 100–150 | Human-written + human-reviewed LLM; locked, not for tuning |
| **Total** | **150–230** | |

Question type distribution (approximate):
- factual_lookup: 35%
- definition: 10%
- numeric_extraction: 15%
- procedural: 15%
- multi_hop: 15%
- comparison: 5%
- scope_authoring: 5%

### 2.6 Storage Format

```
benchmark/
├── passages.jsonl                 # All extracted passages (stable, chunk-agnostic)
├── dev_set.jsonl                  # Development QA pairs with structured gold refs
├── test_set.jsonl                 # Test QA pairs (locked)
├── labeling/
│   ├── raw_llm_candidates.jsonl   # Pre-review LLM output
│   ├── review_log.csv             # Human review decisions
│   └── inter_annotator/           # IAA samples
└── README.md                      # Dataset card
```

`passages.jsonl` contains passage-level structure, including `section_id` and `paragraph_ids`.

`dev_set.jsonl` and `test_set.jsonl` each contain one JSON object per line, matching the schema in §2.1.

---

## Part 3: Retrieval Evaluation

### 3.1 Chunk-to-Passage / Section / Paragraph Resolution at Eval Time

At evaluation time, your retriever returns **chunks** (at whatever `chunk_size` you're testing). Before computing metrics, map chunks to structured passage references:

```python
def resolve_chunks_to_passage_refs(retrieved_chunks: list[dict], top_k: int) -> list[dict]:
    """
    Deduplicate retrieved chunks by (passage_id, section_id),
    preserving rank order (first appearance of each passage-section pair).

    Each returned item has:
      - passage_id
      - section_id
      - paragraph_ids (list[str], may be empty)
    """
    seen = set()
    resolved = []

    for chunk in retrieved_chunks[:top_k]:
        metadata = chunk["metadata"]
        key = (metadata["passage_id"], metadata["section_id"])
        if key in seen:
            continue
        seen.add(key)
        resolved.append({
            "passage_id": metadata["passage_id"],
            "section_id": metadata["section_id"],
            "paragraph_ids": metadata.get("paragraph_ids", []),
        })
    return resolved
```

Define helper functions for matching:

```python
def ref_key(ref: dict) -> tuple[str, str, str | None]:
    return (
        ref["passage_id"],
        ref["section_id"],
        ref.get("paragraph_id"),
    )

def retrieved_matches_gold(retrieved_ref: dict, gold_ref: dict) -> bool:
    """
    A retrieved chunk hits a gold reference if:
    - passage_id matches
    - section_id matches
    - and if gold has paragraph_id, that paragraph is covered by the chunk
    """
    if retrieved_ref["passage_id"] != gold_ref["passage_id"]:
        return False
    if retrieved_ref["section_id"] != gold_ref["section_id"]:
        return False

    gold_paragraph_id = gold_ref.get("paragraph_id")
    if gold_paragraph_id is None:
        return True

    return gold_paragraph_id in set(retrieved_ref.get("paragraph_ids", []))
```

Now you can compute metrics with structured refs regardless of chunk size.

### 3.2 Core Metrics

#### Recall@k

Most important for RAG. Measures: "Did we retrieve all the evidence the LLM needs?"

```python
def recall_at_k(retrieved_refs: list[dict], gold_refs: list[dict], k: int) -> float:
    """
    retrieved_refs: ordered list from retriever (after chunk→passage ref resolution)
    gold_refs: Tier 1 required refs for this query
    """
    top_k = retrieved_refs[:k]
    hits = 0
    for gold_ref in gold_refs:
        if any(retrieved_matches_gold(r, gold_ref) for r in top_k):
            hits += 1
    return hits / len(gold_refs) if gold_refs else 0.0
```

**Example:**
- Query: "What is the recommended inspection frequency for rooftop HVAC units at Westfield Office Tower?"
- Gold required refs:
  ```json
  [
    {
      "passage_id": "WOT_maintenance_plan.pdf::sec3.2::p12",
      "section_id": "3.2",
      "paragraph_id": "3.2¶2"
    }
  ]
  ```
- Retrieved top-5 includes a chunk from the same passage and section that covers `3.2¶2`
- Recall@5 = 1/1 = **1.0** ✓

**Example (multi-hop):**
- Gold required refs:
  ```json
  [
    {"passage_id": "WOT_maintenance_plan.pdf::sec3.2::p12", "section_id": "3.2"},
    {"passage_id": "WOT_cost_estimate.pdf::sec7.4::p7", "section_id": "7.4"}
  ]
  ```
- Retrieved top-5 contains only the first
- Recall@5 = 1/2 = **0.5** — critical gap

#### NDCG@k

Best for ranked evaluation when you have partial relevance labels:

```python
import numpy as np

def relevance_grade(retrieved_ref: dict,
                    gold_required: list[dict],
                    gold_partial: list[dict]) -> int:
    if any(retrieved_matches_gold(retrieved_ref, g) for g in gold_required):
        return 2
    if any(retrieved_matches_gold(retrieved_ref, g) for g in gold_partial):
        return 1
    return 0

def ndcg_at_k(retrieved_refs: list[dict],
              gold_required: list[dict],
              gold_partial: list[dict],
              k: int) -> float:
    gains = [
        relevance_grade(ref, gold_required, gold_partial)
        for ref in retrieved_refs[:k]
    ]

    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))

    ideal_gains = ([2] * len(gold_required) + [1] * len(gold_partial))[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal_gains))

    return dcg / idcg if idcg > 0 else 0.0
```

NDCG@k is a ranking metric for retrieval. It rewards you for putting more relevant items higher in the top-k results.

Practical interpretation:
- 1.0: perfect ordering in top-k
- 0.8–0.9: strong ranking
- 0.5–0.7: mixed quality
- near 0: poor ranking

#### MRR (Mean Reciprocal Rank)

Measures how quickly the first correct passage appears. Use it when there is usually one main correct item. Good for single-hop factual queries:

```python
def reciprocal_rank(retrieved_refs: list[dict], gold_refs: list[dict]) -> float:
    for rank, ref in enumerate(retrieved_refs, start=1):
        if any(retrieved_matches_gold(ref, gold_ref) for gold_ref in gold_refs):
            return 1.0 / rank
    return 0.0
```

Practical interpretation:

Higher is better:
- 1.0 means every query had a relevant result at rank 1
- 0.5 means, on average, first relevant result is around rank 2
- 0.25 means around rank 4
- low MRR means users must scan too far down

#### Precision@k

Measures noise in the retrieved context. Less critical than recall for RAG, but important when `top_k` is small:

```python
def precision_at_k(retrieved_refs: list[dict], gold_refs: list[dict], k: int) -> float:
    top_k = retrieved_refs[:k]
    hits = sum(
        1 for ref in top_k
        if any(retrieved_matches_gold(ref, gold_ref) for gold_ref in gold_refs)
    )
    return hits / k if k > 0 else 0.0
```

### 3.3 Evaluation Runner

```python
def evaluate_retrieval(benchmark: list[dict],
                       retriever,           # callable: query -> list[chunk_dicts]
                       top_k: int = 5) -> dict:
    """
    benchmark: list of QA dicts from dev_set.jsonl or test_set.jsonl
    """
    results = []
    for sample in benchmark:
        query = sample["question"]
        gold_required = sample["gold_passage_ids"]
        gold_partial = sample.get("gold_passage_ids_partial", [])

        raw_chunks = retriever(query, top_k=top_k * 3)  # over-retrieve, then dedup
        retrieved_refs = resolve_chunks_to_passage_refs(raw_chunks, top_k=top_k)

        results.append({
            "qid": sample["qid"],
            "question": sample["question"],
            "qtype": sample["question_type"],
            "gold_passage_ids": gold_required,
            "gold_passage_ids_partial": gold_partial,
            "retrieved_refs": retrieved_refs,
            "recall_5": recall_at_k(retrieved_refs, gold_required, k=5),
            "recall_10": recall_at_k(retrieved_refs, gold_required, k=10),
            "ndcg_5": ndcg_at_k(retrieved_refs, gold_required, gold_partial, k=5),
            "mrr": reciprocal_rank(retrieved_refs, gold_required),
            "precision_5": precision_at_k(retrieved_refs, gold_required, k=5),
        })

    df = pd.DataFrame(results)
    summary = {
        "Recall@5": df["recall_5"].mean(),
        "Recall@10": df["recall_10"].mean(),
        "NDCG@5": df["ndcg_5"].mean(),
        "MRR": df["mrr"].mean(),
        "Precision@5": df["precision_5"].mean(),
        "by_qtype": df.groupby("qtype")[["recall_5", "ndcg_5"]].mean().to_dict(),
    }
    return summary
```

### 3.4 Reporting Format

Store each experiment run as:

```json
{
  "run_id": "exp_007",
  "timestamp": "2025-04-03T14:22:00",
  "config": {
    "chunk_size": 512,
    "chunk_overlap": 100,
    "embedding_model": "text-embedding-3-small",
    "retriever": "dense",
    "top_k": 5
  },
  "metrics": {
    "Recall@5": 0.71,
    "Recall@10": 0.84,
    "NDCG@5": 0.63,
    "MRR": 0.58,
    "Precision@5": 0.42
  },
  "by_qtype": {
    "factual_lookup": {"recall_5": 0.88, "ndcg_5": 0.79},
    "multi_hop": {"recall_5": 0.41, "ndcg_5": 0.35},
    "numeric_extraction": {"recall_5": 0.76, "ndcg_5": 0.68}
  }
}
```

Save to `evaluation/retrieval_runs/exp_007.json`. Build a summary table across runs to track progress.

---

## Part 4: System Improvement Loop

### 4.1 Chunking Sweep

Run the evaluation runner across a grid of chunk sizes:

```python
CHUNK_GRID = [
    {"chunk_size": 256, "chunk_overlap": 50},
    {"chunk_size": 384, "chunk_overlap": 50},
    {"chunk_size": 512, "chunk_overlap": 100},
    {"chunk_size": 768, "chunk_overlap": 100},
    {"chunk_size": 1024, "chunk_overlap": 200},
]
```

For each config: re-index → run eval on dev_set → record metrics. Because passages are stable, you don't re-label. Pick the config that maximizes Recall@5 across all question types, then check Precision@5 to avoid excessive noise.

**Key insight for your domain:** Tenera maintenance documents typically contain complete procedures or cost line items within a 200–400 word section. Cost tables and step-by-step procedures are especially sensitive to splitting — a chunk that cuts a labour rate table in half, or separates a procedure step from its safety note, will degrade both retrieval and generation quality. Expect 384–512 tokens (≈ 280–370 words) to perform best, with larger sizes (768) worth testing for narrative inspection reports.

### 4.2 Hybrid Retrieval

Once chunk size is fixed, add BM25 sparse retrieval alongside dense:

```python
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

bm25_retriever = BM25Retriever.from_defaults(index=index, similarity_top_k=top_k)
dense_retriever = index.as_retriever(similarity_top_k=top_k)

hybrid_retriever = QueryFusionRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    similarity_top_k=top_k,
    num_queries=1,
    mode="reciprocal_rerank",
)
```

BM25 helps especially for `numeric_extraction` queries (unit costs, labour rates, equipment model numbers) and `procedural` queries (exact step labels like "Step 3" or "Phase 2") where semantic embedding underperforms. Evaluate delta in Recall@5 for numeric and procedural types specifically.

### 4.3 Reranker

If MRR is low (first hit often ranked 3rd–5th), add a cross-encoder reranker:

```python
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

reranker = FlagEmbeddingReranker(
    top_n=5,
    model="BAAI/bge-reranker-base",  # or bge-reranker-large
)
```

Reranker operates on the raw text of retrieved chunks, so it's sensitive to chunk size and to whether paragraph coverage metadata is preserved. Evaluate MRR and NDCG@5 before/after.

### 4.4 Error Analysis Protocol

After each experiment run, inspect the bottom 20% performing queries (lowest Recall@5):

```python
def failure_analysis(results: list[dict], passages_db: dict) -> None:
    failures = sorted(results, key=lambda x: x["recall_5"])[:20]
    for f in failures:
        print(f"QID: {f['qid']} | Type: {f['qtype']} | Recall@5: {f['recall_5']}")
        print(f"Question: {f['question']}")
        print(f"Gold required refs: {f['gold_passage_ids']}")
        print(f"Gold partial refs: {f['gold_passage_ids_partial']}")
        print(f"Retrieved refs: {f['retrieved_refs'][:5]}")
        for gold_ref in f["gold_passage_ids"]:
            if not any(retrieved_matches_gold(r, gold_ref) for r in f["retrieved_refs"][:5]):
                pid = gold_ref["passage_id"]
                print(f"MISSED REF: {gold_ref}")
                print(f"MISSED TEXT: {passages_db[pid]['text'][:300]}...")
        print("---")
```

Categorize failures:
- **Vocabulary mismatch**: user asks "repair cost" but document says "corrective maintenance expenditure" → add query rewriting or BM25
- **Passage too short**: cost breakdown split across passages (e.g. labour on p.4, materials on p.5) → merge passages or increase chunk size
- **Passage deeply nested**: answer buried in a cost table or numbered procedure list → improve PDF table parsing
- **Paragraph miss inside correct passage**: retriever found the right passage but not the paragraph containing the answer → improve chunk overlap or paragraph-aware chunking
- **Query too vague**: multi-hop question where embedding retrieves neither passage (e.g. comparing two buildings' boiler costs) → add query decomposition

### 4.5 Decision Framework

```
Start: baseline dense retrieval with current 700-char chunks

Step 1: Run benchmark → if Recall@5 < 0.65:
  → Run chunking sweep (§4.1)
  → Investigate parsing failures (cost tables, scanned inspection reports)

Step 2: Re-run with best chunk config → if Recall@5 < 0.75:
  → Add hybrid retrieval (§4.2)
  → Focus on factual/numeric query types

Step 3: Re-run → if MRR < 0.55 or NDCG@5 < 0.60:
  → Add reranker (§4.3)

Step 4: Re-run → if multi_hop Recall@5 < 0.50:
  → Add query decomposition (break multi-hop Q into sub-queries)
  → Add parent-child retrieval (retrieve child chunk, expand to parent passage)

Step 5: If plateau:
  → add paragraph-aware chunk metadata if missing
  → consider domain-fine-tuned embedding
  (only justified if dataset is large and dense-only recall < 0.70 after all above)
```

---

## Appendix: Inter-Annotator Agreement

For every 50 QA pairs labeled, take 10 and have a second labeler independently label the gold passages. Compute:

```python
from sklearn.metrics import cohen_kappa_score

def passage_level_kappa(labeler_a: list[int], labeler_b: list[int]) -> float:
    """
    labeler_a/b: relevance labels (0/1/2) for the same passages, same queries
    """
    return cohen_kappa_score(labeler_a, labeler_b, weights="linear")
```

Target κ ≥ 0.70. Below 0.60 means the labeling instructions are ambiguous — revise the protocol and re-label the disagreement cases.

When auditing disagreements, inspect not only `passage_id` but also:
- whether both labelers chose the same `section_id`
- whether one labeler added a `paragraph_id` and the other did not
- whether the paragraph-specific label was actually necessary

---

## Summary

| Phase | What You Produce | Key Artifact |
|---|---|---|
| PDF ingestion | Chunk-agnostic passages with `section_id` and `paragraph_ids` | `benchmark/passages.jsonl` |
| QA generation | LLM candidate pairs | `benchmark/labeling/raw_llm_candidates.jsonl` |
| Human review | Verified QA + structured gold passage refs | `benchmark/dev_set.jsonl`, `test_set.jsonl` |
| Retrieval eval | Per-run metric JSON | `evaluation/retrieval_runs/*.json` |
| System tuning | Config grid + failure analysis | Changelog in `CLAUDE.md` or experiment tracker |
