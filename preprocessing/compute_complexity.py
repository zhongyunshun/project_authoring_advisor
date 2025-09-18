
"""
compute_complexity.py

Compute text complexity & coherence metrics from a plain-text corpus.

Metrics:
1) Syntactic complexity
   - Sentence length: mean and 90th percentile tokens/sentence
   - Clause density: mean clauses per sentence (uses spaCy dependency parse if available,
     otherwise a robust heuristic)
   - Parse depth: mean & max dependency-tree depth (spaCy if available, heuristic fallback)

2) Semantic coherence
   - Adjacent-sentence cosine similarity (TF-IDF): mean, 5th, 95th percentiles

3) Lexical sophistication
   - Average IDF of tokens
   - Percent of tokens in top-20% IDF ("low-frequency")
   - Hapax rate: percent of tokens occurring in exactly one sentence (DF == 1)

4) Vector DB count (optional)
   - If --chroma_dir is set: sums counts across all Chroma collections
   - If --faiss_index is set: reads FAISS index.ntotal

Output:
- Prints a compact table to stdout
- Saves JSON/CSV to --outdir (default: ./metrics_out)

Usage examples:
  python compute_metrics_v2.py --text /path/to/output.txt
  python compute_metrics_v2.py --text ./output.txt --chroma_dir /path/to/chroma_db
  python compute_metrics_v2.py --text ./output.txt --faiss_index /path/to/index.faiss

For exact clause density & parse depths, have spaCy + English model installed:
  pip install spacy
  python -m spacy download en_core_web_sm
"""

import os
import re
import json
import math
import argparse
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional spaCy (auto-detected)
def _maybe_load_spacy(disable: Tuple[str,...]=("ner","textcat")):
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm", disable=list(disable))
            if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp
        except Exception:
            return None
    except Exception:
        return None

WORD_RE = re.compile(r"[A-Za-z']+")

def tokenize_words(text: str) -> List[str]:
    return WORD_RE.findall(text)

# Rule-based sentence splitter as fallback
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

def split_into_sentences_rb(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    chunks = SENT_SPLIT_RE.split(text)
    sents = []
    for c in chunks:
        c = c.strip()
        if not c:
            continue
        parts = [sc.strip() for sc in re.split(r"\n{2,}", c) if sc.strip()]
        sents.extend(parts if parts else [c])
    return sents

# Clause density helpers
SUBORDINATORS = set("""after although as because before even if even though if in order that once provided that rather than since so that than that though unless until when whenever where whereas wherever whether while which who whom whose that""".split())
COORD_CONJ = set("and but or nor for yet so".split())

def heuristic_clause_count(sentence: str) -> int:
    s = sentence.strip()
    if not s:
        return 0
    marks = s.count(",") + s.count(";") + s.count(":")
    tokens_lower = [t.lower() for t in tokenize_words(s)]
    markers = sum(1 for t in tokens_lower if t in SUBORDINATORS or t in COORD_CONJ)
    est = 1 + 0.6 * (marks + markers * 0.8)  # conservative
    return max(1, int(round(est)))

def spacy_clause_count(doc) -> int:
    # Count clause-like dependents + root
    clause_like = {"ccomp","xcomp","advcl","acl","relcl","conj","parataxis"}
    count = 1
    for token in doc:
        if token.dep_ in clause_like:
            count += 1
    return count

def compute_dependency_depth(doc) -> int:
    # Max distance in tokens' head-chains to root
    max_depth = 0
    for token in doc:
        depth = 0
        cur = token
        while cur.head is not cur:
            depth += 1
            cur = cur.head
            if depth > 100:  # safety
                break
        max_depth = max(max_depth, depth)
    return max_depth

def percentile(values: List[float], pct: float) -> float:
    if not values:
        return float("nan")
    arr = np.array(values, dtype=float)
    return float(np.percentile(arr, pct))

def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def count_chroma(chroma_dir: str):
    try:
        import chromadb
        client = chromadb.PersistentClient(path=chroma_dir)
        colls = client.list_collections()
        total = 0
        for c in colls:
            try:
                total += c.count()
            except Exception:
                pass
        return True, "chroma", int(total), "Sum of all collection counts"
    except Exception as e:
        return False, None, None, f"Chroma error: {e}"

def count_faiss(index_path: str):
    try:
        import faiss
        index = faiss.read_index(index_path)
        return True, "faiss", int(index.ntotal), ""
    except Exception as e:
        return False, None, None, f"FAISS error: {e}"

def main():
    ap = argparse.ArgumentParser(description="Compute text complexity & coherence metrics.")
    ap.add_argument("--text", required=True, help="Path to corpus .txt (one big text file)." )
    ap.add_argument("--outdir", default="metrics_out", help="Where to save CSV/JSON outputs.")
    ap.add_argument("--force_heuristic", action="store_true",
                    help="Force heuristic mode (skip spaCy even if available)." )
    ap.add_argument("--chroma_dir", default=None, help="Path to Chroma persist directory (optional)." )
    ap.add_argument("--faiss_index", default=None, help="Path to FAISS index file (optional)." )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    raw = load_text(args.text)
    # Try spaCy unless forced off
    nlp = None if args.force_heuristic else _maybe_load_spacy()

    # Sentences
    if nlp is not None:
        doc = nlp(raw)
        sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    else:
        sentences = split_into_sentences_rb(raw)

    # 1) Syntactic complexity
    sent_lengths = [len(tokenize_words(s)) for s in sentences] if sentences else []
    mean_sent_len = float(np.mean(sent_lengths)) if sent_lengths else float("nan")
    p90_sent_len  = percentile(sent_lengths, 90.0)

    clause_counts = []
    max_depths = []

    if nlp is not None:
        for s in sentences:
            parsed = nlp(s)
            clause_counts.append(spacy_clause_count(parsed))
            max_depths.append(compute_dependency_depth(parsed))
    else:
        for s in sentences:
            clause_counts.append(heuristic_clause_count(s))
            depth = 1 + s.count("(") + s.count("[") + s.count("{") + s.count(",")//2
            depth += sum(1 for t in tokenize_words(s) if t.lower() in SUBORDINATORS) // 2
            max_depths.append(depth)

    mean_clause_density = float(np.mean([c for c in clause_counts if c > 0])) if clause_counts else float("nan")
    mean_parse_depth    = float(np.mean(max_depths)) if max_depths else float("nan")
    max_parse_depth     = float(np.max(max_depths)) if max_depths else float("nan")

    # 2) Semantic coherence (TF-IDF) & lexical sophistication
    mean_adjacent_coh = float("nan")
    pct05_adjacent_coh = float("nan")
    pct95_adjacent_coh = float("nan")

    if len(sentences) >= 2:
        vectorizer = TfidfVectorizer(token_pattern=r"[A-Za-z']{2,}")
        X = vectorizer.fit_transform(sentences)
        sims = []
        for i in range(len(sentences)-1):
            sims.append(float(cosine_similarity(X[i], X[i+1])[0,0]))
        if sims:
            mean_adjacent_coh = float(np.mean(sims))
            pct05_adjacent_coh = percentile(sims, 5.0)
            pct95_adjacent_coh = percentile(sims, 95.0)

    avg_idf = float("nan")
    pct_low_freq_top20idf = float("nan")
    hapax_rate = float("nan")

    if sentences:
        sent_vectorizer = TfidfVectorizer(token_pattern=r"[A-Za-z']{2,}")
        Xs = sent_vectorizer.fit_transform(sentences)
        features = sent_vectorizer.get_feature_names_out()
        idf = sent_vectorizer.idf_
        idf_map = {features[i]: float(idf[i]) for i in range(len(features))}

        tokens = [t.lower() for t in tokenize_words(raw)]
        token_idfs = [idf_map[t] for t in tokens if t in idf_map]
        if token_idfs:
            avg_idf = float(np.mean(token_idfs))
            threshold = float(np.quantile(list(idf_map.values()), 0.80))
            lowfreq_tokens = [t for t in tokens if t in idf_map and idf_map[t] >= threshold]
            pct_low_freq_top20idf = 100.0 * len(lowfreq_tokens) / max(1, len(tokens))

        import numpy as _np
        df_counts = _np.asarray((Xs > 0).sum(axis=0)).ravel()
        hapax_features = set(features[_np.where(df_counts == 1)[0]])
        corpus_words = [t.lower() for t in tokens if re.match(r"[A-Za-z']{2,}$", t)]
        hapax_tokens = [t for t in corpus_words if t in hapax_features]
        hapax_rate = 100.0 * len(hapax_tokens) / max(1, len(corpus_words))

    # 3) Vector DB counts
    db_found = False
    db_type = None
    db_path = None
    db_count = None
    db_note = ""

    if args.chroma_dir:
        ok, typ, cnt, note = count_chroma(args.chroma_dir)
        db_found, db_type, db_path, db_count, db_note = ok, typ, args.chroma_dir, cnt, note

    if (not db_found) and args.faiss_index:
        ok, typ, cnt, note = count_faiss(args.faiss_index)
        db_found, db_type, db_path, db_count, db_note = ok, typ, args.faiss_index, cnt, note

    # Aggregate
    metrics = {
        "corpus_source": os.path.abspath(args.text),
        "num_sentences": int(len(sentences)),
        "sentence_length_mean_tokens": round(mean_sent_len, 3) if not math.isnan(mean_sent_len) else None,
        "sentence_length_p90_tokens": round(p90_sent_len, 3) if not math.isnan(p90_sent_len) else None,
        "clause_density_mean_clauses_per_sentence": round(mean_clause_density, 3) if not math.isnan(mean_clause_density) else None,
        "parse_depth_mean": round(mean_parse_depth, 3) if not math.isnan(mean_parse_depth) else None,
        "parse_depth_max": round(max_parse_depth, 3) if not math.isnan(max_parse_depth) else None,
        "semantic_coherence_mean_adjacent_cosine": round(mean_adjacent_coh, 4) if not math.isnan(mean_adjacent_coh) else None,
        "semantic_coherence_p05_adjacent_cosine": round(pct05_adjacent_coh, 4) if not math.isnan(pct05_adjacent_coh) else None,
        "semantic_coherence_p95_adjacent_cosine": round(pct95_adjacent_coh, 4) if not math.isnan(pct95_adjacent_coh) else None,
        "lexical_avg_idf": round(avg_idf, 4) if not math.isnan(avg_idf) else None,
        "lexical_pct_low_freq_top20idf": round(pct_low_freq_top20idf, 2) if not math.isnan(pct_low_freq_top20idf) else None,
        "lexical_hapax_rate_pct": round(hapax_rate, 2) if not math.isnan(hapax_rate) else None,
        "vector_db_found": bool(db_found),
        "vector_db_type": db_type,
        "vector_db_path": db_path,
        "vector_db_count": db_count,
        "vector_db_note": db_note,
        "spaCy_used": bool(nlp is not None),
        "fallback_heuristics_used": bool(nlp is None),
    }

    # Save
    df = pd.DataFrame([metrics])
    csv_path = os.path.join(args.outdir, "complexity_metrics.csv")
    json_path = os.path.join(args.outdir, "complexity_metrics.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Pretty print
    def _fmt(k, v):
        return f"{k:45s} : {v}"
    print("\n=== Complexity & Coherence Metrics ===")
    for k in [
        "corpus_source","num_sentences",
        "sentence_length_mean_tokens","sentence_length_p90_tokens",
        "clause_density_mean_clauses_per_sentence",
        "parse_depth_mean","parse_depth_max",
        "semantic_coherence_mean_adjacent_cosine",
        "semantic_coherence_p05_adjacent_cosine",
        "semantic_coherence_p95_adjacent_cosine",
        "lexical_avg_idf","lexical_pct_low_freq_top20idf","lexical_hapax_rate_pct",
        "vector_db_found","vector_db_type","vector_db_path","vector_db_count","vector_db_note",
        "spaCy_used","fallback_heuristics_used",
    ]:
        print(_fmt(k, metrics[k]))
    print(f"\nSaved CSV to: {csv_path}")
    print(f"Saved JSON to: {json_path}")

if __name__ == "__main__":
    main()
