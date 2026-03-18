"""
factual_qa_eval.py — Factual QA Evaluation Pipeline for TRCA RAG System.

Evaluates RAG outputs using BLEU/ROUGE (lexical) and RAGAS (semantic) metrics
over a folder of CSV result files.

Input CSV columns:
    question          → user_input  (renamed on load)
    answer            → reference   (renamed on load)
    generated_answer  → response    (renamed on load)
    retrieved_contexts              (list of context strings)
    alternative_answer              (optional, used as extra reference for BLEU/ROUGE)

Usage (run from project root):
    # BLEU + ROUGE only
    python evaluation/generation_quality/200_factual_qa_eval/factual_qa_eval.py --mode bleu_rouge

    # RAGAS only
    python evaluation/generation_quality/200_factual_qa_eval/factual_qa_eval.py --mode ragas

    # Both
    python evaluation/generation_quality/200_factual_qa_eval/factual_qa_eval.py --mode both

    # Custom paths
    python evaluation/generation_quality/200_factual_qa_eval/factual_qa_eval.py \\
        --mode both \\
        --input_folder QA_pair/qa_pair_200_0210/output \\
        --output_csv evaluation/generation_quality/200_factual_qa_eval/results.csv
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path

import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from config.settings import Settings

_W = 80  # output width


# ──────────────────────────────────────────────────────── BLEU ───

def calculate_bleu(csv_file: str, n: int = 3) -> tuple[float, list[float]]:
    """
    Calculate BLEU-n score between reference and generated answers.

    Args:
        csv_file: Path to CSV with 'answer' and 'generated_answer' columns.
        n: Maximum n-gram order (1–4).

    Returns:
        (avg_bleu, per_row_scores)
    """
    df = pd.read_csv(csv_file, dtype=str)
    if "answer" not in df.columns or "generated_answer" not in df.columns:
        raise ValueError("CSV must contain 'answer' and 'generated_answer' columns.")

    smoothie = SmoothingFunction().method1
    weights = tuple(1 / n for _ in range(n)) + (0,) * (4 - n)
    scores = []

    for _, row in df.iterrows():
        refs = [str(row["answer"]).strip().lower() if pd.notna(row["answer"]) else ""]
        if "alternative_answer" in df.columns and pd.notna(row.get("alternative_answer")):
            refs.append(str(row["alternative_answer"]).strip().lower())
        references = [r.split() for r in refs if r]

        hyp = str(row["generated_answer"]).strip().lower() if pd.notna(row["generated_answer"]) else ""
        hyp_tokens = hyp.split() if hyp else []

        scores.append(
            sentence_bleu(references, hyp_tokens, weights=weights, smoothing_function=smoothie)
            if references and hyp_tokens else 0.0
        )

    avg = sum(scores) / len(scores) if scores else 0.0
    return avg, scores


# ─────────────────────────────────────────────────────── ROUGE ───

def calculate_rouge(csv_file: str) -> dict[str, float]:
    """
    Calculate ROUGE-1/2/3 F-measure between reference and generated answers.

    Args:
        csv_file: Path to CSV with 'answer' and 'generated_answer' columns.

    Returns:
        {'ROUGE-1': float, 'ROUGE-2': float, 'ROUGE-3': float}
    """
    df = pd.read_csv(csv_file, dtype=str)
    if "answer" not in df.columns or "generated_answer" not in df.columns:
        raise ValueError("CSV must contain 'answer' and 'generated_answer' columns.")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rouge3"], use_stemmer=True)
    r1, r2, r3 = [], [], []

    for _, row in df.iterrows():
        refs = [str(row["answer"]).strip().lower() if pd.notna(row["answer"]) else ""]
        if "alternative_answer" in df.columns and pd.notna(row.get("alternative_answer")):
            refs.append(str(row["alternative_answer"]).strip().lower())

        hyp = str(row["generated_answer"]).strip().lower() if pd.notna(row["generated_answer"]) else ""

        if refs and hyp:
            all_scores = [scorer.score(ref, hyp) for ref in refs]
            r1.append(max(s["rouge1"].fmeasure for s in all_scores))
            r2.append(max(s["rouge2"].fmeasure for s in all_scores))
            r3.append(max(s["rouge3"].fmeasure for s in all_scores))
        else:
            r1.append(0.0); r2.append(0.0); r3.append(0.0)

    return {
        "ROUGE-1": sum(r1) / len(r1) if r1 else 0.0,
        "ROUGE-2": sum(r2) / len(r2) if r2 else 0.0,
        "ROUGE-3": sum(r3) / len(r3) if r3 else 0.0,
    }


# ──────────────────────────────────────────────── BLEU/ROUGE folder ───

def evaluate_folder_bleu_rouge(input_folder: str, output_csv: str) -> pd.DataFrame:
    """
    Run BLEU + ROUGE evaluation over all CSV files in a folder.

    Filenames must follow the pattern: output_chunk<N>_top<K>_<search_type>.csv
    Results are saved to output_csv and also returned as a DataFrame.
    """
    print(f"\n{'═' * _W}")
    print("  BLEU / ROUGE Evaluation")
    print(f"  Input  : {input_folder}")
    print(f"  Output : {output_csv}")
    print(f"{'═' * _W}\n")

    records = []
    csv_files = sorted(Path(input_folder).glob("*.csv"))
    if not csv_files:
        print(f"  [!] No CSV files found in '{input_folder}'")
        return pd.DataFrame()

    for i, path in enumerate(csv_files, 1):
        parts = path.stem.replace("output_", "").split("_")
        chunk = parts[0].replace("chunk", "") if len(parts) > 0 else ""
        top_k = parts[1].replace("top", "") if len(parts) > 1 else ""
        search = parts[2] if len(parts) > 2 else ""

        print(f"  ({i:>2}/{len(csv_files)}) {path.name}", end="\r", flush=True)

        b1, _ = calculate_bleu(str(path), n=1)
        b2, _ = calculate_bleu(str(path), n=2)
        b3, _ = calculate_bleu(str(path), n=3)
        bleu_avg = (b1 + b2 + b3) / 3

        rouge = calculate_rouge(str(path))
        rouge_avg = (rouge["ROUGE-1"] + rouge["ROUGE-2"] + rouge["ROUGE-3"]) / 3

        records.append({
            "Filename":     path.name,
            "Chunk Length": chunk,
            "Top K":        top_k,
            "Search Type":  search,
            "BLEU-1":       round(b1, 4),
            "BLEU-2":       round(b2, 4),
            "BLEU-3":       round(b3, 4),
            "BLEU Avg":     round(bleu_avg, 4),
            "ROUGE-1":      round(rouge["ROUGE-1"], 4),
            "ROUGE-2":      round(rouge["ROUGE-2"], 4),
            "ROUGE-3":      round(rouge["ROUGE-3"], 4),
            "ROUGE Avg":    round(rouge_avg, 4),
            "Total Avg":    round((bleu_avg + rouge_avg) / 2, 4),
        })

    df = pd.DataFrame(records)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    # Summary table
    col_w = 30
    metrics = ["BLEU Avg", "ROUGE-1", "ROUGE-2", "ROUGE-3", "Total Avg"]
    print(f"\n  {'─' * (_W - 2)}")
    print(f"  {'Filename':<{col_w}}" + "".join(f"{m:>12}" for m in metrics))
    print(f"  {'─' * (_W - 2)}")
    for _, row in df.iterrows():
        print(f"  {row['Filename']:<{col_w}}" + "".join(f"{row[m]:>12.4f}" for m in metrics))
    print(f"\n  Results saved → '{output_csv}'\n")
    return df


# ──────────────────────────────────────────────────── RAGAS folder ───

def _load_and_prepare(csv_file: str) -> pd.DataFrame:
    """Load a QA CSV and rename columns to RAGAS-expected names."""
    df = pd.read_csv(csv_file)
    df.rename(columns={
        "question":        "user_input",
        "answer":          "reference",
        "generated_answer": "response",
    }, inplace=True)
    if "retrieved_contexts" in df.columns:
        df["retrieved_contexts"] = df["retrieved_contexts"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
    required = {"user_input", "retrieved_contexts", "response", "reference"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df


def evaluate_single_ragas(csv_file: str, ragas_llm, ragas_embeddings) -> dict[str, float] | None:
    """
    Run RAGAS evaluation on a single CSV file.

    Args:
        csv_file: Path to the QA result CSV.
        ragas_llm: LlamaIndexLLMWrapper instance.
        ragas_embeddings: LlamaIndexEmbeddingsWrapper instance.

    Returns:
        Dict of metric scores, or None on failure.
    """
    from datasets import Dataset
    from ragas.evaluation import RunConfig, evaluate
    from ragas.metrics import (
        answer_relevancy, context_precision, context_recall, faithfulness,
    )

    df = _load_and_prepare(csv_file)
    dataset = Dataset.from_pandas(df)

    try:
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=RunConfig(timeout=180, max_workers=6),
        )
        return dict(result.scores)
    except Exception as e:
        print(f"\n  [!] RAGAS evaluation failed for {csv_file}: {e}")
        return None


def evaluate_folder_ragas(input_folder: str, output_csv: str) -> pd.DataFrame:
    """
    Run RAGAS evaluation over all CSV files in a folder using the LlamaIndex backend.

    Results are saved to output_csv and also returned as a DataFrame.
    """
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI as LlamaOpenAI
    from ragas.embeddings import LlamaIndexEmbeddingsWrapper
    from ragas.llms import LlamaIndexLLMWrapper

    print(f"\n{'═' * _W}")
    print("  RAGAS Evaluation  (LlamaIndex backend — gpt-4o-mini)")
    print(f"  Input  : {input_folder}")
    print(f"  Output : {output_csv}")
    print(f"{'═' * _W}\n")

    ragas_llm = LlamaIndexLLMWrapper(LlamaOpenAI(model="gpt-4o-mini"))
    ragas_embeddings = LlamaIndexEmbeddingsWrapper(
        OpenAIEmbedding(model="text-embedding-3-small")
    )

    records = []
    csv_files = sorted(Path(input_folder).glob("*.csv"))
    if not csv_files:
        print(f"  [!] No CSV files found in '{input_folder}'")
        return pd.DataFrame()

    for i, path in enumerate(csv_files, 1):
        parts = path.stem.replace("output_", "").split("_")
        chunk = parts[0].replace("chunk", "") if len(parts) > 0 else ""
        top_k = parts[1].replace("top", "") if len(parts) > 1 else ""
        search = parts[2] if len(parts) > 2 else ""

        print(f"  ({i:>2}/{len(csv_files)}) {path.name} ...", flush=True)
        scores = evaluate_single_ragas(str(path), ragas_llm, ragas_embeddings)
        if scores is None:
            continue

        records.append({
            "Filename":          path.name,
            "Chunk Length":      chunk,
            "Top K":             top_k,
            "Search Type":       search,
            "Faithfulness":      round(scores.get("faithfulness", 0), 4),
            "Answer Relevancy":  round(scores.get("answer_relevancy", 0), 4),
            "Context Precision": round(scores.get("context_precision", 0), 4),
            "Context Recall":    round(scores.get("context_recall", 0), 4),
        })

    df = pd.DataFrame(records)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    # Summary table
    col_w = 30
    metrics = ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"]
    print(f"\n  {'─' * (_W - 2)}")
    print(f"  {'Filename':<{col_w}}" + "".join(f"{m:>18}" for m in metrics))
    print(f"  {'─' * (_W - 2)}")
    for _, row in df.iterrows():
        print(f"  {row['Filename']:<{col_w}}" + "".join(f"{row[m]:>18.4f}" for m in metrics))
    print(f"\n  Results saved → '{output_csv}'\n")
    return df


# ───────────────────────────────────────────────────────── CLI ───

def main() -> None:
    Settings.from_env().apply_env()

    parser = argparse.ArgumentParser(
        prog="factual_qa_eval",
        description="Factual QA Evaluation Pipeline — TRCA RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
modes:
  bleu_rouge  Lexical evaluation using BLEU-1/2/3 and ROUGE-1/2/3
  ragas       Semantic evaluation using RAGAS (faithfulness, relevancy, precision, recall)
  both        Run both modes and save separate output CSVs

examples:
  python evaluation/generation_quality/200_factual_qa_eval/factual_qa_eval.py --mode bleu_rouge
  python evaluation/generation_quality/200_factual_qa_eval/factual_qa_eval.py --mode ragas
  python evaluation/generation_quality/200_factual_qa_eval/factual_qa_eval.py --mode both \\
      --input_folder QA_pair/qa_pair_200_0210/output \\
      --output_csv evaluation/generation_quality/200_factual_qa_eval/results.csv
""",
    )
    parser.add_argument(
        "--mode", default="both", choices=["bleu_rouge", "ragas", "both"],
        help="Evaluation mode (default: both)",
    )
    parser.add_argument(
        "--input_folder", default="QA_pair/qa_pair_200_0210/output",
        help="Folder containing CSV result files",
    )
    parser.add_argument(
        "--output_csv",
        default="evaluation/generation_quality/200_factual_qa_eval/eval_results.csv",
        help="Output CSV path (suffix _bleu_rouge / _ragas appended when mode=both)",
    )
    args = parser.parse_args()

    out = Path(args.output_csv)

    if args.mode in ("bleu_rouge", "both"):
        out_br = out.with_name(out.stem + "_bleu_rouge.csv") if args.mode == "both" else out
        evaluate_folder_bleu_rouge(args.input_folder, str(out_br))

    if args.mode in ("ragas", "both"):
        out_rg = out.with_name(out.stem + "_ragas.csv") if args.mode == "both" else out
        evaluate_folder_ragas(args.input_folder, str(out_rg))


if __name__ == "__main__":
    main()
