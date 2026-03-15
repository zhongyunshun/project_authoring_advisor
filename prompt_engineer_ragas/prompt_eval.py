"""
prompt_eval.py — Prompt Engineering Evaluation Pipeline for TRCA RAG System.

Consolidates response generation, custom LLM evaluation, RAGAS evaluation,
and result summarization into a single OOP module.

Usage (run from project root):
    # Run all stages
    python prompt_engineer_ragas/prompt_eval.py --stages all

    # Run specific stages
    python prompt_engineer_ragas/prompt_eval.py --stages generate custom_eval

    # Custom patterns and provider
    python prompt_engineer_ragas/prompt_eval.py --patterns rag-only persona+cot+format --stages generate

Stages:
    generate    — Query PromptingRAGEngine for all questions × patterns, save responses
    custom_eval — Use GPT-4o-mini to score responses on 5 quality metrics
    ragas_eval  — Compute RAGAS metrics (faithfulness, relevancy, precision, recall)
    summarize   — Pivot results into per-metric summary CSVs + print overview tables
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

# Allow running directly as: python prompt_engineer_ragas/prompt_eval.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openai import OpenAI

import prompt_engineer_ragas.questions as questions_module
from config.keys import OPENAI_API_KEY
from core.embedding_factory import EmbeddingFactory
from core.llm_factory import LLMFactory
from core.vector_store import VectorStoreManager
from pipeline.prompt_templates import PROMPT_PATTERNS
from pipeline.rag_engine import PromptingRAGEngine

# ─────────────────────────────────────────────────────── constants ───

ALL_PATTERNS: list[str] = list(PROMPT_PATTERNS.keys())

PATTERN_LABELS: dict[str, str] = {
    "gpt-4o-mini":        "Original GPT (No RAG)",
    "rag-only":           "RAG Only",
    "cot+format":         "COT + Format",
    "persona+format":     "Persona + Format",
    "persona+cot":        "Persona + COT",
    "persona+cot+format": "Persona + COT + Format",
}

EVAL_METRICS: list[str] = [
    "Comprehensiveness",
    "Accuracy",
    "Relevance",
    "Clarity and Understandability",
    "Conciseness",
]

STAGES: tuple[str, ...] = ("generate", "custom_eval", "ragas_eval", "summarize")


# ──────────────────────────────────────────────────────── config ───

@dataclass
class EvalConfig:
    """Configuration for a prompt engineering evaluation run."""
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    embedding_provider: str = "openai"
    collection_name: str = "trca_docs"
    storage_path: str = "./vector_db/qdrant_storage"
    patterns: list[str] = field(default_factory=lambda: list(PROMPT_PATTERNS.keys()))
    results_dir: str = "prompt_engineer_ragas/prompting_results"
    prompts_dir: str = "prompt_engineer_ragas/prompts"
    custom_eval_json: str = "prompt_engineer_ragas/custom_eval_results.json"
    ragas_input_csv: str = "prompt_engineer_ragas/ragas_input.csv"
    ragas_output_csv: str = "prompt_engineer_ragas/evaluation_results.csv"
    summary_dir: str = "prompt_engineer_ragas/summary"


# ──────────────────────────────────────────────────── shared helpers ───

def _load_all_questions() -> list[tuple[str, str]]:
    """Return [(var_name, question_text), ...] for all 30 test questions."""
    result = []
    for prefix in ("gm", "hbpe", "pc"):
        for i in range(1, 11):
            var = f"{prefix}_question{i}"
            text = getattr(questions_module, var, None)
            if text:
                result.append((var, text))
    return result


def _label(pattern: str) -> str:
    """Human-readable folder label for a pattern key."""
    return PATTERN_LABELS.get(pattern, pattern)


# ───────────────────────────────────────────── ResponseGenerator ───

class ResponseGenerator:
    """
    Queries PromptingRAGEngine for every question × pattern combination.

    Writes to disk:
        results_dir/<label>/<var_name>.txt  — plain answer text
        prompts_dir/<label>/<var_name>.txt  — query + retrieved contexts (used by RagasEvaluator)
    """

    def __init__(self, config: EvalConfig):
        self._cfg = config

    def generate(self) -> None:
        embed_model = EmbeddingFactory.create(provider=self._cfg.embedding_provider)
        vsm = VectorStoreManager(storage_path=self._cfg.storage_path)
        index = vsm.get_index(self._cfg.collection_name, embed_model)
        llm = LLMFactory.create(provider=self._cfg.llm_provider, model=self._cfg.llm_model)

        questions = _load_all_questions()
        total = len(questions) * len(self._cfg.patterns)
        done = 0

        print(f"\n[Generate] {len(questions)} questions × {len(self._cfg.patterns)} patterns = {total} calls")

        for pattern in self._cfg.patterns:
            engine = PromptingRAGEngine(index=index, llm=llm, pattern=pattern)
            label = _label(pattern)
            ans_dir = Path(self._cfg.results_dir) / label
            prm_dir = Path(self._cfg.prompts_dir) / label
            ans_dir.mkdir(parents=True, exist_ok=True)
            prm_dir.mkdir(parents=True, exist_ok=True)

            for var_name, question in questions:
                done += 1
                print(f"  ({done:>3}/{total}) [{label}] {var_name}", end="\r")
                response = engine.query(question)

                # Answer file
                (ans_dir / f"{var_name}.txt").write_text(response.answer, encoding="utf-8")

                # Prompt log: query + retrieved context blocks (for RAGAS parsing)
                context_log = f"#### {question} ####\n\n"
                for i, src in enumerate(response.sources):
                    context_log += f"--- Document {i + 1} ---\n{src.text[:700]}\n\n"
                (prm_dir / f"{var_name}.txt").write_text(context_log, encoding="utf-8")

        print(f"\n[Generate] Done. Responses saved to '{self._cfg.results_dir}'")


# ──────────────────────────────────────────────── CustomEvaluator ───

_EVAL_PROMPT_TEMPLATE = """\
Act as an expert answer evaluator for a technical RAG system.

You will be given a question and {n} answers generated by different prompting strategies.
Score each answer on the 5 metrics below (0–20 pts each, 100 pts max).

Metrics:
  • Comprehensiveness (0-20): Fully covers all aspects and sub-questions.
  • Accuracy (0-20): Factually correct per TRCA technical documents.
  • Relevance (0-20): Stays on-topic and directly addresses the question.
  • Clarity and Understandability (0-20): Clear, logical, well-structured.
  • Conciseness (0-20): Succinct without unnecessary repetition.

Question: {question}

{answers_section}

Return ONLY valid JSON (no markdown fences) using this exact schema:
{{
  "<pattern_label>": {{
    "Comprehensiveness": <int 0-20>,
    "Accuracy": <int 0-20>,
    "Relevance": <int 0-20>,
    "Clarity and Understandability": <int 0-20>,
    "Conciseness": <int 0-20>,
    "Total": <int 0-100>
  }},
  ...
}}
"""


class CustomEvaluator:
    """
    Evaluates generated responses using GPT-4o-mini as judge.

    Scores each answer on 5 metrics (0-20 each, 100 total).
    Saves structured results to custom_eval_json.
    Prints a formatted comparison table per question.
    """

    def __init__(self, config: EvalConfig):
        self._cfg = config
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        self._client = OpenAI()

    def _load_answers(self) -> dict[str, dict[str, str]]:
        """Returns {var_name: {pattern_key: answer_text}}."""
        grouped: dict[str, dict[str, str]] = {}
        for pattern in self._cfg.patterns:
            ans_dir = Path(self._cfg.results_dir) / _label(pattern)
            if not ans_dir.exists():
                print(f"\n[Eval] Warning: missing dir {ans_dir}")
                continue
            for f in sorted(ans_dir.iterdir()):
                if f.suffix == ".txt":
                    grouped.setdefault(f.stem, {})[pattern] = f.read_text(encoding="utf-8").strip()
        return grouped

    def _call_gpt(self, question: str, answers: dict[str, str]) -> Optional[dict]:
        answers_section = "\n\n".join(
            f"[{_label(p)}]\n{text}" for p, text in answers.items()
        )
        prompt = _EVAL_PROMPT_TEMPLATE.format(
            n=len(answers),
            question=question,
            answers_section=answers_section,
        )
        try:
            resp = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"\n[Eval] API error: {e}")
            return None

    @staticmethod
    def _format_table(var_name: str, question: str, scores: dict) -> str:
        col_w = 30
        short_metrics = [m.split()[0] for m in EVAL_METRICS]  # first word only for header
        header = f"  {'Pattern':<{col_w}}" + "".join(f"{m:>7}" for m in short_metrics) + f"{'Total':>8}"
        sep = "  " + "-" * (col_w + 7 * len(EVAL_METRICS) + 8)
        rows = [header, sep]
        for pat_label, vals in scores.items():
            if not isinstance(vals, dict):
                continue
            nums = [vals.get(m, 0) for m in EVAL_METRICS]
            total = vals.get("Total", sum(nums))
            rows.append(f"  {pat_label:<{col_w}}" + "".join(f"{v:>7}" for v in nums) + f"{total:>8}")
        q_preview = question[:110] + ("..." if len(question) > 110 else "")
        return "\n".join([
            f"\n{'═' * 72}",
            f"  {var_name}",
            f"  {q_preview}",
            f"{'═' * 72}",
        ] + rows + [""])

    def evaluate(self) -> None:
        grouped = self._load_answers()
        q_lookup = dict(_load_all_questions())
        all_results: dict[str, dict] = {}
        total = len(grouped)

        print(f"\n[Eval] Evaluating {total} questions × {len(self._cfg.patterns)} patterns...")

        for i, (var_name, answers) in enumerate(sorted(grouped.items()), 1):
            question = q_lookup.get(var_name, var_name)
            print(f"  ({i:>2}/{total}) {var_name}", end="\r")
            scores = self._call_gpt(question, answers)
            if scores:
                all_results[var_name] = scores
                print(self._format_table(var_name, question, scores))
            else:
                print(f"\n  [!] Evaluation failed for {var_name}")

        Path(self._cfg.custom_eval_json).parent.mkdir(parents=True, exist_ok=True)
        Path(self._cfg.custom_eval_json).write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n[Eval] Results saved to '{self._cfg.custom_eval_json}'")


# ───────────────────────────────────────────────── RagasEvaluator ───

class RagasEvaluator:
    """
    Evaluates responses using RAGAS metrics:
        faithfulness, answer_relevancy, context_precision, context_recall.

    Uses persona+cot+format answers as the ground-truth reference.
    Caches the assembled dataset to ragas_input_csv to avoid repeated file I/O.
    """

    REFERENCE_PATTERN = "persona+cot+format"

    def __init__(self, config: EvalConfig):
        self._cfg = config

    def _parse_prompt_file(self, path: Path) -> tuple[str, list[str]]:
        """Extract (query, context_list) from a saved prompt log file."""
        text = path.read_text(encoding="utf-8")
        q_match = re.search(r"####\s*(.*?)\s*####", text, re.DOTALL)
        query = q_match.group(1).strip() if q_match else ""
        contexts = re.findall(
            r"--- Document \d+ ---\n(.*?)(?=--- Document \d+ ---|\Z)",
            text, re.DOTALL,
        )
        return query, [c.strip() for c in contexts if c.strip()]

    def _build_dataset(self) -> "Dataset":
        from datasets import Dataset

        ref_label = _label(self.REFERENCE_PATTERN)
        ref_dir = Path(self._cfg.results_dir) / ref_label
        ref_answers: dict[str, str] = {
            f.stem: f.read_text(encoding="utf-8").strip()
            for f in ref_dir.iterdir()
            if f.suffix == ".txt"
        } if ref_dir.exists() else {}

        rows = []
        for pattern in self._cfg.patterns:
            if pattern == self.REFERENCE_PATTERN:
                continue
            label = _label(pattern)
            prm_dir = Path(self._cfg.prompts_dir) / label
            ans_dir = Path(self._cfg.results_dir) / label
            if not prm_dir.exists():
                print(f"[RAGAS] Skipping '{label}': prompts dir not found.")
                continue
            for pf in sorted(prm_dir.iterdir()):
                if pf.suffix != ".txt":
                    continue
                query, contexts = self._parse_prompt_file(pf)
                af = ans_dir / pf.name
                answer = af.read_text(encoding="utf-8").strip() if af.exists() else ""
                if query and contexts and answer:
                    rows.append({
                        "prompting_type": label,
                        "question_name": pf.stem,
                        "user_input": query,
                        "retrieved_contexts": contexts,
                        "response": answer,
                        "reference": ref_answers.get(pf.stem, ""),
                    })

        pd.DataFrame(rows).to_csv(self._cfg.ragas_input_csv, index=False)
        print(f"[RAGAS] Dataset: {len(rows)} samples → '{self._cfg.ragas_input_csv}'")
        return Dataset.from_list(rows)

    def evaluate(self) -> pd.DataFrame:
        from langchain_openai import ChatOpenAI
        from ragas.evaluation import RunConfig, evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

        # Re-use cached input CSV if available
        if Path(self._cfg.ragas_input_csv).exists():
            df_in = pd.read_csv(self._cfg.ragas_input_csv)
            if "retrieved_contexts" in df_in.columns:
                df_in["retrieved_contexts"] = df_in["retrieved_contexts"].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
            from datasets import Dataset
            dataset = Dataset.from_pandas(df_in)
            print(f"[RAGAS] Loaded cached dataset ({len(df_in)} samples)")
        else:
            dataset = self._build_dataset()
            df_in = pd.read_csv(self._cfg.ragas_input_csv)

        print("[RAGAS] Running evaluation (this may take several minutes)...")
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
            run_config=RunConfig(timeout=180, max_workers=6),
        )
        df = result.to_pandas()
        for col in ("prompting_type", "question_name"):
            if col not in df.columns and col in df_in.columns:
                df[col] = df_in[col].values

        Path(self._cfg.ragas_output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self._cfg.ragas_output_csv, index=False)
        print(f"[RAGAS] Results saved to '{self._cfg.ragas_output_csv}'")
        return df


# ─────────────────────────────────────────────── ResultSummarizer ───

class ResultSummarizer:
    """
    Aggregates evaluation results into per-metric summary CSVs and prints overview tables.

    Outputs:
        summary/custom/<metric>_summary.csv  — from custom LLM evaluation (JSON)
        summary/ragas/<metric>_summary.csv   — from RAGAS evaluation (CSV)
    """

    def __init__(self, config: EvalConfig):
        self._cfg = config

    def summarize_custom(self) -> None:
        json_path = Path(self._cfg.custom_eval_json)
        if not json_path.exists():
            print(f"[Summary] Custom eval JSON not found: {json_path}")
            return

        data: dict[str, dict] = json.loads(json_path.read_text(encoding="utf-8"))
        records = []
        for var_name, pattern_scores in data.items():
            for pat_label, scores in pattern_scores.items():
                if isinstance(scores, dict):
                    row = {"question": var_name, "pattern": pat_label}
                    row.update({m: scores.get(m, 0) for m in EVAL_METRICS})
                    row["Total"] = scores.get("Total", sum(scores.get(m, 0) for m in EVAL_METRICS))
                    records.append(row)

        if not records:
            print("[Summary] No custom eval records found.")
            return

        df = pd.DataFrame(records)
        out_dir = Path(self._cfg.summary_dir) / "custom"
        out_dir.mkdir(parents=True, exist_ok=True)

        for metric in EVAL_METRICS + ["Total"]:
            if metric not in df.columns:
                continue
            pivot = df.pivot_table(index="pattern", columns="question", values=metric, aggfunc="mean")
            pivot["Average"] = pivot.mean(axis=1)
            safe = metric.lower().replace(" ", "_").replace("(", "").replace(")", "")
            pivot.to_csv(out_dir / f"{safe}_summary.csv")

        print(f"[Summary] Custom eval summaries → '{out_dir}'")
        agg = df.groupby("pattern")[EVAL_METRICS + ["Total"]].mean().round(1)
        print("\n" + "═" * 72)
        print("  Custom Evaluation — Average Scores per Pattern")
        print("═" * 72)
        print(agg.to_string())
        print()

    def summarize_ragas(self) -> None:
        csv_path = Path(self._cfg.ragas_output_csv)
        if not csv_path.exists():
            print(f"[Summary] RAGAS CSV not found: {csv_path}")
            return

        df = pd.read_csv(csv_path)
        skip = {"prompting_type", "question_name", "user_input", "response",
                "retrieved_contexts", "reference", "question", "answer", "contexts", "ground_truth"}
        ragas_metrics = [c for c in df.columns if c not in skip]

        out_dir = Path(self._cfg.summary_dir) / "ragas"
        out_dir.mkdir(parents=True, exist_ok=True)

        for metric in ragas_metrics:
            pivot = df.pivot_table(
                index="prompting_type", columns="question_name",
                values=metric, aggfunc="mean",
            )
            pivot["Average"] = pivot.mean(axis=1)
            pivot.to_csv(out_dir / f"{metric}_summary.csv")

        print(f"[Summary] RAGAS summaries → '{out_dir}'")
        agg = df.groupby("prompting_type")[ragas_metrics].mean().round(3)
        print("\n" + "═" * 72)
        print("  RAGAS Evaluation — Average Scores per Pattern")
        print("═" * 72)
        print(agg.to_string())
        print()


# ──────────────────────────────────────────────────── EvalPipeline ───

class EvalPipeline:
    """Orchestrates the full prompt engineering evaluation workflow."""

    def __init__(self, config: EvalConfig):
        self._cfg = config

    def run(self, stages: list[str]) -> None:
        print(f"\n{'═' * 72}")
        print("  Prompt Engineering Evaluation Pipeline — TRCA RAG System")
        print(f"  Patterns : {', '.join(self._cfg.patterns)}")
        print(f"  Stages   : {', '.join(stages)}")
        print(f"{'═' * 72}")

        if "generate" in stages:
            ResponseGenerator(self._cfg).generate()

        if "custom_eval" in stages:
            CustomEvaluator(self._cfg).evaluate()

        if "ragas_eval" in stages:
            RagasEvaluator(self._cfg).evaluate()

        if "summarize" in stages:
            s = ResultSummarizer(self._cfg)
            s.summarize_custom()
            s.summarize_ragas()

        print(f"\n{'═' * 72}")
        print("  All stages complete.")
        print(f"{'═' * 72}\n")


# ───────────────────────────────────────────────────────── CLI ───

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prompt_eval",
        description="Prompt Engineering Evaluation Pipeline — TRCA RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
stages:
  generate    Query PromptingRAGEngine for all questions × patterns
  custom_eval Use GPT-4o-mini to score each answer on 5 quality metrics
  ragas_eval  Compute RAGAS metrics (faithfulness, answer_relevancy, context_precision, context_recall)
  summarize   Pivot results into per-metric summary CSVs and print overview tables

available patterns:
  {chr(10).join("  " + p for p in ALL_PATTERNS)}

examples:
  python prompt_engineer_ragas/prompt_eval.py --stages all
  python prompt_engineer_ragas/prompt_eval.py --stages generate custom_eval
  python prompt_engineer_ragas/prompt_eval.py --stages summarize
  python prompt_engineer_ragas/prompt_eval.py --patterns rag-only persona+cot+format --stages generate
""",
    )
    parser.add_argument(
        "--stages", nargs="+", default=["all"],
        choices=list(STAGES) + ["all"],
        help="Pipeline stages to run (default: all)",
    )
    parser.add_argument(
        "--patterns", nargs="+", default=None,
        choices=ALL_PATTERNS, metavar="PATTERN",
        help="Prompt patterns to evaluate (default: all 6)",
    )
    parser.add_argument(
        "--llm", default="openai", choices=["openai", "gemini", "claude"],
        help="LLM provider for generation (default: openai)",
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="Model name for the LLM provider (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--embedding", default="openai", choices=["openai", "huggingface"],
        help="Embedding provider — must match the indexed collection (default: openai)",
    )
    parser.add_argument(
        "--collection", default="trca_docs",
        help="Qdrant collection name (default: trca_docs)",
    )
    parser.add_argument(
        "--results_dir", default="prompt_engineer_ragas/prompting_results",
        help="Directory for generated responses",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    stages = list(STAGES) if "all" in args.stages else args.stages
    config = EvalConfig(
        llm_provider=args.llm,
        llm_model=args.model,
        embedding_provider=args.embedding,
        collection_name=args.collection,
        patterns=args.patterns or list(PROMPT_PATTERNS.keys()),
        results_dir=args.results_dir,
    )
    EvalPipeline(config).run(stages)


if __name__ == "__main__":
    main()
