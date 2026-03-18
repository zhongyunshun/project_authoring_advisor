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

from config.settings import Settings
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

_W = 80  # standard output width


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

_QUESTIONS_CSV = Path(__file__).parent / "thrity_open_ended_questions.csv"


def _load_all_questions() -> list[tuple[str, str]]:
    """Return [(var_name, question_text), ...] loaded from thrity_open_ended_questions.csv."""
    df = pd.read_csv(_QUESTIONS_CSV)
    return list(zip(df["var_name"], df["question"]))


def _label(pattern: str) -> str:
    """Human-readable folder label for a pattern key."""
    return PATTERN_LABELS.get(pattern, pattern)


def _banner(title: str, char: str = "═") -> str:
    """Return a full-width banner line with centered title."""
    return f"\n{char * _W}\n  {title}\n{char * _W}"


def _section(title: str) -> str:
    """Return a lighter section divider."""
    return f"\n  ── {title} {'─' * (_W - len(title) - 6)}"


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
        print(_banner("Stage 1 / 4 — Response Generation"))

        embed_model = EmbeddingFactory.create(provider=self._cfg.embedding_provider)
        vsm = VectorStoreManager(storage_path=self._cfg.storage_path)
        index = vsm.get_index(self._cfg.collection_name, embed_model)
        llm = LLMFactory.create(provider=self._cfg.llm_provider, model=self._cfg.llm_model)

        questions = _load_all_questions()
        n_q = len(questions)
        n_p = len(self._cfg.patterns)
        total = n_q * n_p
        done = 0

        print(f"\n  Questions : {n_q}")
        print(f"  Patterns  : {n_p}  ({', '.join(self._cfg.patterns)})")
        print(f"  Total LLM calls : {total}\n")

        for p_idx, pattern in enumerate(self._cfg.patterns, 1):
            engine = PromptingRAGEngine(index=index, llm=llm, pattern=pattern)
            label = _label(pattern)
            ans_dir = Path(self._cfg.results_dir) / label
            prm_dir = Path(self._cfg.prompts_dir) / label
            ans_dir.mkdir(parents=True, exist_ok=True)
            prm_dir.mkdir(parents=True, exist_ok=True)

            print(f"  [{p_idx}/{n_p}] Pattern: {label}")
            for q_idx, (var_name, question) in enumerate(questions, 1):
                done += 1
                print(f"        ({done:>3}/{total}) {var_name} ...", end="\r", flush=True)
                response = engine.query(question)

                (ans_dir / f"{var_name}.txt").write_text(response.answer, encoding="utf-8")

                context_log = f"#### {question} ####\n\n"
                for i, src in enumerate(response.sources):
                    context_log += f"--- Document {i + 1} ---\n{src.text[:700]}\n\n"
                (prm_dir / f"{var_name}.txt").write_text(context_log, encoding="utf-8")

            print(f"        {n_q}/{n_q} questions done.{' ' * 20}")

        print(f"\n  Responses saved → '{self._cfg.results_dir}'")
        print(f"  Prompts   saved → '{self._cfg.prompts_dir}'")


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
        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

    def _load_answers(self) -> dict[str, dict[str, str]]:
        """Returns {var_name: {pattern_key: answer_text}}."""
        grouped: dict[str, dict[str, str]] = {}
        for pattern in self._cfg.patterns:
            ans_dir = Path(self._cfg.results_dir) / _label(pattern)
            if not ans_dir.exists():
                print(f"\n  [!] Warning: missing results dir — {ans_dir}")
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
            from openai.types.chat import ChatCompletionUserMessageParam
            from openai.types.shared_params import ResponseFormatJSONObject
            resp = self._client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[ChatCompletionUserMessageParam(role="user", content=prompt)],
                temperature=0.0,
                response_format=ResponseFormatJSONObject(type="json_object"),
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            print(f"\n  [!] API error: {e}")
            return None

    @staticmethod
    def _format_table(var_name: str, question: str, scores: dict) -> str:
        """Render a per-question score table."""
        col_w = 28
        metric_shorts = ["Compreh.", "Accuracy", "Relevance", "Clarity", "Concise"]
        hdr = f"  {'Pattern':<{col_w}}" + "".join(f"{m:>10}" for m in metric_shorts) + f"{'Total':>8}"
        sep = "  " + "-" * (col_w + 10 * len(EVAL_METRICS) + 8)
        rows = [hdr, sep]
        for pat_label, vals in scores.items():
            if not isinstance(vals, dict):
                continue
            nums = [vals.get(m, 0) for m in EVAL_METRICS]
            total = vals.get("Total", sum(nums))
            rows.append(
                f"  {pat_label:<{col_w}}" + "".join(f"{v:>10}" for v in nums) + f"{total:>8}"
            )
        q_preview = question[:_W - 4] + ("…" if len(question) > _W - 4 else "")
        return "\n".join([
            f"\n  {'─' * (_W - 2)}",
            f"  {var_name}",
            f"  Q: {q_preview}",
            f"  {'─' * (_W - 2)}",
        ] + rows + [""])

    def evaluate(self) -> None:
        print(_banner("Stage 2 / 4 — Custom LLM Evaluation (GPT-4o-mini judge)"))

        grouped = self._load_answers()
        q_lookup = dict(_load_all_questions())
        all_results: dict[str, dict] = {}
        total = len(grouped)

        print(f"\n  Questions to evaluate : {total}")
        print(f"  Patterns per question : {len(self._cfg.patterns)}\n")

        for i, (var_name, answers) in enumerate(sorted(grouped.items()), 1):
            question = q_lookup.get(var_name, var_name)
            print(f"  Evaluating ({i:>2}/{total}) {var_name} ...", end="\r", flush=True)
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
        print(f"\n  Results saved → '{self._cfg.custom_eval_json}'")


# ───────────────────────────────────────────────── RagasEvaluator ───

class RagasEvaluator:
    """
    Evaluates responses using RAGAS metrics via LlamaIndex LLM/embedding wrappers:
        faithfulness, answer_relevancy, context_precision, context_recall.

    Uses persona+cot+format answers as the ground-truth reference.
    Caches the assembled dataset to ragas_input_csv to avoid repeated file I/O.
    """

    REFERENCE_PATTERN = "persona+cot+format"

    def __init__(self, config: EvalConfig):
        self._cfg = config

    @staticmethod
    def _parse_prompt_file(path: Path) -> tuple[str, list[str]]:
        """Extract (query, context_list) from a saved prompt log file."""
        text = path.read_text(encoding="utf-8")
        q_match = re.search(r"####\s*(.*?)\s*####", text, re.DOTALL)
        query = q_match.group(1).strip() if q_match else ""
        contexts = re.findall(
            r"--- Document \d+ ---\n(.*?)(?=--- Document \d+ ---|\Z)",
            text, re.DOTALL,
        )
        return query, [c.strip() for c in contexts if c.strip()]

    def _build_dataset(self):
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
                print(f"  [!] Skipping '{label}': prompts dir not found.")
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
        print(f"  Dataset: {len(rows)} samples → '{self._cfg.ragas_input_csv}'")
        return Dataset.from_list(rows)

    def evaluate(self) -> pd.DataFrame:
        from datasets import Dataset
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI as LlamaOpenAI
        from ragas.embeddings import LlamaIndexEmbeddingsWrapper
        from ragas.evaluation import RunConfig, evaluate
        from ragas.llms import LlamaIndexLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        print(_banner("Stage 3 / 4 — RAGAS Evaluation (LlamaIndex backend)"))

        ragas_llm = LlamaIndexLLMWrapper(LlamaOpenAI(model="gpt-4o-mini"))
        ragas_embeddings = LlamaIndexEmbeddingsWrapper(
            OpenAIEmbedding(model="text-embedding-3-small")
        )

        # Re-use cached input CSV if available
        if Path(self._cfg.ragas_input_csv).exists():
            df_in = pd.read_csv(self._cfg.ragas_input_csv)
            if "retrieved_contexts" in df_in.columns:
                df_in["retrieved_contexts"] = df_in["retrieved_contexts"].apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
            dataset = Dataset.from_pandas(df_in)
            print(f"\n  Loaded cached dataset — {len(df_in)} samples")
        else:
            print("\n  Building dataset from generated responses...")
            dataset = self._build_dataset()
            df_in = pd.read_csv(self._cfg.ragas_input_csv)

        print("  Running RAGAS metrics (this may take several minutes)...")
        print(f"  Metrics  : faithfulness, answer_relevancy, context_precision, context_recall")
        print(f"  LLM      : gpt-4o-mini  (LlamaIndex wrapper)")
        print(f"  Embedding: text-embedding-3-small  (LlamaIndex wrapper)\n")

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=RunConfig(timeout=180, max_workers=6),
        )
        df = result.to_pandas()
        for col in ("prompting_type", "question_name"):
            if col not in df.columns and col in df_in.columns:
                df[col] = df_in[col].values

        Path(self._cfg.ragas_output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(self._cfg.ragas_output_csv, index=False)
        print(f"  Results saved → '{self._cfg.ragas_output_csv}'")
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
        print(_section("Custom Evaluation Summary"))

        json_path = Path(self._cfg.custom_eval_json)
        if not json_path.exists():
            print(f"\n  [!] Custom eval JSON not found: {json_path}")
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
            print("\n  [!] No custom eval records found.")
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

        agg = df.groupby("pattern")[EVAL_METRICS + ["Total"]].mean().round(1)
        agg = agg.sort_values("Total", ascending=False)

        col_w = 28
        metric_shorts = ["Compreh.", "Accuracy", "Relevance", "Clarity", "Concise", "Total"]
        hdr = f"\n  {'Pattern':<{col_w}}" + "".join(f"{m:>10}" for m in metric_shorts)
        sep = "  " + "-" * (col_w + 10 * len(metric_shorts))
        print(hdr)
        print(sep)
        for pat, row in agg.iterrows():
            vals = [row.get(m, 0) for m in EVAL_METRICS] + [row.get("Total", 0)]
            print(f"  {pat:<{col_w}}" + "".join(f"{v:>10.1f}" for v in vals))
        print(f"\n  Per-metric CSVs → '{out_dir}'")

    def summarize_ragas(self) -> None:
        print(_section("RAGAS Evaluation Summary"))

        csv_path = Path(self._cfg.ragas_output_csv)
        if not csv_path.exists():
            print(f"\n  [!] RAGAS CSV not found: {csv_path}")
            return

        df = pd.read_csv(csv_path)
        skip = {
            "prompting_type", "question_name", "user_input", "response",
            "retrieved_contexts", "reference", "question", "answer",
            "contexts", "ground_truth",
        }
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

        agg = df.groupby("prompting_type")[ragas_metrics].mean().round(3)
        agg = agg.sort_values(ragas_metrics[0], ascending=False)

        col_w = 28
        hdr = f"\n  {'Pattern':<{col_w}}" + "".join(f"{m:>18}" for m in ragas_metrics)
        sep = "  " + "-" * (col_w + 18 * len(ragas_metrics))
        print(hdr)
        print(sep)
        for pat, row in agg.iterrows():
            vals = [row.get(m, 0) for m in ragas_metrics]
            print(f"  {pat:<{col_w}}" + "".join(f"{v:>18.3f}" for v in vals))
        print(f"\n  Per-metric CSVs → '{out_dir}'")


# ──────────────────────────────────────────────────── EvalPipeline ───

class EvalPipeline:
    """Orchestrates the full prompt engineering evaluation workflow."""

    def __init__(self, config: EvalConfig):
        self._cfg = config

    def run(self, stages: list[str]) -> None:
        # Load .env and push all API keys into os.environ before any stage runs
        Settings.from_env().apply_env()

        print(_banner("Prompt Engineering Evaluation Pipeline — TRCA RAG System"))
        print(f"\n  LLM       : {self._cfg.llm_provider} / {self._cfg.llm_model}")
        print(f"  Embedding : {self._cfg.embedding_provider}")
        print(f"  Collection: {self._cfg.collection_name}")
        print(f"  Patterns  : {', '.join(self._cfg.patterns)}")
        print(f"  Stages    : {', '.join(stages)}")

        if "generate" in stages:
            ResponseGenerator(self._cfg).generate()

        if "custom_eval" in stages:
            CustomEvaluator(self._cfg).evaluate()

        if "ragas_eval" in stages:
            RagasEvaluator(self._cfg).evaluate()

        if "summarize" in stages:
            print(_banner("Stage 4 / 4 — Result Summarization"))
            s = ResultSummarizer(self._cfg)
            s.summarize_custom()
            s.summarize_ragas()

        print(_banner("Pipeline Complete"))


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
