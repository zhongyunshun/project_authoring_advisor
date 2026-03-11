"""Unified CLI entry point for the TRCA RAG system.

Replaces main_model.py and main_model_configurable_models_embeddings.py.

Usage:
    python main.py --mode chat --model openai --embedding openai
    python main.py --mode csv --input_csv_file questions.csv --output_csv_path output/
    python main.py --mode agent --model openai  # agentic mode with tool selection
"""

import argparse
import os

import pandas as pd
from tqdm import tqdm

from config.settings import Settings
from core.llm_factory import LLMFactory
from core.embedding_factory import EmbeddingFactory
from core.vector_store import VectorStoreManager
from ingest.indexer import Indexer
from pipeline.rag_engine import RAGEngine, PromptingRAGEngine
from agents.rag_agent import RAGAgent
from agents.tools import create_document_query_tool, create_web_search_tool


def interactive_chat(engine):
    """Interactive REPL for conversational Q&A."""
    print("\nConversational RAG Mode Activated! Type 'exit' to quit.\n")
    while True:
        user_input = input("Ask a question: ").strip()
        if user_input.lower() == "exit":
            print("\nExiting. See you next time!")
            break
        result = engine.query(user_input)
        print(f"\nAnswer: {result.answer}\n")


def process_csv(engine, input_csv: str, output_csv: str):
    """Batch-process questions from CSV."""
    df = pd.read_csv(input_csv)
    if "question" not in df.columns:
        raise ValueError("CSV file must contain a 'question' column.")

    answers, contexts = [], []
    for q in tqdm(df["question"], desc="Processing Questions"):
        result = engine.query(q)
        answers.append(result.answer)
        contexts.append([s.text for s in result.sources])

    df["generated_answer"] = answers
    df["retrieved_contexts"] = contexts

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nProcessed questions saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="TRCA RAG System")
    parser.add_argument("--mode", default="chat", choices=["csv", "chat", "agent"],
                        help="'csv' for batch, 'chat' for interactive, 'agent' for agentic mode")
    parser.add_argument("--model", default="openai", choices=["openai", "gemini", "qwen", "llama"],
                        help="LLM provider")
    parser.add_argument("--embedding", default="openai", choices=["openai", "huggingface", "sbert", "ds"],
                        help="Embedding model provider")
    parser.add_argument("--model_path", default="", help="Path to local GGUF model file")
    parser.add_argument("--n_ctx", type=int, default=8192, help="Context window for local LLMs")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max output tokens")
    parser.add_argument("--top_k", type=int, default=22, help="Number of chunks to retrieve")
    parser.add_argument("--chunk_size", type=int, default=700, help="Chunk size for text splitting")
    parser.add_argument("--collection", default="trca_documents", help="Qdrant collection name")
    parser.add_argument("--pdf_dir", default="data", help="Directory with source PDFs")
    parser.add_argument("--input_csv_file", default="", help="Input CSV file for batch mode")
    parser.add_argument("--output_csv_path", default="output", help="Output directory for CSV results")
    parser.add_argument("--reindex", action="store_true", help="Force re-indexing of PDFs")
    parser.add_argument("--web_search", action="store_true", help="Enable web search tool (agent mode)")
    parser.add_argument("--pattern", default="rag-only",
                        choices=["persona+cot+format", "cot+format", "persona+format",
                                 "persona+cot", "rag-only", "gpt-4o-mini"],
                        help="Prompt pattern for prompting mode")
    args = parser.parse_args()

    # Load settings from environment (keys, etc.)
    settings = Settings.from_env()
    settings.apply_env()

    # Also try to load from config/keys.py if env vars are empty
    if not settings.openai_api_key:
        try:
            from config.keys import OPENAI_API_KEY, GOOGLE_API_KEY
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            os.environ.setdefault("GOOGLE_API_KEY", GOOGLE_API_KEY)
        except ImportError:
            pass

    # Create LLM
    llm_provider = args.model if args.model not in ("qwen",) else "llama_cpp"
    llm = LLMFactory.create(
        provider=llm_provider,
        temperature=0.7,
        max_tokens=args.max_tokens,
        model_path=args.model_path,
        n_ctx=args.n_ctx,
    )
    print(f"LLM: {args.model}")

    # Create embedding model
    embed_model = EmbeddingFactory.create(provider=args.embedding)
    print(f"Embedding: {args.embedding}")

    # Set up vector store and index
    vsm = VectorStoreManager(storage_path="./vector_db/qdrant_storage")
    indexer = Indexer(vsm, embed_model, chunk_size=args.chunk_size)

    if args.reindex or not vsm.collection_exists(args.collection):
        print(f"Indexing PDFs from {args.pdf_dir}...")
        index = indexer.index_directory(args.pdf_dir, args.collection)
        print("Indexing complete.")
    else:
        print(f"Loading existing collection: {args.collection}")
        index = indexer.load_index(args.collection)

    # Build the appropriate engine/agent
    if args.mode == "agent":
        tools = [create_document_query_tool(index, llm, top_k=args.top_k)]
        if args.web_search:
            tools.append(create_web_search_tool())
        engine = RAGAgent(tools=tools, llm=llm, verbose=True)
        interactive_chat(engine)

    elif args.mode == "chat":
        engine = RAGEngine(
            index=index, llm=llm,
            system_prompt=settings.system_prompt_chat,
            top_k=args.top_k, conversational=True,
        )
        interactive_chat(engine)

    elif args.mode == "csv":
        if not args.input_csv_file:
            parser.error("--input_csv_file is required for csv mode")
        engine = RAGEngine(
            index=index, llm=llm,
            system_prompt=settings.system_prompt_csv,
            top_k=args.top_k, conversational=False,
        )
        output_csv = os.path.join(
            args.output_csv_path,
            f"output_chunk{args.chunk_size}_top{args.top_k}_{args.model}.csv",
        )
        process_csv(engine, args.input_csv_file, output_csv)


if __name__ == "__main__":
    main()
