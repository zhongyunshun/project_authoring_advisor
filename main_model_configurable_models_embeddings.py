import os
import pandas as pd
import argparse
from tqdm import tqdm

from preprocessing.text_splitter import read_and_split_text
from pipelines.rag_pipeline_cme import ConversationalRAG, StatelessRAG
from embeddings.embeddings import generate_and_save_embeddings, load_embeddings_from_file
from config.keys import OPENAI_API_KEY, GOOGLE_API_KEY

from langchain_openai import ChatOpenAI
from langchain_community.chat_models.llamacpp import ChatLlamaCpp
from langchain_google_genai import ChatGoogleGenerativeAI  # NEW (Gemini)
# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

def answer_question(question, rag_model):
    """
    Answer a single question using the Conversational RAG Pipeline.
    """
    answer, _ = rag_model.invoke(question)
    print(f"💡 **Answer**: {answer}\n")


def process_questions(csv_file, rag_model, output_csv_file):
    """
    Reads questions from a CSV, processes them through RAG pipeline with memory,
    and saves the answers in a new CSV file.
    """
    df = pd.read_csv(csv_file)

    if "question" not in df.columns:
        raise ValueError("CSV file must contain a 'question' column.")

    tqdm_iterator = tqdm(df["question"], desc="Processing Questions")

    # Initialize empty lists to store the results
    generated_answers = []
    retrieved_contexts = []

    # Iterate over the questions and invoke the RAG model
    for q in tqdm_iterator:
        generated_answer, retrieved_context = rag_model.invoke(q)
        generated_answers.append(generated_answer)
        retrieved_contexts.append(retrieved_context)

    # Assign the results to the DataFrame columns
    df["generated_answer"] = generated_answers
    df["retrieved_contexts"] = retrieved_contexts

    df.to_csv(output_csv_file, index=False)
    print(f"\n✅ Processed questions saved to: {output_csv_file}.")


def ConversationalRAGChat(rag_model):
    """
    Interactive mode for asking questions using Conversational RAG.
    Runs in a while loop until 'exit' is typed.
    """
    print("\n🤖 **Conversational RAG Mode Activated!** Type 'exit' to quit.\n")
    while True:
        user_input = input("💬 Ask a question: ").strip()
        
        if user_input.lower() == "exit":
            print("\n👋 Exiting Conversational Mode. See you next time!")
            break
        
        answer_question(user_input, rag_model)

def _build_llamacpp(model_path: str, target_ctx: int, max_new: int):
    """Try to load llama.cpp at target context; fall back if VRAM is insufficient."""
    for n_ctx in (target_ctx, 8192, 4096, 2048):
        try:
            return ChatLlamaCpp(
                model_path=model_path,
                n_ctx=n_ctx,                 # critical: context window
                n_batch=min(256, n_ctx // 2),
                n_gpu_layers=-1,             # offload as many as possible to GPU
                max_tokens=min(max_new, n_ctx // 2),
                temperature=0.2,
                verbose=False,
            )
        except Exception as e:
            print(f"⚠️ Failed to init with n_ctx={n_ctx}: {e}")
    raise RuntimeError("Could not initialize LlamaCpp; try a smaller n_ctx.")

def main():
    # 22, 700, similarity
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_length", type=int, default=700, help="Chunk length for text splitting")
    parser.add_argument("--top_k", type=int, default=22, help="Top k chunks to retrieve")
    parser.add_argument("--search_type", type=str, default="similarity", choices=["similarity", "mmr", "similarity_score_threshold"], help="Similarity measuring method")
    parser.add_argument("--input_csv_file", type=str, default="QA_pair/qa_pair_200_0210/sample/TRCA_All_Files_Combined_with_alternative_answers.csv", help="Input csv file")
    parser.add_argument("--output_csv_path", type=str, default="QA_pair/qa_pair_200_0210/output_200", help="Output csv path")
    parser.add_argument("--mode", type=str, default="csv", choices=["csv", "chat"], help="Choose 'csv' for batch processing or 'chat' for interactive mode")

    # For Configurable Models and Embeddings
    parser.add_argument("--model", type=str, default="openai",
                         choices=["openai", "qwen", "llama", "gemini"],
                         help="LLM to use for question answering: 'openai' (GPT-4o-mini API), 'qwen' (Qwen-7B local), 'llama' (Llama-3.1-8B local) or 'gemini' (Gemini API)")
    parser.add_argument("--embedding", type=str, default="ds",
                        choices=["ds", "sentencebert", "sbert"],
                        help="Embedding model for vector DB: 'ds' (Domain Specific Embeddings) or 'sentencebert' (all-MiniLM-L6-v2)")
    parser.add_argument("--model_path", type=str, default="",
                         help="Path to local model file (GGUF/AWQ) for Qwen or Llama (optional)")
    # NEW: control local LLM context window & output length
    parser.add_argument("--n_ctx", type=int, default=8192, help="Context window for local LLMs (Qwen/Llama)")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max new tokens for local LLMs")

    args = parser.parse_args()
    # Include embedding model name in the vector DB filename for clarity
    embedding_file = f"vector_db/{args.embedding}_chunk_{args.chunk_length}_embedding"

    if os.path.exists(embedding_file):
        print(f"\n📂 Loading existing embeddings from {embedding_file}...")
        vector_db = load_embeddings_from_file(embedding_file, embed_model=args.embedding)
        print("✅ Embeddings loaded from file.")
    else:
        input_file = "./output/cleaned_17.txt"
        chunks = read_and_split_text(input_file)
        vector_db = generate_and_save_embeddings(chunks, embedding_file, embed_model=args.embedding)
        print(f"✅ New embeddings generated and saved to {embedding_file}.")

    # Select and initialize the QA model based on user choice
    if args.model == "openai":
        # OpenAI API model (requires valid API key in config.keys)
        qa_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    elif args.model == "qwen":
        model_path = args.model_path or "./models/Qwen2.5-7B-Instruct/qwen2.5-7b-instruct-q3_k_m.gguf"  # default path (update as needed)
        qa_llm = ChatLlamaCpp(
            model_path=model_path,
            n_ctx=args.n_ctx,  # <-- real context window
            max_tokens=args.max_tokens,      # new tokens to generate
            n_gpu_layers=-1,                 # offload as many layers to GPU as possible
            n_batch=min(256, args.n_ctx // 2),
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            repeat_penalty=1.1,
            stop=["<|im_end|>", "<|eot_id|>", "</s>", "User:", "Assistant:"],
            verbose=False)
        print(f"✅ Loaded Qwen-7B model from {model_path}")
    elif args.model == "llama":
        model_path = args.model_path or "./models/Llama-3.1-8B-Instruct/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
        qa_llm = ChatLlamaCpp(
            model_path=model_path,
            n_ctx=args.n_ctx,  # <-- real context window
            max_tokens=args.max_tokens,  # new tokens to generate
            n_gpu_layers=-1,  # offload as many layers to GPU as possible
            n_batch=min(256, args.n_ctx // 2),
            temperature=0.7,
            verbose=False)
        print(f"✅ Loaded Llama-3.1-8B model from {model_path}")
    elif args.model == "gemini":
        qa_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
        print("✅ Loaded Gemini model: gemini-2.0-flash")

    # Define system message
    system_message_csv = "Please generate the correct answer for the given fill-in-the-blank question. Avoid including unnecessary context, restating the question, or adding explanations—only return the precise answer."
    system_message_chat = "You are a helpful assistant with memory. Answer questions accordingly."

    if args.mode == "csv":
        rag_model = StatelessRAG(vector_db, system_message_csv, args.top_k, args.search_type, llm=qa_llm)
        input_csv_file = args.input_csv_file
        output_csv_file = os.path.join(args.output_csv_path, f"output_chunk{args.chunk_length}_top{args.top_k}_{args.search_type}.csv")
        process_questions(input_csv_file, rag_model, output_csv_file)
    elif args.mode == "chat":
        rag_model = ConversationalRAG(vector_db, system_message_chat, args.top_k, args.search_type, llm=qa_llm)
        ConversationalRAGChat(rag_model)


if __name__ == "__main__":
    main()
