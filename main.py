import os
import pandas as pd
import argparse
from tqdm import tqdm

from preprocessing.text_splitter import read_and_split_text
from pipelines.rag_pipeline import ConversationalRAG, StatelessRAG
from embeddings.embeddings import generate_and_save_embeddings, load_embeddings_from_file
from config.keys import OPENAI_API_KEY

# Set OpenAI API Key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def answer_question(question, rag_model):
    """
    Answer a single question using the Conversational RAG Pipeline.
    """
    answer = rag_model.invoke(question)
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
    df["generated_answer"] = [rag_model.invoke(q) for q in tqdm_iterator]

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


def main():
    # 20, 700, similarity
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_length", type=int, default=200, help="Chunk length for text splitting")
    parser.add_argument("--top_k", type=int, default=5, help="Top k chunks to retrieve")
    parser.add_argument("--search_type", type=str, default="similarity", choices=["similarity", "mmr", "similarity_score_threshold"], help="Similarity measuring method")
    parser.add_argument("--input_csv_file", type=str, default="QA_pair/qa_pair_200_0210/sample/TRCA_All_Files_Combined_with_alternative_answers.csv", help="Input csv file")
    parser.add_argument("--output_csv_path", type=str, default="QA_pair/qa_pair_200_0210/output_200", help="Output csv path")
    parser.add_argument("--mode", type=str, default="csv", choices=["csv", "chat"], help="Choose 'csv' for batch processing or 'chat' for interactive mode")

    args = parser.parse_args()

    embedding_file = f"vector_db/openai_chunk_{args.chunk_length}_embedding"

    if os.path.exists(embedding_file):
        print(f"\n📂 Loading existing embeddings from {embedding_file}...")
        vector_db = load_embeddings_from_file(embedding_file)
        print("✅ Embeddings loaded from file.")
    else:
        input_file = "./output/cleaned_17.txt"
        chunks = read_and_split_text(input_file)
        vector_db = generate_and_save_embeddings(chunks, embedding_file)

    # Define system message
    system_message_csv = "Please generate the correct answer for the given fill-in-the-blank question. Avoid including unnecessary context, restating the question, or adding explanations—only return the precise answer."
    system_message_chat = "You are a helpful assistant with memory. Answer questions accordingly."

    if args.mode == "csv":
        rag_model = StatelessRAG(vector_db, system_message_csv, args.top_k, args.search_type)
        input_csv_file = args.input_csv_file
        output_csv_file = os.path.join(args.output_csv_path, f"output_chunk{args.chunk_length}_top{args.top_k}_{args.search_type}.csv")
        process_questions(input_csv_file, rag_model, output_csv_file)
    elif args.mode == "chat":
        rag_model = ConversationalRAG(vector_db, system_message_chat, args.top_k, args.search_type)
        ConversationalRAGChat(rag_model)


if __name__ == "__main__":
    main()
