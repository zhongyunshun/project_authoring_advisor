import os
import pandas as pd
from preprocessing.text_splitter import read_and_split_text
from pipelines.rag_pipeline import rag_pipeline
from embeddings.embeddings import generate_and_save_embeddings, load_embeddings_from_file
from tqdm import tqdm
import argparse


from config.keys import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def answer_question(question, vector_db, system_message, top_k, search_type):
    """
    Answer a single question using RAG Pipeline.
    """
    answer = rag_pipeline(question, vector_db, system_message, top_k, search_type)
    print(f"Question: {question}")
    print(f"Answer: {answer}")


def process_questions(csv_file, vector_db, system_message, output_csv_file, top_k, search_type):
    """
    Reads questions from csv, process them through RAG pipeline,
    saves the answers in a new CSV file.
    """
    df = pd.read_csv(csv_file)

    if "question" not in df.columns:
        raise ValueError("CSV file must contain a 'question' column.")

    tqdm_iterator = tqdm(df["question"], desc="Processing Questions")
    df["generated_answer"] = [rag_pipeline(q, vector_db, system_message, top_k, search_type) for q in tqdm_iterator]
    df.to_csv(output_csv_file, index=False)
    print(f"Processed questions saved to {output_csv_file}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_length", type=int, default=200, help="Chunk length for text splitting")
    parser.add_argument("--top_k", type=int, default=5, help="Top k chunks to retrieve")
    parser.add_argument("--search_type", type=str, default="similarity", choices=["similarity", "mmr", "similarity_score_threshold"], help="Similarity measuring method")
    parser.add_argument("--input_csv_file", type=str, default="QA_pair/qa_pair_200_0210/sample/TRCA_All_Files_Combined_with_alternative_answers_100.csv", help="Input csv file")
    args = parser.parse_args()

    embedding_file = f"vector_db/openai_chunk_{args.chunk_length}_embedding"
    
    if os.path.exists(embedding_file):
        # Load the existing embeddings
        print(f"Loading existing embeddings from {embedding_file}")
        vector_db = load_embeddings_from_file(embedding_file)
        print("Embeddings loaded from file.")
    else:
        # Read txt, split into chunks, generate and store embeddings, save to file
        input_file = "./output/cleaned_17.txt"
        chunks = read_and_split_text(input_file)

        vector_db = generate_and_save_embeddings(chunks, embedding_file)

    # Step 4: Test the RAG pipeline
    system_message = "Here is a fill in blank question, please generate the answer, only output the answer without giving the original sentence: "
    
    # # Single question pipeline
    # test_question = "In 2017 Bercy Wycliffe Workplan, Based on the Infrastructure Hazard Monitoring Program, which site is the highest priority for remedial action in the Region?"  # Example question
    # answer_question(test_question, vector_db, system_message, args.top_k, args.search_type)
    
    # Output csv from a qa.csv
    input_csv_file = args.input_csv_file
    output_csv_file = f"QA_pair/qa_pair_200_0210/output/output_chunk{args.chunk_length}_top{args.top_k}_{args.search_type}.csv"
    process_questions(input_csv_file, vector_db, system_message, output_csv_file, args.top_k, args.search_type)


if __name__ == "__main__":
    main()
