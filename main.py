import os
import pandas as pd
from preprocessing.text_splitter import split_text_into_chunks_nltk
from embeddings.embeddings import save_embeddings_to_database
from pipelines.rag_pipeline import rag_pipeline
from config.keys import OPENAI_API_KEY
from embeddings.embeddings import save_embeddings_to_file, load_embeddings_from_file


os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def read_and_split_text(input_file, chunk_size=200):
    """
    Reads a text file and splits it into chunks of the specified size.
    """
    with open(input_file, "r", encoding="utf-8") as infile:
        combined_text = infile.read()
    
    chunks = split_text_into_chunks_nltk(combined_text, chunk_size=chunk_size)
    return chunks


def generate_and_save_embeddings(chunks, embedding_file):
    """
    Generates embeddings from the chunks and saves them to a file.
    """
    print("Generating new embeddings...")
    vector_db = save_embeddings_to_database(chunks)
    save_embeddings_to_file(vector_db, embedding_file)
    print(f"Embeddings saved to {embedding_file}.")
    return vector_db


def answer_question(question, vector_db):
    """
    Answer a single question using RAG Pipeline.
    """
    prefix = "Here is a fill in blank question, please generate the answer, only output the answer without giving the original sentence: "

    answer = rag_pipeline(prefix + question, vector_db)
    print(f"Question: {question}")
    print(f"Answer: {answer}")


def process_questions(csv_file, vector_db, output_csv_file):
    """
    Reads questions from csv, process them through RAG pipeline,
    saves the answers in a new CSV file.
    """
    df = pd.read_csv(csv_file)

    if "question" not in df.columns:
        raise ValueError("CSV file must contain a 'question' column.")

    prefix = "Here is a fill in blank question, please generate the answer, only output the answer without giving the original sentence: "
    df["generated_answer"] = df["question"].apply(lambda q: rag_pipeline(prefix + q, vector_db))

    df.to_csv(output_csv_file, index=False)
    print(f"Processed questions saved to {output_csv_file}.")


def main():
    embedding_file = "embeddings.faiss"
    
    if os.path.exists(embedding_file):
        # Load the existing embeddings
        print(f"Loading existing embeddings from {embedding_file}")
        vector_db = load_embeddings_from_file(embedding_file)
        print("Embeddings loaded from file.")
    else:
        # Read txt, split into chunks, generate and store embeddings, save to file
        input_file = "./output/output.txt"
        chunks = read_and_split_text(input_file)

        vector_db = generate_and_save_embeddings(chunks, embedding_file)

    # Step 4: Test the RAG pipeline
    # test_question = "In 2017 Bercy Wycliffe Workplan, Based on the Infrastructure Hazard Monitoring Program, which site is the highest priority for remedial action in the Region?"  # Example question
    # answer_question(test_question, vector_db)
    
    # Output csv from a qa.csv
    input_csv_file = "data/QA_pair/fill_in_the_blanks_questions.csv"
    output_csv_file = "data/QA_pair/generated_qa.csv"
    process_questions(input_csv_file, vector_db, output_csv_file)


if __name__ == "__main__":
    main()
