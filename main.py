import os
import pandas as pd
from preprocessing.text_splitter import read_and_split_text
from pipelines.rag_pipeline import rag_pipeline
from embeddings.embeddings import generate_and_save_embeddings, load_embeddings_from_file


from config.keys import OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


def answer_question(question, vector_db, sys_sms):
    """
    Answer a single question using RAG Pipeline.
    """
    answer = rag_pipeline(question, vector_db, sys_sms)
    print(f"Question: {question}")
    print(f"Answer: {answer}")


def process_questions(csv_file, vector_db, sys_sms, output_csv_file):
    """
    Reads questions from csv, process them through RAG pipeline,
    saves the answers in a new CSV file.
    """
    df = pd.read_csv(csv_file)

    if "question" not in df.columns:
        raise ValueError("CSV file must contain a 'question' column.")

    df["generated_answer"] = df["question"].apply(lambda q: rag_pipeline(q, vector_db, sys_sms))
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
        input_file = "./output/cleaned_17.txt"
        chunks = read_and_split_text(input_file)

        vector_db = generate_and_save_embeddings(chunks, embedding_file)

    # Step 4: Test the RAG pipeline
    sys_sms = "Here is a fill in blank question, please generate the answer, only output the answer without giving the original sentence: "
    # test_question = "In 2017 Bercy Wycliffe Workplan, Based on the Infrastructure Hazard Monitoring Program, which site is the highest priority for remedial action in the Region?"  # Example question
    # answer_question(test_question, vector_db)
    
    # Output csv from a qa.csv
    input_csv_file = "QA_pair/qa_pair_170_0204/TRCA_All_Files_Combined.csv"
    output_csv_file = "QA_pair/qa_pair_170_0204/TRCA_All_Files_Combined_output.csv"
    process_questions(input_csv_file, vector_db, sys_sms, output_csv_file)


if __name__ == "__main__":
    main()
