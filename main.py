from preprocessing.pdf_processing import pdfs_to_text
from preprocessing.text_splitter import split_text_into_chunks
from embeddings.embeddings import save_embeddings_to_database
from pipelines.rag_pipeline import rag_pipeline
from utils.save_file import save_text_to_file

def main():
    # Step 1: Extract text from all PDFs in the directory
    pdf_directory = "data/ALL_PDFS"
    combined_text = pdfs_to_text(pdf_directory)
    print("Step 1 Done")

    # Step 1.5: Save combined text to a .txt file for reference
    output_file_path = "output/combined_text.txt"
    saved_file_path = save_text_to_file(combined_text, output_file_path)
    print(f"Combined text saved to: {saved_file_path}")
    return
    
    # Step 2: Split combined text into chunks
    chunks = split_text_into_chunks(combined_text, chunk_size=200)
    print(chunks)
    print("Step 2 Done")
    
    # Step 3: Generate and store embeddings
    vector_db = save_embeddings_to_database(chunks)
    print("Step 3 Done")
    
    # Step 4: Test the RAG pipeline
    test_question = "What is the purpose of the TRCA document?"  # Example question
    answer = rag_pipeline(test_question, vector_db)
    print(f"Question: {test_question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
