import os
from preprocessing.text_splitter import split_text_into_chunks_nltk
from embeddings.embeddings import save_embeddings_to_database
from embeddings.database import save_embeddings_to_database_pickel, load_embeddings_from_database_pickel
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

def main():
    # Step 3: Check if embeddings file exists and load it, otherwise generate and save embeddings
    embedding_file = "embeddings.faiss"
    
    if os.path.exists(embedding_file):
        # Load the existing embeddings
        print(f"Loading existing embeddings from {embedding_file}")
        vector_db = load_embeddings_from_file(embedding_file)
        print(type(vector_db))
        print("Embeddings loaded from file.")
    else:
        # Step 1 & 2: Read document from txt file and split into chunks
        input_file = "./output/output.txt"
        chunks = read_and_split_text(input_file)
        
        # Generate and store embeddings, then save to file
        vector_db = generate_and_save_embeddings(chunks, embedding_file)
        print(type(vector_db))

    # Step 4: Test the RAG pipeline
    test_question = "In 2017 Bercy Wycliffe Workplan, Based on the Infrastructure Hazard Monitoring Program, which site is the highest priority for remedial action in the Region?"  # Example question
    answer = rag_pipeline(test_question, vector_db)
    print(f"Question: {test_question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
