from preprocessing.text_splitter import split_text_into_chunks_nltk
from embeddings.database import save_embeddings_to_database_pickel
import torch

def process_text_and_store_embeddings(input_file, chunk_size=200, output_embeddings_file="./output/embeddings.txt"):
    """
    Process text from a file, split it into chunks, generate embeddings, and store them in a file.

    Args:
        input_file (str): Path to the input text file.
        chunk_size (int): Size of text chunks.
        output_embeddings_file (str): Path to save the embeddings data.

    Returns:
        str: Path to the output embeddings file.
    """
    with open(input_file, "r", encoding="utf-8") as infile:
        combined_text = infile.read()

    chunks = split_text_into_chunks_nltk(combined_text, chunk_size=chunk_size)

    vector_db = save_embeddings_to_database_pickel(chunks, save_path=output_embeddings_file)

    print(f"Embeddings saved to {output_embeddings_file}")
    return output_embeddings_file
