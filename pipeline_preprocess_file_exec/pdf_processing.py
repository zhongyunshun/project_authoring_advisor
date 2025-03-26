import re
from preprocessing.pdf_processing import pdf_to_text, pdfs_to_text
from embeddings.embeddings import generate_and_save_embeddings
from preprocessing.text_splitter import read_and_split_text, split_text_into_chunks_nltk

def generate_embeddings_from_single_pdf(pdf_path: str, chunk_size=700):
    """
    Extracts text from a single PDF file, cleans it, splits into chunks, and generates embeddings.
    
    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        list: List of generated embeddings.
    """

    # Step 1: Extract text from a single PDF
    extracted_text = pdf_to_text(pdf_path)  

    # Step 2: Clean the extracted text
    pattern = r"[^a-zA-Z0-9.,!?;:'\[\]\"()\-\s\n]"  # Keep only valid characters
    filtered_text = re.sub(pattern, " ", extracted_text)  # Remove unwanted characters
    filtered_text = re.sub(r'[^\S\n]+', ' ', filtered_text)  # Remove extra spaces

    # Step 3: Split text into chunks
    chunks = split_text_into_chunks_nltk(filtered_text, chunk_size=chunk_size)

    # Step 4: Generate embeddings from text chunks
    vector_db = generate_and_save_embeddings(chunks)

    return vector_db


def generate_embeddings_from_pdf_directory(pdf_directory: str, chunk_size=700):
    """
    Extracts text from all PDFs in a directory, cleans it, splits into chunks, and generates embeddings.
    
    Args:
        pdf_directory (str): Directory containing multiple PDF files.

    Returns:
        list: List of generated embeddings.
    """

    # Step 1: Extract text from all PDFs in the directory
    combined_text = pdfs_to_text(pdf_directory)  

    # Step 2: Clean the extracted text
    pattern = r"[^a-zA-Z0-9.,!?;:'\[\]\"()\-\s\n]"  # Keep only valid characters
    filtered_text = re.sub(pattern, " ", combined_text)  # Remove unwanted characters
    filtered_text = re.sub(r'[^\S\n]+', ' ', filtered_text)  # Remove extra spaces

    # Step 3: Split text into chunks
    chunks = split_text_into_chunks_nltk(filtered_text, chunk_size=chunk_size)

    # Step 4: Generate embeddings from text chunks
    vector_db = generate_and_save_embeddings(chunks)

    return vector_db
