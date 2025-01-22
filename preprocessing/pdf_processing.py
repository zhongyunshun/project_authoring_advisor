from PyPDF2 import PdfReader
import os

def pdf_to_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def pdfs_to_text(pdf_directory):
    """
    Extract text from all PDF files in a specified directory and combine them into a single string.

    Args:
        pdf_directory (str): Path to the directory containing PDF files.

    Returns:
        str: Combined text extracted from all PDFs.
    """
    all_text = ""
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_directory, filename)
            print(f"Processing {filename}...")
            try:
                reader = PdfReader(file_path)
                for page in reader.pages:
                    all_text += page.extract_text() + "\n"  # Add separator between pages
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return all_text
