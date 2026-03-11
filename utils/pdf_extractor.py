"""
Extract text from all PDFs in a directory and save to a single text file.
"""

from ingest.pdf_loader import PDFLoader
from preprocessing.save_file import save_text_to_file

# Step 1: Extract text from all PDFs in the directory
pdf_directory = "data/ALL_PDFS"
documents = PDFLoader.load_directory(pdf_directory)
combined_text = "\n".join(doc.text for doc in documents)
print("Step 1 Done")

# Step 2: Save combined text to a .txt file for reference
output_file_path = "output/combined_text.txt"
saved_file_path = save_text_to_file(combined_text, output_file_path)
print(f"Combined text saved to: {saved_file_path}")
