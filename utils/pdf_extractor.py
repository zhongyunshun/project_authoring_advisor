"""
This script is used for converting all pdf files in a directory into a txt file without filtering or formating the content.
"""

from preprocessing.pdf_processing import pdfs_to_text
from preprocessing.save_file import save_text_to_file

# Step 1: Extract text from all PDFs in the directory
pdf_directory = "data/ALL_PDFS"
combined_text = pdfs_to_text(pdf_directory)
print("Step 1 Done")

# Step 1.5: Save combined text to a .txt file for reference
output_file_path = "output/combined_text.txt"
saved_file_path = save_text_to_file(combined_text, output_file_path)
print(f"Combined text saved to: {saved_file_path}")