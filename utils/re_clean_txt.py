"""
This script is using re to filter and clean the txt content in a txt file.
It keeps all the normal expressions and filter out non utf8 characters.
It writes the filtered content into another txt file.
"""

import re

input_file = "output/combined_text.txt"  # Replace with your input file path
output_file = "output/output.txt"  # Replace with your desired output file path

# Regular expression pattern: Keep letters, numbers, punctuation, spaces, and newlines
pattern = r"[^a-zA-Z0-9.,!?;:'\[\]\"()\-\s\n]"

# Read the input file
with open(input_file, "r", encoding="utf-8") as infile:
    content = infile.read()

# Apply the regular expression filter
filtered_content = re.sub(pattern, " ", content)

filtered_content_filter_blank = re.sub(r'[^\S\n]+', ' ', filtered_content)

# Write the filtered content to the output file
with open(output_file, "w", encoding="utf-8") as outfile:
    outfile.write(filtered_content_filter_blank)

print(f"Filtered content written to {output_file}")
