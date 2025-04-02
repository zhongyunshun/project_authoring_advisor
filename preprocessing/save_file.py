import os

def save_text_to_file(text, output_file_path):
    """
    Save the given text to a .txt file.

    Args:
        text (str): The text to save.
        output_file_path (str): The path to the output .txt file.

    Returns:
        str: The absolute path of the saved file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    
    # Write the text to the file
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(text)
    
    return os.path.abspath(output_file_path)
