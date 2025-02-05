import nltk
from nltk.tokenize import sent_tokenize
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Ensure NLTK resources are downloaded
nltk.download("punkt", quiet=True)

def split_text_into_chunks_nltk(text, chunk_size=200):
    """
    Split text into chunks, ensuring no sentence is split across chunks.

    Args:
        text (str): The input text to split.
        chunk_size (int): The maximum number of words in a chunk.

    Returns:
        list: A list of text chunks.
    """
    sentences = sent_tokenize(text)  # Tokenize text into sentences
    chunks = []
    current_chunk = []
    current_chunk_word_count = 0

    for sentence in sentences:
        sentence_word_count = len(sentence.split())
        
        # If adding the sentence exceeds the chunk size, finalize the current chunk
        if current_chunk_word_count + sentence_word_count > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_chunk_word_count = 0

        # Add the sentence to the current chunk
        current_chunk.append(sentence)
        current_chunk_word_count += sentence_word_count

    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def read_and_split_text(input_file, chunk_size=200):
    """
    Reads a text file and splits it into chunks of the specified size.
    """
    with open(input_file, "r", encoding="utf-8") as infile:
        combined_text = infile.read()
    
    chunks = split_text_into_chunks_nltk(combined_text, chunk_size=chunk_size)
    return chunks


def split_text_into_chunks_langchain(text, chunk_size=200, chunk_overlap=50):
    """
    This function is not in use. This is saved for future use.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks
