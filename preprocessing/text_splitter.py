import nltk
from nltk.tokenize import sent_tokenize

# Ensure NLTK resources are downloaded
nltk.download("punkt", quiet=True)

def split_text_into_chunks(text, chunk_size=200):
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
