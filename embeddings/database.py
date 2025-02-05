import pickle
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

"""
This file is not in use. This is saved for future use.
"""

def save_embeddings_to_database_pickel(chunks, save_path="./embeddings/vector_db.pkl"):
    """
    Generate embeddings using OpenAIEmbeddings and store them in a FAISS vector database.
    The database is saved as a pickle file.

    Args:
        chunks (list): List of text chunks to be embedded.
        save_path (str): Path to save the pickle file.

    Returns:
        FAISS: FAISS vector database with embeddings.
    """
    # Initialize the embeddings model
    embeddings_model = OpenAIEmbeddings()

    # Generate the FAISS vector store
    vector_db = FAISS.from_texts(chunks, embeddings_model)

    # Save the vector database to file using pickle
    FAISS.write_index(vector_db.index, save_path)
    print(f"FAISS index saved to {save_path}")
    return vector_db

def load_embeddings_from_database_pickel(save_path="./embeddings/vector_db.pkl"):
    """
    Load a FAISS vector database from a pickle file.

    Args:
        save_path (str): Path to the pickle file.

    Returns:
        FAISS: Loaded FAISS vector database.
    """
    # with open(save_path, "rb") as f:
    #     vector_db = pickle.load(f)

    embeddings_model = OpenAIEmbeddings()

    index = FAISS.read_index(save_path)

    # Create a FAISS vector store with the loaded index and embeddings model
    vector_db = FAISS(index, embeddings_model)
    print(f"FAISS index loaded from {save_path}")

    return vector_db
