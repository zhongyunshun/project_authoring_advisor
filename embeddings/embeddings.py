from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from typing import List
import tempfile

from langchain.embeddings import HuggingFaceEmbeddings  # for Sentence-BERT embeddings

def save_embeddings_to_database(chunks):
    embeddings_model = OpenAIEmbeddings()
    vector_db = FAISS.from_texts(chunks, embeddings_model)
    return vector_db

def save_embeddings_to_file(vector_db, filename):
    # # Extract the FAISS index from the vector_db
    # faiss_index = vector_db.index  # Assuming vector_db is a LangChain FAISS wrapper
    # faiss.write_index(faiss_index, filename)
    vector_db.save_local(filename)

# def load_embeddings_from_file(filename):
#     # # Load the FAISS index from a file
#     # return faiss.read_index(filename)
#     faiss_index = FAISS.load_local(filename, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
#     return faiss_index

# For Configurable Models and Embeddings
def load_embeddings_from_file(file_path, embed_model="openai"):
    """
    Load a FAISS vector store from disk, using the same embedding model that was originally used.
    """
    if embed_model.lower() == "openai":
        embedding_model = OpenAIEmbeddings()
    elif embed_model.lower() in ("sentencebert", "sbert"):
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        raise ValueError(f"Unsupported embed_model '{embed_model}'.")

    # Load the FAISS index with the correct embedding function for queries
    vector_db = FAISS.load_local(file_path, embedding_model, allow_dangerous_deserialization=True)
    return vector_db

# def generate_and_save_embeddings(chunks, embedding_file=None):
#     """
#     Generates embeddings from the chunks and saves them to a file.
#     """
#     print("Generating new embeddings...")
#     vector_db = save_embeddings_to_database(chunks)
#     if embedding_file:
#         save_embeddings_to_file(vector_db, embedding_file)
#         print(f"Embeddings saved to {embedding_file}.")
#     return vector_db

# For Configurable Models and Embeddings
def generate_and_save_embeddings(text_chunks, file_path, embed_model="openai"):
    """
    Create a FAISS vector store from text_chunks using the specified embedding model,
    then save it to disk at file_path.
    """
    # Choose embedding model based on flag
    if embed_model.lower() == "openai":
        embedding_model = OpenAIEmbeddings()
    elif embed_model.lower() in ("sentencebert", "sbert"):
        # Use Sentence-BERT (all-MiniLM-L6-v2) for embeddings
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        raise ValueError(f"Unsupported embed_model '{embed_model}'.")

    # Create FAISS vector index from text chunks
    vector_db = FAISS.from_texts(text_chunks, embedding_model)
    # Save index to disk for later reuse
    vector_db.save_local(file_path)
    return vector_db



def merge_faiss_vector_dbs(vector_db1: FAISS, vector_db2: FAISS) -> FAISS:
    """
    Merges two FAISS vector databases into a single FAISS index.
    
    Args:
        vector_db1 (FAISS): First FAISS vector database.
        vector_db2 (FAISS): Second FAISS vector database.

    Returns:
        FAISS: A new FAISS vector database containing all embeddings.
    """
    # Check if no db is None
    if vector_db1 is None:
        return vector_db2
    if vector_db2 is None:
        return vector_db1
    
    vector_db1.merge_from(vector_db2)

    return vector_db1


def load_pdf_files_from_folder(folder_path: str) -> List[str]:
    """
    Loads and returns a list of file paths to .pdf files from the given folder.
    """
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".pdf") and os.path.isfile(os.path.join(folder_path, f))
    ]


def load_documents_with_metadata_pdf(file_paths: List[str]):
    """
    Loads PDF documents and attaches metadata (like filename).
    """
    all_docs = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = os.path.basename(path)
        all_docs.extend(docs)
    return all_docs


def create_vector_db_from_pdfs(folder_path: str) -> FAISS:
    """
    Loads PDFs, splits into chunks, and builds a FAISS vector DB with metadata.
    """
    file_paths = load_pdf_files_from_folder(folder_path)
    documents = load_documents_with_metadata_pdf(file_paths)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embedding_model = OpenAIEmbeddings()
    vector_db = FAISS.from_documents(chunks, embedding_model)
    return vector_db


def save_vector_db(vector_db: FAISS, path: str):
    vector_db.save_local(path)


def load_vector_db(path: str) -> FAISS:
    return FAISS.load_local(path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)


def create_vector_db_from_pdf(uploaded_file):
    # NOTE:
    # A temporary file is needed here because PyPDFLoader (LangChain) expects a file PATH, not a file-like object which is from upload file function in Streamlit.
    # On the other hand, PdfReader can directly take file-like objects.
    # In this case, since we are using PyPDFLoader instead of PdfReader, we have to save the uploaded file to disk temporarily to feed in a file path for it.

    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Load the PDF using PyPDFLoader
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # Add metadata (optional)
    for doc in documents:
        doc.metadata["source"] = uploaded_file.name

    # Split into chunks
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Create FAISS vector DB
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    embeddings = OpenAIEmbeddings()
    vector_db = FAISS.from_documents(chunks, embeddings)

    return vector_db
