# embeddings.py  — provider-agnostic, no OpenAI dependency required

from typing import List, Optional, Union
import os
import tempfile

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Use community import (the old `langchain.embeddings` import is deprecated)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings  # Interface / Protocol


# ---------- Factory (optional helper) ----------

def get_hf_embeddings(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device: Optional[str] = None,
    normalize_embeddings: bool = True,
    **kwargs,
) -> Embeddings:
    """
    Returns a Hugging Face embedding model that implements LangChain's Embeddings interface.
    Works with any Sentence-Transformers checkpoint (E5, GTE, instructor-* etc.).
    """
    model_kwargs = {}
    if device is not None:
        model_kwargs["device"] = device

    encode_kwargs = {"normalize_embeddings": normalize_embeddings}
    encode_kwargs.update(kwargs.pop("encode_kwargs", {}))

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        **kwargs,
    )


# ---------- Core I/O over FAISS ----------

def generate_and_save_embeddings(
    text_chunks: List[str],
    file_path: str,
    embedding: Embeddings,
) -> FAISS:
    """
    Create a FAISS vector store from text_chunks using ANY LangChain Embeddings object,
    then save it to disk at file_path (folder path).
    """
    vector_db = FAISS.from_texts(text_chunks, embedding)
    vector_db.save_local(file_path)
    return vector_db


def load_embeddings_from_file(
    file_path: str,
    embedding: Embeddings,
) -> FAISS:
    """
    Load a FAISS vector store from disk with the SAME embedding function for searching.
    (FAISS.load_local requires an Embeddings object for query-time encoding.)
    """
    return FAISS.load_local(file_path, embedding, allow_dangerous_deserialization=True)


def merge_faiss_vector_dbs(vector_db1: Optional[FAISS], vector_db2: Optional[FAISS]) -> Optional[FAISS]:
    """Merge two FAISS vector DBs (in-place on db1)."""
    if vector_db1 is None:
        return vector_db2
    if vector_db2 is None:
        return vector_db1
    vector_db1.merge_from(vector_db2)
    return vector_db1


# ---------- PDF helpers (embedding-agnostic) ----------

def load_pdf_files_from_folder(folder_path: str) -> List[str]:
    return [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".pdf") and os.path.isfile(os.path.join(folder_path, f))
    ]


def load_documents_with_metadata_pdf(file_paths: List[str]):
    all_docs = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = os.path.basename(path)
        all_docs.extend(docs)
    return all_docs


def create_vector_db_from_pdfs(
    folder_path: str,
    embedding: Embeddings,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> FAISS:
    file_paths = load_pdf_files_from_folder(folder_path)
    documents = load_documents_with_metadata_pdf(file_paths)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    return FAISS.from_documents(chunks, embedding)


def create_vector_db_from_pdf(
    uploaded_file,
    embedding: Embeddings,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> FAISS:
    """
    Streamlit upload note: PyPDFLoader expects a file path, so we spool to a temp file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    for doc in documents:
        # Show original filename in results
        if hasattr(uploaded_file, "name"):
            doc.metadata["source"] = uploaded_file.name

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    return FAISS.from_documents(chunks, embedding)
