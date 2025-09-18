# database.py — simple, embedding-agnostic save/load helpers for FAISS

from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


def save_vector_db(vector_db: FAISS, path: str) -> None:
    """
    Save a LangChain FAISS vector store (folder with index + store).
    """
    vector_db.save_local(path)


def load_vector_db(path: str, embedding: Embeddings) -> FAISS:
    """
    Load a LangChain FAISS vector store with the SAME embedding function.
    """
    return FAISS.load_local(path, embedding, allow_dangerous_deserialization=True)


def merge_vector_dbs(db1: Optional[FAISS], db2: Optional[FAISS]) -> Optional[FAISS]:
    if db1 is None:
        return db2
    if db2 is None:
        return db1
    db1.merge_from(db2)
    return db1
