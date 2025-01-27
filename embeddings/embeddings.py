from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import faiss

def save_embeddings_to_database(chunks):
    embeddings_model = OpenAIEmbeddings()
    vector_db = FAISS.from_texts(chunks, embeddings_model)
    return vector_db

def save_embeddings_to_file(vector_db, filename):
    # # Extract the FAISS index from the vector_db
    # faiss_index = vector_db.index  # Assuming vector_db is a LangChain FAISS wrapper
    # faiss.write_index(faiss_index, filename)
    vector_db.save_local(filename)

def load_embeddings_from_file(filename):
    # # Load the FAISS index from a file
    # return faiss.read_index(filename)
    faiss_index = FAISS.load_local(filename, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    return faiss_index
