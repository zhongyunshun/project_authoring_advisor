from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def save_embeddings_to_database(chunks):
    embeddings_model = OpenAIEmbeddings()
    vector_db = FAISS.from_texts(chunks, embeddings_model)
    return vector_db
