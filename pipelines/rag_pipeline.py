from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA

def rag_pipeline(question, vector_db, top_k=5):
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA(llm=llm, retriever=retriever)
    return qa_chain.run(question)
