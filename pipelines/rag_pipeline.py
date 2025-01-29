from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI

def rag_pipeline(question, vector_db, top_k=5):
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    # llm = OpenAI(model_name="gpt-4o-mini") #temperature=0.1
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Load a question-answering chain
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    retrieval_qa_chain = RetrievalQA(retriever=retriever, combine_documents_chain=qa_chain)

    return retrieval_qa_chain.run(question)