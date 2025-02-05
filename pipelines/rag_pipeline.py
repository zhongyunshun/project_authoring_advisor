from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import ChatOpenAI

def rag_pipeline(question, vector_db, system_message=None, top_k=5):
    retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Load a question-answering chain
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    retrieval_qa_chain = RetrievalQA(retriever=retriever, combine_documents_chain=qa_chain)
    
    # Construct messages format with system prompt
    # messages = [
    #     {"role": "system", "content": system_message},
    #     {"role": "user", "content": question}
    # ]
    
    return retrieval_qa_chain.run(query=question, context=system_message)
