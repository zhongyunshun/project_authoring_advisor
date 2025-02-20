from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

def rag_pipeline(question, vector_db, system_message=None, top_k=5, search_type="similarity"):
    # Initialize the retriever and llm
    retriever = vector_db.as_retriever(search_type=search_type, search_kwargs={"k": top_k})
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Define system prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_message or "You are a helpful assistant. Answer all questions to the best of your ability.",
            ),
            ("human", "Context: {context}\nQuestion: {question}"),
        ]
    )

    # Define the RAG chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Invoke the chain and return
    result = rag_chain.invoke(question)
    return result
