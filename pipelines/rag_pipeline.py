from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

class ConversationalRAG:
    def __init__(self, vector_db, system_message=None, top_k=5, search_type="similarity"):
        """
        Initializes the Conversational RAG model with memory storage.
        """
        self.vector_db = vector_db
        self.system_message = system_message or "You are a helpful assistant with memory. Answer questions accordingly."
        self.top_k = top_k
        self.search_type = search_type
        self.llm = ChatOpenAI(model="gpt-4o-mini")

        # Initialize an empty FAISS vector database if none is provided
        if vector_db is None:
            embedding_model = OpenAIEmbeddings()
            # Init to empty string because it expects list of strings, but an empty list would cause error. So init it as a list with a empty string
            self.vector_db = FAISS.from_texts([""], embedding_model)
        else:
            self.vector_db = vector_db

        # Initialize retriever
        self.retriever = self.vector_db.as_retriever(search_type=self.search_type, search_kwargs={"k": self.top_k})

        # Memory Storage
        self.memory = ConversationBufferMemory(return_messages=True)

        # Define Prompt with Memory
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_message),
            ("human", "Context: {context}\n\nChat History: {chat_history}\n\nQuestion: {question}"),
        ])

    def format_docs(self, docs):
        """Formats retrieved documents into a readable context."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def format_docs_to_list(self, docs):
        """Formats retrieved documents into a list of lists."""
        return [[doc.page_content] for doc in docs]

    def invoke(self, question):
        """
        Processes a question through the RAG pipeline, leveraging memory for context.
        """
        # Retrieve relevant documents
        context = self.retriever.invoke(question)
        formatted_context = self.format_docs(context)
        context_list = self.format_docs_to_list(context)

        # Fetch past chat history from memory
        chat_history_objects = self.memory.load_memory_variables({}).get("history", [])

        # Convert chat history to a readable string format
        chat_history = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in chat_history_objects
        ])

        # Create RAG pipeline with correct formatting
        rag_chain = (
            {
                "context": RunnablePassthrough() | (lambda x: formatted_context),
                "chat_history": RunnablePassthrough() | (lambda x: chat_history),
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # Get response
        response = rag_chain.invoke(question)

        # Store conversation in memory
        self.memory.save_context(
            inputs={"question": question},
            outputs={"response": response}
        )

        return response, context_list


class StatelessRAG:
    def __init__(self, vector_db, system_message=None, top_k=5, search_type="similarity"):
        """
        Stateless RAG (No Memory)
        """
        self.vector_db = vector_db
        self.system_message = system_message or "You are a helpful assistant."
        self.top_k = top_k
        self.search_type = search_type
        self.llm = ChatOpenAI(model="gpt-4o-mini")

        # Initialize an empty FAISS vector database if none is provided
        if vector_db is None:
            embedding_model = OpenAIEmbeddings()
            self.vector_db = FAISS.from_texts([""], embedding_model)
        else:
            self.vector_db = vector_db

        # Initialize retriever
        self.retriever = self.vector_db.as_retriever(search_type=self.search_type, search_kwargs={"k": self.top_k})

        # Define Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_message),
            ("human", "Context: {context}\n\nQuestion: {question}"),
        ])

    def format_docs(self, docs):
        """Formats retrieved documents into a readable context."""
        return "\n\n".join(doc.page_content for doc in docs)

    def format_docs_to_list(self, docs):
        """Formats retrieved documents into a list of lists."""
        return [doc.page_content for doc in docs]

    def invoke(self, question):
        """
        Processes a question through the RAG pipeline (No Memory).
        """
        # Retrieve relevant documents
        context = self.retriever.invoke(question)
        formatted_context = self.format_docs(context)
        context_list = self.format_docs_to_list(context)

        # Create RAG pipeline
        rag_chain = (
            {
                "context": RunnablePassthrough() | (lambda x: formatted_context),
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # Get response
        return rag_chain.invoke(question), context_list
