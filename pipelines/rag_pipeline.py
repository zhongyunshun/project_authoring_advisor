from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from embeddings.embeddings import load_vector_db
import re


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

# For streamlit upload pdf usage (document retrieval)
class ConversationalPDFRAG:
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

        # Print content and metadata
        for i, doc in enumerate(context):
            # Replace one or more occurrences of ASCII character 3 (ETX) with a space
            doc.page_content = re.sub(r'\x03+', ' ', doc.page_content)

            print(f"\n--- Document {i + 1} ---")
            print(f"Content:\n{doc.page_content[:300]}...")  # Print first 300 chars
            print("Metadata:")
            for key, value in doc.metadata.items():
                if key == "page":
                    print(f"- **{key}**: {value + 1}") # page count start with 0 instead of 1
                else:
                    print(f"- **{key}**: {value}")

        # Store conversation in memory
        self.memory.save_context(
            inputs={"question": question},
            outputs={"response": response}
        )

        return response, context
    
# For prompting testing:
class PromptingRAG:
    def __init__(self, vector_db=None, top_k=5, search_type="similarity", pattern="persona+cot+format"):
        """
        Stateless RAG for prompting engineer
        """
        self.delimiter = "####"
        self.pattern = pattern
        self.top_k = top_k
        self.search_type = search_type
        self.llm = ChatOpenAI(model="gpt-4o-mini")

        # Default to empty FAISS if no vector DB is provided
        if vector_db is None:
            embedding_model = OpenAIEmbeddings()
            self.vector_db = FAISS.from_texts([""], embedding_model)
        else:
            self.vector_db = vector_db

        self.retriever = self.vector_db.as_retriever(search_type=self.search_type, search_kwargs={"k": self.top_k})

        # Define Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("human", "{question}"),
        ])

        # Static prompt templates used in generate_prompt
        self.persona = f"""
# Persona
"goal": "You are designed to be a specialized question-answering assistant, focusing on providing accurate answers based on Toronto and Region Conservation Authority (TRCA)'s technical documents, supplemented by web search results and GPT-4's knowledge base. The query will be delimited with four hashtags (i.e., {self.delimiter})."
        """

        self.cot = f"""
# Chain of Thought
Step 1: {self.delimiter} Refer to TRCA's technical documents first.
Step 2: {self.delimiter} If the information is incomplete, use web search for current data.
Step 3: {self.delimiter} If still unresolved, utilize GPT-4's knowledge (up to its training cutoff).
Step 4: {self.delimiter} Cite sources from TRCA docs or web. Indicate if info is based on GPT-4's training data.
        """

        self.format_template = """
# Format Template
You are designed to ask for clarifications in case of ambiguous queries or when more specific details are needed.
The tone of the responses will be professional, focusing on clarity, accuracy, and relevance, suitable for the technical nature of TRCA's content.
Cite the sources of information from TRCA's documents, web search results, or GPT-4's training data when applicable.
        """

        self.few_shot = """
# Few-Shot Example
Query: "Can you outline the phased approach for the Humber Bay Park East Shoreline Maintenance Project?"
Retrieved passages: "The eastern armourstone headland ... a risk to park users."
Answer: "The Humber Bay Park East Shoreline Maintenance Project is divided into multiple phases, each with specific timelines. The available document focuses on Phase I..."
        """

    def _generate_prompt(self, query, domain_info=""):
        user_input = f"The query will be delimited with {self.delimiter} characters: {self.delimiter} {query} {self.delimiter}"

        if self.pattern == 'persona+cot+format':
            return f"{self.persona}\n{domain_info}\n{self.cot}\n{self.format_template}\n{user_input}\n{self.few_shot}"
        elif self.pattern == 'cot+format':
            return f"{domain_info}\n{self.cot}\n{self.format_template}\n{user_input}\n{self.few_shot}"
        elif self.pattern == 'persona+format':
            return f"{self.persona}\n{domain_info}\n{self.format_template}\n{user_input}\n{self.few_shot}"
        elif self.pattern == 'persona+cot':
            return f"{self.persona}\n{domain_info}\n{self.cot}\n{user_input}\n{self.few_shot}"
        elif self.pattern == 'rag-only':
            return f"{domain_info}\n{user_input}"
        elif self.pattern == 'gpt-4o-mini':
            return f"{user_input}"
        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def format_docs_to_list(self, docs):
        return [doc.page_content for doc in docs]

    def invoke(self, query):
        # Retrieve top-k documents
        context = self.retriever.invoke(query)

        retrieval_log = ""

        for i, doc in enumerate(context):
            doc.page_content = re.sub(r'\x03+', ' ', doc.page_content)

            retrieval_log += f"\n--- Document {i + 1} ---\n"
            retrieval_log += f"Content:\n{doc.page_content[:700]}...\n"
            retrieval_log += "Metadata:\n"

            for key, value in doc.metadata.items():
                if key == "page":
                    retrieval_log += f"- **{key}**: {value + 1}\n"
                else:
                    retrieval_log += f"- **{key}**: {value}\n"

        # ---Document 1---
        # Content: (700 char chunk)
        # Metadata:
        # source: (filename)
        # page: (1,2,...)
        # ---Document 2---
        # ...

        domain_info = f"""
# Inject Domain Information
Here is the retrieved passage:
{{
    "{retrieval_log}"
}}
        """

        # Generate the full prompt string
        full_prompt = self._generate_prompt(query, domain_info=domain_info)
        # print(full_prompt)

        # Create dynamic chain
        rag_chain = (
            {
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        response = rag_chain.invoke(full_prompt)
        return response, full_prompt

