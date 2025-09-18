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
# For Configurable Models and Embeddings
from langchain.chat_models.base import BaseChatModel  # to identify chat vs non-chat LLMs
from langchain_core.messages import SystemMessage, HumanMessage


class ConversationalRAG:
    def __init__(self, vector_db, system_message=None, top_k=5, search_type="similarity", **llm_kwargs):
        """Initializes the Conversational RAG model with memory (chat history)."""
        self.vector_db = vector_db or FAISS.from_texts([""], OpenAIEmbeddings())
        self.system_message = system_message or "You are a helpful assistant with memory. Answer questions accordingly."
        self.top_k = top_k
        self.search_type = search_type
        # Use provided LLM or default to OpenAI GPT-4o-mini
        provided_llm = llm_kwargs.get("llm", None)
        self.llm = provided_llm if provided_llm is not None else ChatOpenAI(model="gpt-4o-mini")

        # Check if the LLM is a chat model (OpenAI) or a standard local model
        self.is_chat_model = isinstance(self.llm, BaseChatModel)

        # Initialize retriever (vector store with similarity search)
        self.retriever = self.vector_db.as_retriever(search_type=self.search_type, search_kwargs={"k": self.top_k})

        # Conversation memory for chat history
        self.memory = ConversationBufferMemory(return_messages=True)

        # Define the prompt template (system + user prompt)
        if self.is_chat_model:
            # For chat models, use role-based prompt template
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_message),
                ("human", "Context: {context}\n\nChat History: {chat_history}\n\nQuestion: {question}")
            ])
        else:
            # For local LLMs, we'll format prompts manually in invoke()
            self.prompt = None  # not used for non-chat models

    def format_docs(self, docs):
        """Formats retrieved documents into a readable context."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def format_docs_to_list(self, docs):
        """Formats retrieved documents into a list of lists."""
        return [[doc.page_content] for doc in docs]

    def invoke(self, question):
        """Processes a question through the RAG pipeline (with memory)."""
        # Retrieve relevant documents as context
        context_docs = self.retriever.invoke(question)
        formatted_context = self.format_docs(context_docs)
        context_list = self.format_docs_to_list(context_docs)

        # Get past chat history from memory (as list of messages)
        history_msgs = self.memory.load_memory_variables({}).get("history", [])
        # Convert history to a string (e.g., "User: ...\nAssistant: ...")
        chat_history_str = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in history_msgs
        ])

        if self.is_chat_model:
            # Use LangChain pipeline: inject context, history, question into prompt template, then call LLM
            rag_chain = (
                    {
                        "context": RunnablePassthrough() | (lambda _: formatted_context),
                        "chat_history": RunnablePassthrough() | (lambda _: chat_history_str),
                        "question": RunnablePassthrough()
                    }
                    | self.prompt
                    | self.llm
                    | StrOutputParser()
            )
            response = rag_chain.invoke(question)
        else:
            # Manually compose prompt for local LLM
            prompt_text = f"{self.system_message}\n"
            prompt_text += f"Context: {formatted_context}\n\n"
            if chat_history_str:
                prompt_text += f"Chat History:\n{chat_history_str}\n\n"
            prompt_text += f"Question: {question}"
            # Generate response using the local LLM (direct call)
            response = self.llm.invoke(prompt_text)  # assumes LLM.__call__ returns the generated string
            response = response.content if hasattr(response, "content") else str(response)


        # Save the new question/answer pair to memory (for conversation continuity)
        self.memory.save_context(inputs={"question": question}, outputs={"response": response})
        return response, context_list


class StatelessRAG:
    def __init__(self, vector_db, system_message=None, top_k=5, search_type="similarity", **llm_kwargs):
        """Stateless RAG (no conversation memory)."""
        self.vector_db = vector_db or FAISS.from_texts([""], OpenAIEmbeddings())
        self.system_message = system_message or "You are a helpful assistant."
        self.top_k = top_k
        self.search_type = search_type
        provided_llm = llm_kwargs.get("llm", None)
        self.llm = provided_llm if provided_llm is not None else ChatOpenAI(model="gpt-4o-mini")
        self.is_chat_model = isinstance(self.llm, BaseChatModel)

        self.retriever = self.vector_db.as_retriever(search_type=self.search_type, search_kwargs={"k": self.top_k})
        if self.is_chat_model:
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_message),
                ("human", "Context: {context}\n\nQuestion: {question}")
            ])
        else:
            self.prompt = None

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def format_docs_to_list(self, docs):
        return [doc.page_content for doc in docs]

    def invoke(self, question):
        """Processes a question through the RAG pipeline without memory."""
        context_docs = self.retriever.invoke(question)
        formatted_context = self.format_docs(context_docs)
        context_list = self.format_docs_to_list(context_docs)

        if self.is_chat_model:
            # Chain for chat LLM
            rag_chain = (
                { "context": RunnablePassthrough() | (lambda _: formatted_context),
                  "question": RunnablePassthrough() }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            answer = rag_chain.invoke(question)
        else:
            # Direct prompt for local LLM
            prompt_text = f"{self.system_message}\nContext: {formatted_context}\n\nQuestion: {question}"
            answer = self.llm.invoke(prompt_text)


        return answer, context_list


# For completeness, similar changes are applied to other RAG variants:

# For streamlit upload pdf usage (document retrieval)
class ConversationalPDFRAG:
    def __init__(self, vector_db, system_message=None, top_k=5, search_type="similarity", **llm_kwargs):
        """RAG model with memory for document-based QA (PDF)."""
        self.vector_db = vector_db or FAISS.from_texts([""], OpenAIEmbeddings())
        self.system_message = system_message or "You are a helpful assistant with memory. Answer questions accordingly."
        self.top_k = top_k
        self.search_type = search_type
        provided_llm = llm_kwargs.get("llm", None)
        self.llm = provided_llm if provided_llm is not None else ChatOpenAI(model="gpt-4o-mini")
        self.is_chat_model = isinstance(self.llm, BaseChatModel)
        self.retriever = self.vector_db.as_retriever(search_type=self.search_type, search_kwargs={"k": self.top_k})
        self.memory = ConversationBufferMemory(return_messages=True)
        # Define Prompt with Memory
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_message),
            ("human", "Context: {context}\n\nChat History: {chat_history}\n\nQuestion: {question}"),
        ])

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def format_docs_to_list(self, docs):
        return [[doc.page_content] for doc in docs]

    def invoke(self, question):
        """Process question through PDF RAG pipeline (includes printing sources)."""
        context_docs = self.retriever.invoke(question)
        formatted_context = self.format_docs(context_docs)
        history_msgs = self.memory.load_memory_variables({}).get("history", [])
        chat_history_str = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in history_msgs
        ])

        if self.is_chat_model:
            rag_chain = (
                { "context": RunnablePassthrough() | (lambda _: formatted_context),
                  "chat_history": RunnablePassthrough() | (lambda _: chat_history_str),
                  "question": RunnablePassthrough() }
                | self.prompt
                | self.llm
                | StrOutputParser()
            )
            answer = rag_chain.invoke(question)

        else:
            prompt_text = f"{self.system_message}\nContext: {formatted_context}\n\n"
            if chat_history_str:
                prompt_text += f"Chat History:\n{chat_history_str}\n\n"
            prompt_text += f"Question: {question}"
            answer = self.llm.invoke(prompt_text)


        # Print retrieved context documents (for transparency in PDF QA)
        print("🔎 Retrieved documents:")
        for i, doc in enumerate(context_docs):
            meta = getattr(doc, "metadata", {})  # assuming metadata exists for PDF pages
            print(f"  [{i+1}] {doc.page_content[:100]}... (source: {meta.get('source', 'N/A')})")

        self.memory.save_context(inputs={"question": question}, outputs={"response": answer})
        return answer, self.format_docs_to_list(context_docs)
    
# For prompting testing:
class PromptingRAG:
    def __init__(self, vector_db=None, top_k=5, search_type="similarity", pattern="rag-only ", **llm_kwargs):
        """
        Stateless RAG for prompting engineer
        """
        self.vector_db = vector_db or FAISS.from_texts([""], OpenAIEmbeddings())
        self.delimiter = "####"
        self.pattern = pattern
        self.top_k = top_k
        self.search_type = search_type
        provided_llm = llm_kwargs.get("llm", None)
        self.llm = provided_llm if provided_llm is not None else ChatOpenAI(model="gpt-4o-mini")
        self.is_chat_model = isinstance(self.llm, BaseChatModel)
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

