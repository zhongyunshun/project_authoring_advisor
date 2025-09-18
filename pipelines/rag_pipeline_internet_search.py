from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.docstore.document import Document
# Search wrappers (only the one you plan to use is required at runtime)
# Tavily:
from langchain_community.tools.tavily_search import TavilySearchResults
# Serper:
from langchain_community.utilities import GoogleSerperAPIWrapper
# Bing:
from langchain_community.utilities import BingSearchAPIWrapper
def web_search_and_load(query: str, k: int = 3, provider: str = "tavily"):
    """
    Returns a list of LangChain Documents fetched from the open web.
    Provider options: 'tavily' (default), 'serper', 'bing'.
    """
    try:
        urls = []

        if provider == "tavily":
            # Requires TAVILY_API_KEY
            search = TavilySearchResults(k=k)
            results = search.invoke(query)  # [{'url':..., 'content':...}, ...]
            urls = [r["url"] for r in results if "url" in r][:k]

        elif provider == "serper":
            # Requires SERPER_API_KEY
            serper = GoogleSerperAPIWrapper()
            results = serper.results(query)  # dict with 'organic'
            urls = [o["link"] for o in results.get("organic", [])[:k] if "link" in o]

        elif provider == "bing":
            # Requires BING_SUBSCRIPTION_KEY
            bing = BingSearchAPIWrapper()
            # returns list of {'title':..., 'link':..., 'snippet':...}
            urls = [r["link"] for r in bing.results(query, top_k=k) if "link" in r]

        if not urls:
            return []

        loader = WebBaseLoader(urls)
        docs = loader.load()
        # Tag source so it’s visible in downstream UIs
        for d in docs:
            d.metadata["source_type"] = "web"
        return docs

    except Exception:
        # Fail safe: if web search fails, just return empty
        return []
class ConversationalRAG:
    def __init__(self, vector_db, system_message=None, top_k=5, search_type="similarity",
                 use_web=False, web_provider="tavily", web_k=3, rerank_with_embeddings=True):
        """
        Initializes the Conversational RAG model with memory storage.
        """
        self.use_web = use_web
        self.web_provider = web_provider
        self.web_k = web_k
        self.rerank_with_embeddings = rerank_with_embeddings
        ...
        # Keep your existing retriever:
        self.retriever = self.vector_db.as_retriever(search_type=self.search_type,
                                                     search_kwargs={"k": self.top_k})
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
        # 1) Local retrieval
        local_docs = self.retriever.invoke(question)

        # 2) Web retrieval (optional)
        web_docs = web_search_and_load(question, k=self.web_k, provider=self.web_provider) if self.use_web else []

        # 3) Merge + optional re-rank with embeddings against the query
        merged_docs = list(local_docs) + list(web_docs)

        if self.rerank_with_embeddings and len(merged_docs) > 0:
            emb = OpenAIEmbeddings()
            temp_vs = FAISS.from_documents(merged_docs, emb)
            # Re-rank to a reasonable final budget (local k + web k)
            final_k = min(len(merged_docs), self.top_k + self.web_k)
            reranked = temp_vs.similarity_search(question, k=final_k)
            docs_for_context = reranked
        else:
            # Keep original order
            docs_for_context = merged_docs

        formatted_context = self.format_docs(docs_for_context)
        context_list = self.format_docs_to_list(docs_for_context)

        # Build chat history as you already do
        chat_history_objects = self.memory.load_memory_variables({}).get("history", [])
        chat_history = "\n".join([
            f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
            for msg in chat_history_objects
        ])

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

        response = rag_chain.invoke(question)

        self.memory.save_context(inputs={"question": question}, outputs={"response": response})
        return response, context_list

'''
# Usage example
# Local-only
rag = ConversationalRAG(vector_db=my_faiss, top_k=5)

# Local + Web (Tavily), re-rank combined context
rag_web = ConversationalRAG(
    vector_db=my_faiss,
    top_k=5,
    use_web=True,
    web_provider="tavily",
    web_k=3,
    rerank_with_embeddings=True
)

answer, ctx_docs = rag_web.invoke("What permits are required for in-water works near migratory fish habitat?")
print(answer)
# ctx_docs includes both local DB chunks and web pages (with metadata['source_type'] == 'web'])

'''

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
