"""Streamlit subpage for PDF upload and on-the-fly indexing.

Replaces Pages/2_Upload_Files_PDF.py.
Uses Qdrant incremental upsert (no merge needed).
"""

import os
import streamlit as st

from core.llm_factory import LLMFactory
from core.embedding_factory import EmbeddingFactory
from ingest.indexer import Indexer
from pipeline.rag_engine import RAGEngine
from agents.rag_agent import RAGAgent
from agents.tools import create_document_query_tool
from ui.streamlit_app import get_vector_store_manager


st.title("Upload Files")

# Sidebar API key
with st.sidebar:
    if "openai_api_key" not in st.session_state or not st.session_state.openai_api_key:
        st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password")
    else:
        st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, type="password")

if "openai_api_key" in st.session_state and st.session_state.openai_api_key:
    os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key

# Init state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

COLLECTION_NAME = "pdf_uploads"


def upload_files_form():
    uploaded_files = st.file_uploader(
        "Choose PDF files", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        embed_model = EmbeddingFactory.create(provider="openai")
        vsm = get_vector_store_manager()
        indexer = Indexer(vsm, embed_model)

        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_files:
                st.session_state.uploaded_files.append(uploaded_file.name)
                index = indexer.index_uploaded_file(uploaded_file, COLLECTION_NAME)
                st.success(f"{uploaded_file.name} uploaded and indexed.")

        # Rebuild the engine with the updated index
        provider = st.session_state.get("llm_provider", "openai")
        llm = LLMFactory.create(provider=provider)
        index = indexer.load_index(COLLECTION_NAME)

        if st.session_state.get("use_agent"):
            tools = [create_document_query_tool(index, llm)]
            st.session_state.rag_engine = RAGAgent(tools=tools, llm=llm, conversational=True)
        else:
            st.session_state.rag_engine = RAGEngine(
                index=index,
                llm=llm,
                system_prompt="You are a helpful assistant with memory. Answer questions accordingly.",
                top_k=22,
                conversational=True,
            )

        st.success("All files uploaded and processed!")


# Display uploaded files
st.subheader("Uploaded Files")
if st.session_state.uploaded_files:
    for fname in st.session_state.uploaded_files:
        st.write(f"  {fname}")
else:
    st.info("No files uploaded yet.")

if not os.environ.get("OPENAI_API_KEY"):
    st.info("Please add your OpenAI API key to continue.")
else:
    upload_files_form()
