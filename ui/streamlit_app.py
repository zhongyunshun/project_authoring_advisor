"""Main Streamlit chat UI for the TRCA RAG system.

Replaces StreamLitUI.py with LlamaIndex + Qdrant backend.
"""

import os
import streamlit as st

from config.settings import Settings
from core.llm_factory import LLMFactory
from core.embedding_factory import EmbeddingFactory
from core.vector_store import VectorStoreManager
from ingest.indexer import Indexer
from pipeline.rag_engine import RAGEngine
from agents.rag_agent import RAGAgent
from agents.tools import create_document_query_tool, create_web_search_tool
from streamlit_class.conversations import Conversation


@st.cache_resource
def get_vector_store_manager(storage_path: str = "./vector_db/qdrant_storage") -> VectorStoreManager:
    """Qdrant local mode allows one client per process per storage path.
    Cache the manager so the chat page and upload page share it."""
    return VectorStoreManager(storage_path=storage_path)


@st.cache_resource
def _pdf_path_index(data_dir: str = "data") -> dict:
    """Walk the data folder once, map filename -> absolute path so chunks can resolve
    back to a clickable PDF even when only `source` (filename) is in metadata."""
    index: dict = {}
    if not os.path.isdir(data_dir):
        return index
    for root, _dirs, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                index.setdefault(f, os.path.join(root, f))
    return index


def _resolve_pdf_path(metadata: dict) -> str:
    """Best-effort resolution of an absolute PDF path from a chunk's metadata."""
    fp = metadata.get("file_path")
    if fp and os.path.isfile(fp):
        return fp
    src = metadata.get("source")
    if src:
        return _pdf_path_index().get(src, "")
    return ""


@st.cache_data
def _read_pdf_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _render_sources(sources: list, key_prefix: str):
    """Render the collapsible chunk list with PDF download buttons."""
    for i, src in enumerate(sources[:5]):
        md = src.get("metadata") or {}
        src_name = md.get("source", "unknown")
        page = md.get("page")
        if isinstance(page, int):
            page_display = f"page {page + 1}"
        else:
            page_display = "page n/a (re-index required)"

        st.markdown(f"**Document {i + 1} — `{src_name}` ({page_display})**")

        pdf_path = _resolve_pdf_path(md)
        if pdf_path:
            try:
                st.download_button(
                    label=f"Open {src_name}",
                    data=_read_pdf_bytes(pdf_path),
                    file_name=src_name,
                    mime="application/pdf",
                    key=f"{key_prefix}_pdf_{i}",
                )
            except Exception as e:
                st.caption(f"PDF unavailable: {e}")
        else:
            st.caption("PDF file not found in `data/`.")

        st.markdown(f"*Content:* `{src['text']}`")

        if md:
            st.markdown("*Metadata:*")
            for k, v in md.items():
                display_val = v + 1 if k == "page" and isinstance(v, int) else v
                st.markdown(f"- **{k}**: {display_val}")


def init_sidebar():
    """Render the sidebar: API key, model selection, chat history."""
    with st.sidebar:
        # API key
        if "openai_api_key" not in st.session_state or not st.session_state.openai_api_key:
            st.session_state.openai_api_key = st.text_input(
                "OpenAI API Key", key="chatbot_api_key", type="password"
            )
        else:
            st.text_input(
                "OpenAI API Key",
                value=st.session_state.openai_api_key,
                type="password",
                key="chatbot_api_key",
            )

        # Model selection
        model_choice = st.selectbox(
            "LLM Provider",
            ["openai", "gemini", "claude"],
            key="llm_provider_select",
        )
        st.session_state.setdefault("llm_provider", model_choice)
        if model_choice != st.session_state.get("llm_provider"):
            st.session_state.llm_provider = model_choice
            st.session_state.pop("rag_engine", None)  # force rebuild

        # Specific model within provider
        model_options = Settings().AVAILABLE_MODELS.get(model_choice, [])
        if model_options:
            model_name = st.selectbox("Model", model_options, key="model_name_select")
            st.session_state["model_name"] = model_name
            if model_name != st.session_state.get("_prev_model_name"):
                st.session_state["_prev_model_name"] = model_name
                st.session_state.pop("rag_engine", None)

        # Agentic mode toggle
        use_agent = st.checkbox("Agentic Mode (auto web search)", key="use_agent")
        if use_agent != st.session_state.get("_prev_use_agent"):
            st.session_state["_prev_use_agent"] = use_agent
            st.session_state.pop("rag_engine", None)  # force rebuild on toggle

        st.divider()

        # Chat history
        st.subheader("Chat History")
        for conv in st.session_state.get("conversations", []):
            if st.button(conv.title, key=f"conv_{conv.session_id}"):
                st.session_state.current_conversation = conv
                st.session_state.title = conv.title

        if st.button("New Conversation"):
            new_id = len(st.session_state.conversations) + 1
            new_conv = Conversation(new_id, f"Conversation {new_id}", [
                {"role": "assistant", "content": "This is a ChatBot designed for TRCA. How can I help you?"}
            ])
            st.session_state.conversations.append(new_conv)
            st.session_state.current_conversation = new_conv
            st.session_state.title = new_conv.title
            st.rerun()


def init_engine():
    """Initialize or retrieve the RAG engine from session state."""
    if "openai_api_key" in st.session_state and st.session_state.openai_api_key:
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key

    # Load all keys from .env
    settings = Settings.from_env()
    settings.apply_env()

    if "rag_engine" in st.session_state:
        return st.session_state.rag_engine

    # Need at least one API key to proceed
    has_key = any([
        os.environ.get("OPENAI_API_KEY"),
        os.environ.get("GOOGLE_API_KEY"),
        os.environ.get("ANTHROPIC_API_KEY"),
    ])
    if not has_key:
        return None

    provider = st.session_state.get("llm_provider", "openai")
    model_name = st.session_state.get("model_name", "")
    llm = LLMFactory.create(provider=provider, model=model_name)
    embed_model = EmbeddingFactory.create(provider="huggingface")

    vsm = get_vector_store_manager()
    collection_name = "trca_documents"

    if vsm.collection_exists(collection_name):
        indexer = Indexer(vsm, embed_model)
        index = indexer.load_index(collection_name)
    else:
        # No pre-indexed data yet; create an empty-ish state
        # The user should upload PDFs or run indexing via CLI
        st.session_state.rag_engine = None
        return None

    if st.session_state.get("use_agent"):
        tools = [create_document_query_tool(index, llm)]
        if os.environ.get("TAVILY_API_KEY"):
            tools.append(create_web_search_tool())
            st.sidebar.success("Agent enabled with web search (Tavily)")
        else:
            st.sidebar.warning("Agent enabled, but TAVILY_API_KEY missing — web search disabled")
        engine = RAGAgent(tools=tools, llm=llm, conversational=True)
    else:
        engine = RAGEngine(
            index=index,
            llm=llm,
            system_prompt="You are a helpful assistant with memory. Answer questions accordingly.",
            top_k=22,
            conversational=True,
        )

    st.session_state.rag_engine = engine
    return engine


def render_chat():
    """Render the chat interface."""
    conv = st.session_state.current_conversation

    st.title(st.session_state.title)

    # Editable title
    new_title = st.text_input("Edit Conversation Title", value=conv.title)
    if new_title != conv.title:
        conv.title = new_title
        st.session_state.title = new_title
        st.rerun()

    # Display chat history
    for msg_idx, msg in enumerate(conv.chat_history):
        st.chat_message(msg["role"]).write(msg["content"])

        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander("Context Used for This Response"):
                _render_sources(
                    msg["sources"],
                    key_prefix=f"hist_{conv.session_id}_{msg_idx}",
                )

    # Chat input
    if prompt := st.chat_input():
        engine = st.session_state.get("rag_engine")
        if not engine:
            st.info("Please add your OpenAI API key and ensure documents are indexed.")
            st.stop()

        conv.chat_history.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        result = engine.query(prompt)

        sources_data = [
            {"text": s.text, "metadata": s.metadata, "score": s.score}
            for s in result.sources
        ]
        conv.chat_history.append({
            "role": "assistant",
            "content": result.answer,
            "sources": sources_data,
        })
        st.chat_message("assistant").write(result.answer)

        if sources_data:
            with st.expander("Context Used for Response"):
                _render_sources(
                    sources_data,
                    key_prefix=f"new_{conv.session_id}_{len(conv.chat_history)}",
                )


def main():
    st.set_page_config(page_title="TRCA ChatBot", layout="wide")

    # Init session state
    if "conversations" not in st.session_state:
        st.session_state.conversations = []

    if not st.session_state.conversations:
        initial = Conversation(1, "Conversation 1", [
            {"role": "assistant", "content": "This is a ChatBot designed for TRCA. How can I help you?"}
        ])
        st.session_state.conversations.append(initial)

    if "current_conversation" not in st.session_state:
        st.session_state.current_conversation = st.session_state.conversations[0]
        st.session_state.title = st.session_state.conversations[0].title

    init_sidebar()
    init_engine()
    render_chat()


if __name__ == "__main__":
    main()
