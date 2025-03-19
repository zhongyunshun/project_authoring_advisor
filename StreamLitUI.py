import os
import streamlit as st
from pipelines.rag_pipeline import ConversationalRAG
from embeddings.embeddings import load_embeddings_from_file

# Sidebar for OpenAI API Key
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")

# Set OpenAI API Key
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# Embedding and RAG Model Setup
embedding_file = "vector_db/openai_chunk_700_embedding"  # can change the chunk size if needed

# Setup Model and Embedding instead of repeatively loading
# Load only if api key is given
if "vector_db" not in st.session_state and openai_api_key:
    if os.path.exists(embedding_file):
        print(f"📂 Loading existing embeddings from {embedding_file}...\n")
        vector_db = load_embeddings_from_file(embedding_file)
        st.session_state.vector_db = vector_db
        print("✅ Embeddings loaded from file.\n")
    else:
        st.error("Embedding file not found. Exiting Streamlit.")
        st.stop()

if "rag_model" not in st.session_state and openai_api_key:
    print(f"🤖 Loading ConversationalRAG Model...\n")
    system_message_chat = "You are a helpful assistant with memory. Answer questions accordingly."
    rag_model = ConversationalRAG(
        st.session_state.vector_db, system_message_chat, top_k=22, search_type="similarity"
    )
    st.session_state.rag_model = rag_model

# Streamlit UI
st.title("💬 Conversational RAG Chatbot")
st.caption("🚀 A Streamlit chatbot powered by OpenAI")

# Initialize chat history if not exists in session_state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "This is a ChatBot designed for TRCA. How can I help you?"}]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # Append user message and responses to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    response, _ = st.session_state.rag_model.invoke(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

# Run Streamlit with: `streamlit run StreamLitUI.py`
