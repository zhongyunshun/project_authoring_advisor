"""
Subpage 1 for Streamlit UI
Upload files and process files into embeddings
"""

import streamlit as st
import os
from pipeline_preprocess_file_exec.pdf_processing import generate_embeddings_from_single_pdf
from embeddings.embeddings import merge_faiss_vector_dbs
from pipelines.rag_pipeline import ConversationalRAG

st.title("📁 Upload Files")

with st.sidebar:
    if "openai_api_key" not in st.session_state or not st.session_state.openai_api_key:
        st.session_state.openai_api_key = st.text_input("OpenAI API Key", type="password")
    else:
        st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, type="password")
    
    st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")

# Set OpenAI API Key in environment if available
if "openai_api_key" in st.session_state and st.session_state.openai_api_key:
    os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key

# Initialize session state for uploaded files and embeddings
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

def upload_files_form():
    uploaded_files = st.file_uploader("Choose files", type=["pdf"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_files:
                # Save filename in session state
                st.session_state.uploaded_files.append(uploaded_file.name)

                # Generate embeddings and append to memory
                embeddings = generate_embeddings_from_single_pdf(uploaded_file)
                
                prev_db = st.session_state.vector_db
                st.session_state.vector_db = merge_faiss_vector_dbs(prev_db, embeddings)

                st.success(f"✅ {uploaded_file.name} uploaded and processed.")
        
        # Reload model after loop
        system_message_chat = "You are a helpful assistant with memory. Answer questions accordingly."
        rag_model = ConversationalRAG(
            st.session_state.vector_db, system_message_chat, top_k=22, search_type="similarity"
        )
        st.session_state.rag_model = rag_model

# Display uploaded files
st.subheader("Uploaded Files")

if st.session_state.uploaded_files:
    for file_name in st.session_state.uploaded_files:
        st.write(f"📄 {file_name}")
else:
    st.info("No files uploaded yet.")

if "openai_api_key" not in st.session_state or not st.session_state.openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
else:
    upload_files_form()
