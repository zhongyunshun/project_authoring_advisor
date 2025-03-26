import streamlit as st
import os

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

def upload_files_form():
    uploaded_files = st.file_uploader("Choose files", type=["txt", "pdf"], accept_multiple_files=True)

    if uploaded_files is not None:
        # Display details of each uploaded file
        for uploaded_file in uploaded_files:
            st.write(f"File name: {uploaded_file.name}")
            st.write(f"File type: {uploaded_file.type}")
            st.write(f"File size: {uploaded_file.size} bytes")

    # TODO: implement file processing later
    if st.button("Upload"):
        st.write("TODO: implement file processing later")
        # Create Embedding for given file and concat to previous embeddings

# Check if API key is available in session state
if "openai_api_key" not in st.session_state or not st.session_state.openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
else:
    upload_files_form()
