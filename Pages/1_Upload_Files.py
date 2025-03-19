import streamlit as st

st.title("📁 Upload Files")

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")

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

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
else:
    upload_files_form()
