import os
import streamlit as st
from pipelines.rag_pipeline import ConversationalRAG
from embeddings.embeddings import load_embeddings_from_file
from streamlit_class.conversations import Conversation

# Sidebar for OpenAI API Key
with st.sidebar:
    # Check if API key is already in session state
    if "openai_api_key" not in st.session_state or not st.session_state.openai_api_key:
        st.session_state.openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
    else:
        # If the API key is already in session, use the saved value
        st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, type="password", key="chatbot_api_key")
    
    st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")

# Set OpenAI API Key in environment if available
if "openai_api_key" in st.session_state and st.session_state.openai_api_key:
    os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key

# Embedding and RAG Model Setup
# embedding_file = "vector_db/openai_chunk_700_embedding"  # can change the chunk size if needed

# # Setup Model and Embedding instead of repeatedly loading
# # Load only if API key is given
# if "vector_db" not in st.session_state and "openai_api_key" in st.session_state and st.session_state.openai_api_key:
#     if os.path.exists(embedding_file):
#         print(f"📂 Loading existing embeddings from {embedding_file}...\n")
#         vector_db = load_embeddings_from_file(embedding_file)
#         st.session_state.vector_db = vector_db
#         print("✅ Embeddings loaded from file.\n")
#     else:
#         st.error("Embedding file not found. Exiting Streamlit.")
#         st.stop()

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "rag_model" not in st.session_state and "openai_api_key" in st.session_state and st.session_state.openai_api_key:
    print(f"🤖 Loading ConversationalRAG Model...\n")
    system_message_chat = "You are a helpful assistant with memory. Answer questions accordingly."
    rag_model = ConversationalRAG(
        st.session_state.vector_db, system_message_chat, top_k=22, search_type="similarity"
    )
    st.session_state.rag_model = rag_model

# init variables in session state
if "title" not in st.session_state:
    st.session_state.title = "TRCA ChatBot"

if "conversations" not in st.session_state:
    st.session_state.conversations = []

if len(st.session_state.conversations) == 0:
    # Initialize with the first conversation as default
    initial_chat_history = [{"role": "assistant", "content": "This is a ChatBot designed for TRCA. How can I help you?"}]
    st.session_state.conversations.append(Conversation(1, "Conversation 1", initial_chat_history))

# Initialize chat history if not exists in session_state
if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = st.session_state.conversations[0]
    st.session_state.title = st.session_state.current_conversation.title

# Sidebar: Display previous chat history as clickable titles and New Conversation button
with st.sidebar:
    st.subheader("Chat History")

    # Display and manage previous chat sessions
    for conv in st.session_state.conversations:
        if st.button(conv.title):
            st.session_state.current_conversation = conv
            st.session_state.title = conv.title

    # New conversation button
    if st.button("New Conversation"):
        new_session_id = len(st.session_state.conversations) + 1
        new_conversation = Conversation(new_session_id, f"Conversation {new_session_id}", [
            {"role": "assistant", "content": "This is a ChatBot designed for TRCA. How can I help you?"}
        ])
        st.session_state.conversations.append(new_conversation)
        st.session_state.current_conversation = new_conversation
        st.session_state.title = new_conversation.title
        st.rerun()

# Streamlit UI Starts here
st.title(st.session_state.title)

# Allow the user to edit the conversation title
if st.session_state.current_conversation:
    new_title = st.text_input("Edit Conversation Title", value=st.session_state.current_conversation.title)
    if new_title != st.session_state.current_conversation.title:
        st.session_state.current_conversation.title = new_title
        st.session_state.title = new_title
        st.rerun()

# Display chat history and context list
for msg in st.session_state.current_conversation.chat_history:
    st.chat_message(msg["role"]).write(msg["content"])

    if msg["role"] == "assistant" and "context" in msg:
        with st.expander("🔍 Context Used for This Response"):
            for i, doc in enumerate(msg["context"]):
                if i < 5: # print first 5
                    st.markdown(f"**Document {i + 1}**")
                    st.markdown(f"*Content:* `{doc.page_content[:]}`")
                    st.markdown("*Metadata:*")
                    for key, value in doc.metadata.items():
                        if key == "page":
                            st.markdown(f"- **{key}**: {value + 1}") # page count start with 0 instead of 1
                        else:
                            st.markdown(f"- **{key}**: {value}")

if prompt := st.chat_input():
    if "openai_api_key" not in st.session_state or not st.session_state.openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    # Append user message and responses to chat history
    st.session_state.current_conversation.chat_history.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    response, context = st.session_state.rag_model.invoke(prompt)

    st.session_state.current_conversation.chat_history.append({"role": "assistant", "content": response, "context": context})
    st.chat_message("assistant").write(response)

    # Show context documents
    with st.expander("🔍 Context Used for Response"):
        for i, doc in enumerate(context):
            if i < 5: # print first 5
                st.markdown(f"**Document {i + 1}**")
                st.markdown(f"*Content:* `{doc.page_content[:]}`")
                st.markdown("*Metadata:*")
                for key, value in doc.metadata.items():
                    if key == "page":
                        st.markdown(f"- **{key}**: {value + 1}") # page count start with 0 instead of 1
                    else:
                        st.markdown(f"- **{key}**: {value}")

# Run Streamlit with: `streamlit run StreamLitUI.py`
