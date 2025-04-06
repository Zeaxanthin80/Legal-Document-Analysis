import streamlit as st
import sys
from pathlib import Path
import time 

# Add project root to sys.path to allow imports from src
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.interface.chatbot import LegalChatbot

# --- Page Configuration ---
st.set_page_config(
    page_title="Legal Document Assistant",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Legal Document Assistant")
st.caption("Query information from Chapter 39, Florida Statutes")

# --- Initialization ---
@st.cache_resource # Cache the chatbot instance for efficiency
def load_chatbot():
    try:
        chatbot_instance = LegalChatbot() # Uses default INDEX_PATH
        return chatbot_instance
    except ValueError as e:
        st.error(f"Error initializing chatbot: {e}. Please ensure the index has been created using the CLI 'load' command.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during initialization: {e}")
        return None

chatbot = load_chatbot()

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you query the legal documents today?"}]

# --- Chat Interface ---
# Display existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new user input
if prompt := st.chat_input("Ask a question about Chapter 39..."):
    if not chatbot:
        st.error("Chatbot is not initialized. Cannot process the request.")
    else:
        # Add user message to session state and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the question and get the response
        with st.chat_message("assistant"):
            message_placeholder = st.empty() # Placeholder for the streaming-like effect
            full_response = ""
            message_placeholder.markdown("Thinking... ⏳")
            
            try:
                # Get response from the chatbot logic
                result = chatbot.ask(prompt) # Use the existing ask method
                answer = result.get("answer", "Sorry, I couldn't find an answer.")
                sources = result.get("source_documents")

                # Format the response with sources
                full_response = answer
                if sources:
                    full_response += "\n\n**Sources:**\n"
                    unique_sources = set()
                    for doc in sources:
                        source_name = doc.metadata.get('source', 'Unknown')
                        if source_name not in unique_sources:
                             # Only display the filename, not the full path
                            display_source = Path(source_name).name
                            full_response += f"* {display_source}\n"
                            unique_sources.add(source_name)

                # Simulate streaming effect
                message_placeholder.empty() # Clear the 'Thinking...' message
                # Write the final response character by character for effect
                # for chunk in full_response.split():
                #     message_placeholder.markdown(full_response)
                #     time.sleep(0.05)
                message_placeholder.markdown(full_response) # Display full response at once

            except Exception as e:
                full_response = f"An error occurred: {e}"
                message_placeholder.error(full_response)

        # Add assistant response to session state
        st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- Sidebar Information (Optional) ---
st.sidebar.header("About")
st.sidebar.info(
    "This chatbot uses Retrieval-Augmented Generation (RAG) to answer questions "
    "based on the content of Chapter 39, Florida Statutes (2022-10-24 version). "
    "It utilizes a FAISS vector store for retrieval and an OpenAI language model for generation."
)
st.sidebar.header("How to Use")
st.sidebar.markdown(
    "1. Ensure you have run `python -m src.cli load data/raw/Chapter\ 39\ 2022-10-24.pdf` first.\n"
    "2. Type your question about Chapter 39 in the chat input box below.\n"
    "3. The chatbot will retrieve relevant sections and generate an answer."
) 