import streamlit as st
import sys
from pathlib import Path
import time 
import json
import uuid # For session IDs

# Add project root to sys.path to allow imports from src
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.interface.chatbot import LegalChatbot
from src.storage.database import DatabaseManager, DEFAULT_DB_PATH # Import DB Manager

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

# Helper function to get a DB connection within the current context
def get_db():
    # Note: Instantiating DatabaseManager here implicitly calls __enter__ which calls connect()
    # The 'with' block ensures __exit__ (and thus close()) is called.
    return DatabaseManager(DEFAULT_DB_PATH)

# --- Session State Management ---
# Ensure a unique session ID exists
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
session_id = st.session_state.session_id

# Initialize messages: Load from DB if possible, otherwise use default
if "messages" not in st.session_state:
    initial_messages = []
    try:
        with get_db() as db_manager: # Use context manager
            initial_messages = db_manager.get_chat_history(session_id)
    except Exception as e:
        st.warning(f"Could not load chat history: {e}")
        initial_messages = [] # Ensure it's defined even on error
        
    if not initial_messages:
        initial_messages = [{"role": "assistant", "content": "How can I help you query the legal documents today?"}]
        # Optionally save the initial assistant message to DB
        try:
            with get_db() as db_manager: # Use context manager
                 db_manager.add_chat_message(session_id, "assistant", initial_messages[0]["content"])
        except Exception as e:
            st.warning(f"Could not save initial assistant message: {e}")
            
    # Assign to session state *only* if it wasn't already there
    st.session_state.messages = initial_messages

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
        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Save user message to DB
        try:
            with get_db() as db_manager: # Use context manager
                db_manager.add_chat_message(session_id, user_message["role"], user_message["content"])
        except Exception as e:
            st.warning(f"Could not save user message: {e}")

        # Process the question and get the response
        with st.chat_message("assistant"):
            message_placeholder = st.empty() 
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

        # Add assistant response to session state and save to DB
        assistant_message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(assistant_message)
        try:
            with get_db() as db_manager: # Use context manager
                db_manager.add_chat_message(session_id, assistant_message["role"], assistant_message["content"])
        except Exception as e:
            st.warning(f"Could not save assistant message: {e}")

# --- Sidebar Information ---
st.sidebar.header("About")
st.sidebar.info(
    "This chatbot uses Retrieval-Augmented Generation (RAG) to answer questions "
    "based on the content of Chapter 39, Florida Statutes (2022-10-24 version). "
    "It utilizes a FAISS vector store for retrieval and an OpenAI language model for generation."
)
st.sidebar.header("How to Use")
st.sidebar.markdown(
    "1. Ensure you have run `python -m src.cli load data/raw` first.\n"
    "2. Type your question about Chapter 39 in the chat input box.\n"
    "3. The chatbot will retrieve relevant sections and generate an answer."
) 

# --- Sidebar: View Analysis Results ---
st.sidebar.divider() # Add a visual separator
st.sidebar.header("📊 View Analysis Results")

try:
    with get_db() as db_manager: # Use context manager
        documents = db_manager.get_all_documents() 

        if not documents:
            st.sidebar.info("No documents found in the database. Use the CLI 'load' command to add documents.")
        else:
            # Create a mapping from display path to doc_id
            # Use Path().name to show only filename in selectbox
            doc_options = {Path(path).name: doc_id for doc_id, path in documents}
            doc_paths_map = {Path(path).name: path for _, path in documents} # Map filename back to full path if needed
            
            selected_filename = st.sidebar.selectbox(
                "Select Document:",
                options=sorted(doc_options.keys()), # Sort filenames alphabetically
                key="analysis_doc_select" # Add unique key for widget state
            )

            if selected_filename:
                selected_doc_id = doc_options[selected_filename]
                selected_full_path = doc_paths_map[selected_filename] # Get the full path for display
                
                # Fetch analysis types using the same db_manager instance
                analysis_types = db_manager.get_analysis_types_for_doc(selected_doc_id)

                if not analysis_types:
                    st.sidebar.info(f"No analysis results found for '{selected_filename}'. Run `analyze {selected_full_path} --type <type>` in the CLI.")
                else:
                    selected_analysis_type = st.sidebar.selectbox(
                        "Select Analysis Type:",
                        options=analysis_types,
                        key="analysis_type_select" # Add unique key
                    )

                    if selected_analysis_type:
                        # Fetch result using the same db_manager instance
                        result_json_str = db_manager.get_analysis_result(selected_doc_id, selected_analysis_type)

                        if result_json_str:
                            try:
                                result_data = json.loads(result_json_str)
                                st.sidebar.caption(f"Showing '{selected_analysis_type}' for '{selected_filename}':")
                                st.sidebar.json(result_data, expanded=False) # Show JSON, initially collapsed
                            except json.JSONDecodeError as e:
                                st.sidebar.error(f"Failed to parse stored result: {e}")
                                st.sidebar.code(result_json_str) # Show the raw string if parsing failed
                        else:
                            st.sidebar.warning("Could not retrieve the analysis result.")
except Exception as e:
    st.sidebar.error(f"Database error displaying analysis results: {e}")
    st.sidebar.info("Database connection may not be available.") 