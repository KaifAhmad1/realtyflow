import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# Import the LangGraph app and state management functions
# Ensure chatbot_engine.py is in src and PYTHONPATH is set, or Streamlit handles it.
try:
    from src.chatbot_engine import chatbot_app, create_initial_state, log_interaction
except ImportError as e:
    st.error(f"Failed to import chatbot engine. Ensure 'src' directory is accessible: {e}")
    st.stop() # Stop app execution if core components can't be loaded
except Exception as e: # Catch other init errors from chatbot_engine (like API key)
    st.error(f"Error initializing chatbot components: {e}")
    st.stop()


st.set_page_config(page_title="RealtyFlow", page_icon="🏠", layout="wide")

# --- Page Styling ---
st.markdown("""
<style>
    .stApp {
        background_color: #f0f2f6;
    }
    .stChatInputContainer > div { /* Target the inner div for background */
        background-color: #FFFFFF;
    }
    [data-testid="stChatMessageContent"] {
        border-radius: 10px;
        padding: 10px 12px; /* More padding */
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        line-height: 1.6; /* Better readability */
    }
    [data-testid="stChatMessageContent"] p {
        margin: 0;
        line-height: 1.6;
    }
    [data-testid="stSidebar"] {
        background-color: #e8eef4;
        padding: 15px; /* Add some padding to sidebar content */
    }
    .stButton>button {
        border-radius: 8px;
        border: 1px solid #007bff;
        background-color: #007bff;
        color: white;
        padding: 8px 16px; /* Better button padding */
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
        border: 1px solid #0056b3;
    }
    .stChatInput { /* Ensure input field is visible */
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)


# --- Chatbot State Initialization ---
if "chat_state" not in st.session_state:
    if 'chatbot_app' not in globals() or chatbot_app is None:
        st.error("Chatbot application is not available. Please check server logs.")
        # Create a minimal state to prevent further errors
        st.session_state.chat_state = {
            "messages": [AIMessage(content="Chatbot is currently unavailable.")],
            "conversation_ended": True,
            "interaction_history": [] # Ensure this key exists
        }
    else:
        st.session_state.chat_state = create_initial_state()
        try:
            # Invoke the app to get the initial greeting
            st.session_state.chat_state = chatbot_app.invoke(
                st.session_state.chat_state,
                {"recursion_limit": 50} # Good to set a limit
            )
        except Exception as e:
            print(f"ERROR: Initializing chatbot via invoke: {e}") # Log for server admin
            st.session_state.chat_state["messages"].append(AIMessage(content="Sorry, an error occurred during startup. Please try refreshing."))
            st.session_state.chat_state["conversation_ended"] = True
            if "interaction_history" not in st.session_state.chat_state:
                 st.session_state.chat_state["interaction_history"] = []
            log_interaction(st.session_state.chat_state, "streamlit_init_invoke_error", details={"error": str(e)})


# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/plasticine/100/000000/real-estate.png", width=100, caption="RealtyFlow")
    st.title("RealtyFlow")
    st.markdown("Your Intelligent Real Estate Assistant")
    
    if st.button("🔄 Restart Conversation", use_container_width=True, key="restart_button_main"):
        if 'chatbot_app' not in globals() or chatbot_app is None:
            st.error("Cannot restart: Chatbot application is not available.")
        else:
            old_session_id = st.session_state.chat_state.get("session_id", str(uuid.uuid4()))
            old_history = st.session_state.chat_state.get("interaction_history", [])

            st.session_state.chat_state = create_initial_state()
            st.session_state.chat_state["session_id"] = old_session_id
            st.session_state.chat_state["interaction_history"] = old_history
            log_interaction(st.session_state.chat_state, "user_restart_streamlit_button")

            try:
                st.session_state.chat_state = chatbot_app.invoke(st.session_state.chat_state, {"recursion_limit": 50})
            except Exception as e:
                print(f"ERROR: Restarting chatbot invoke: {e}")
                st.session_state.chat_state["messages"].append(AIMessage(content="Sorry, couldn't restart properly. Please try again."))
            st.rerun()

    st.markdown("---")
    st.caption("© 2024 RealtyFlow. All rights reserved.")

# --- Main Chat Interface ---
st.header("Chat with RealtyFlow 💬")

# Display chat messages
messages_to_display = st.session_state.get("chat_state", {}).get("messages", [])
if not messages_to_display and not st.session_state.get("chat_state", {}).get("conversation_ended", False):
    # This case should ideally be covered by the initial invoke adding a greeting.
    # If it happens, it might indicate an issue with the initial state setup.
    st.info("Welcome to RealtyFlow! How can I assist you today?")


for msg in messages_to_display:
    if isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg.content, unsafe_allow_html=True)
    elif isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="🧑"):
            st.markdown(msg.content, unsafe_allow_html=True)

# User input
is_chat_disabled = st.session_state.get("chat_state", {}).get("conversation_ended", False) or \
                   ('chatbot_app' not in globals() or chatbot_app is None)

input_prompt = "Your message..."
if is_chat_disabled:
    if not ('chatbot_app' not in globals() or chatbot_app is None): # If ended but app is fine
        input_prompt = "Conversation ended. Please restart."
    else: # If app itself is not loaded
        input_prompt = "Chatbot is unavailable."


user_input = st.chat_input(input_prompt, disabled=is_chat_disabled, key="chat_input_main")

if user_input and not is_chat_disabled:
    st.session_state.chat_state["messages"].append(HumanMessage(content=user_input))
    
    try:
        st.session_state.chat_state = chatbot_app.invoke(st.session_state.chat_state, {"recursion_limit": 50})
    except Exception as e:
        print(f"ERROR: Invoking chatbot with user input '{user_input}': {e}")
        error_msg = "I'm sorry, an unexpected issue occurred. Please try again or restart."
        st.session_state.chat_state["messages"].append(AIMessage(content=error_msg))
        st.session_state.chat_state["last_error"] = str(e) # Store last error in state
        log_interaction(st.session_state.chat_state, "streamlit_invoke_error_user_input", details={"error": str(e)})
    
    st.rerun()
elif is_chat_disabled and not messages_to_display:
    st.warning("Chat is currently unavailable or has ended. Try restarting if the button is active.")
