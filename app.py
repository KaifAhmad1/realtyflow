import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import time # For simulating thinking (though not actively used for typing effect here)
import uuid

# Import the LangGraph app and state management functions
try:
    from src.chatbot_engine import chatbot_app, create_initial_state, log_interaction
except ImportError as e:
    st.error(f"Failed to import chatbot engine. Ensure 'src' directory is accessible: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error initializing chatbot components: {e}")
    st.stop()


st.set_page_config(page_title="RealtyFlow", page_icon="🏠", layout="centered")

# --- Dark Theme Styling ---
st.markdown("""
<style>
    /* Overall App Background & Font */
    .stApp {
        background-color: #1E1E1E; /* Dark charcoal background */
        color: #E0E0E0; /* Light grey text */
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
    }

    /* Main content container */
    .main .block-container {
        max-width: 750px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Chat Input Area */
    .stChatInputContainer > div {
        background-color: #2A2A2A; /* Slightly lighter dark for input container */
        border-top: 1px solid #444444;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.2);
    }
    .stChatInput textarea { /* The actual text input field */
        background-color: #333333 !important; /* Darker input field */
        color: #F0F0F0 !important; /* Bright text for input */
        border: 1px solid #555555 !important;
        border-radius: 8px !important;
    }
    .stChatInput textarea::placeholder { /* Placeholder text color */
        color: #AAAAAA !important;
    }
    /* Send button in chat input - if you want to style it */
    .stChatInput button {
        background-color: #00A9E0 !important; /* Bright cyan for send */
        color: white !important;
        border-radius: 8px !important;
    }
    .stChatInput button:hover {
        background-color: #008CBA !important;
    }


    /* Message Bubbles Styling */
    [data-testid="stChatMessageContent"] {
        border-radius: 18px;
        padding: 14px 20px; /* Slightly more padding */
        box-shadow: 0 3px 8px rgba(0,0,0,0.25);
        line-height: 1.65;
        word-wrap: break-word;
    }
    [data-testid="stChatMessageContent"] p {
        margin: 0;
        line-height: 1.65;
    }
    [data-testid="stChatMessageContent"] strong { /* Make bold text more prominent */
        color: #FFFFFF; /* White for bold text within bubbles */
    }

    /* AI Message Bubble - Using a vibrant accent */
    div[data-testid="stChatMessage"]:has(div[data-testid="stAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #00A9E0, #0074D9); /* Cyan to Blue gradient */
        color: white;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] a {
        color: #B3E5FC; /* Light cyan for links in AI bubble */
        text-decoration: underline;
    }


    /* User Message Bubble - Distinct but subtle */
    div[data-testid="stChatMessage"]:has(div[data-testid="stAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
        background-color: #3C3C3C; /* Dark grey for User */
        color: #E8E8E8; /* Light text for user */
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stAvatarIcon-user"]) [data-testid="stChatMessageContent"] a {
        color: #60A5FA; /* Brighter blue for user links */
        text-decoration: underline;
    }
    
    /* Avatars */
    [data-testid="stAvatarIcon-assistant"] span, /* Targeting the emoji span */
    [data-testid="stAvatarIcon-user"] span {
        font-size: 1.8rem; /* Make emojis a bit larger */
    }


    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #252526; /* Dark sidebar */
        border-right: 1px solid #3A3A3A;
        padding: 25px;
    }
    [data-testid="stSidebar"] .stImage {
        margin-bottom: 1.5rem;
        text-align: center; /* Center the logo if it's in st.image */
    }
    [data-testid="stSidebar"] .stButton>button {
        background-color: #00A9E0; /* Cyan */
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 18px;
        font-weight: 600; /* Bolder button text */
        transition: background-color 0.2s ease-in-out, transform 0.1s ease;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background-color: #008CBA; /* Darker Cyan */
        transform: translateY(-1px); /* Slight lift on hover */
    }
    [data-testid="stSidebar"] h1 { /* Sidebar Title */
        color: #FFFFFF; 
        font-size: 2rem; /* Larger sidebar title */
        text-align: center;
    }
    [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] .stMarkdown li {
        color: #C0C0C0; /* Lighter grey for sidebar text */
        font-size: 0.98rem;
    }
     [data-testid="stSidebar"] .stMarkdown h5 { /* Sub-header in sidebar */
        color: #E0E0E0;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #444;
        padding-bottom: 0.3rem;
    }
    [data-testid="stSidebar"] .stCaption {
        color: #888888; /* Dimmer color for caption */
        text-align: center;
    }


    /* Header (Chat Title) */
    header[data-testid="stHeader"] { /* Hide default Streamlit header if not needed */
        display: none;
    }
    h1[data-testid="stHeading"] { /* Main chat title */
        color: #FFFFFF; 
        text-align: center;
        font-size: 2.2rem;
        font-weight: 600;
        padding-bottom: 1.5rem;
        margin-top: 1rem; /* Add some space from top */
    }

    /* Spinner color */
    .stSpinner > div > div { /* This targets the actual spinner element */
        border-top-color: #00A9E0 !important; /* Cyan spinner */
    }
    .stSpinner > div:nth-child(2) { /* Text next to spinner */
        color: #E0E0E0 !important;
    }

</style>
""", unsafe_allow_html=True)


# --- Chatbot State Initialization & Initial Greeting ---
if "chat_state" not in st.session_state:
    if 'chatbot_app' not in globals() or chatbot_app is None:
        st.error("Chatbot application is not available. Please check server logs.")
        st.session_state.chat_state = {
            "messages": [AIMessage(content="Chatbot is currently unavailable.")],
            "conversation_ended": True, "interaction_history": []
        }
    else:
        st.session_state.chat_state = create_initial_state()
        try:
            st.session_state.chat_state = chatbot_app.invoke(
                st.session_state.chat_state, {"recursion_limit": 50}
            )
        except Exception as e:
            print(f"ERROR: Initializing chatbot via invoke: {e}")
            st.session_state.chat_state["messages"].append(AIMessage(content="Sorry, an error occurred during startup. Please try refreshing."))
            st.session_state.chat_state["conversation_ended"] = True
            if "interaction_history" not in st.session_state.chat_state:
                 st.session_state.chat_state["interaction_history"] = []
            log_interaction(st.session_state.chat_state, "streamlit_init_invoke_error", details={"error": str(e)})

# --- Sidebar ---
with st.sidebar:
    # For a dark theme, ensure your logo has a transparent background or works well on dark.
    # st.image("assets/realtyflow_logo_dark_theme.png", width=150) # Example
    st.image("https://img.icons8.com/ios-filled/100/00A9E0/city-buildings.png", width=70) # Cyan icon
    st.title("RealtyFlow")
    st.markdown("Your AI Real Estate Partner")
    st.markdown("---")

    if st.button("🔄 New Conversation", use_container_width=True, key="restart_button_dark"):
        if 'chatbot_app' not in globals() or chatbot_app is None:
            st.error("Cannot restart: Chatbot application is not available.")
        else:
            # Preserve session_id and history across restarts for logging continuity
            old_session_id = st.session_state.chat_state.get("session_id", str(uuid.uuid4()))
            old_history = st.session_state.chat_state.get("interaction_history", [])

            st.session_state.chat_state = create_initial_state() # Reset chat specific state
            st.session_state.chat_state["session_id"] = old_session_id
            st.session_state.chat_state["interaction_history"] = old_history
            log_interaction(st.session_state.chat_state, "user_restart_streamlit_button")

            try:
                st.session_state.chat_state = chatbot_app.invoke(st.session_state.chat_state, {"recursion_limit": 50})
            except Exception as e:
                print(f"ERROR: Restarting chatbot invoke: {e}")
                st.session_state.chat_state["messages"].append(AIMessage(content="Sorry, couldn't restart properly. Please try again."))
            st.rerun() # Rerun to reflect the new state
    
    st.markdown("---")
    st.markdown("##### Quick Tips")
    st.caption(f"""
        - **Intent**: Clearly state if you're buying or selling.
        - **Budget**: Provide an approximate budget for purchases.
        - **Location**: Mention postcodes for precise area searches.
    """)
    st.markdown("---")
    st.caption("© 2024 RealtyFlow")


# --- Main Chat Interface ---
st.header("Chat with RealtyFlow AI 💬") # This might be hidden by CSS, but good for structure

# Chat messages container
chat_container = st.container() # Use a container for potentially fixed height scrolling later if needed

with chat_container:
    for msg in st.session_state.chat_state.get("messages", []):
        avatar_icon = "🤖" if isinstance(msg, AIMessage) else "👤" # Slightly different user avatar
        with st.chat_message("assistant" if isinstance(msg, AIMessage) else "user", avatar=avatar_icon):
            st.markdown(msg.content, unsafe_allow_html=True) # Allow HTML for bolding etc.

# User input handling
is_chat_disabled = st.session_state.get("chat_state", {}).get("conversation_ended", False) or \
                   ('chatbot_app' not in globals() or chatbot_app is None)

input_prompt = "How can I assist you with your property needs?"
if is_chat_disabled:
    if not ('chatbot_app' not in globals() or chatbot_app is None):
        input_prompt = "Conversation ended. Start a new one!"
    else:
        input_prompt = "Chatbot is currently unavailable."

user_input = st.chat_input(input_prompt, disabled=is_chat_disabled, key="chat_input_dark")

if user_input and not is_chat_disabled:
    # Add user message to official chat state
    st.session_state.chat_state["messages"].append(HumanMessage(content=user_input))
    
    # Display user's message immediately in the UI (it's now the last in chat_state.messages)
    # This gives instant feedback. We'll redraw everything after bot response.
    with chat_container: # Ensure it's added to the same container
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input, unsafe_allow_html=True)

    # Show thinking indicator and process
    with st.spinner("RealtyFlow is processing your request..."):
        try:
            st.session_state.chat_state = chatbot_app.invoke(
                st.session_state.chat_state,
                {"recursion_limit": 50}
            )
        except Exception as e:
            print(f"ERROR: Invoking chatbot with user input '{user_input}': {e}")
            error_msg = "I'm sorry, an unexpected issue occurred. Please try again or restart."
            st.session_state.chat_state["messages"].append(AIMessage(content=error_msg))
            st.session_state.chat_state["last_error"] = str(e)
            log_interaction(st.session_state.chat_state, "streamlit_invoke_error_user_input", details={"error": str(e)})
    
    st.rerun() # Rerun to display bot's response and update the entire chat log

elif is_chat_disabled and not st.session_state.chat_state.get("messages", []):
    st.warning("Chat is currently unavailable. Please try refreshing or contact support.")
