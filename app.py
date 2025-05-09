import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import time # For simulating thinking
import uuid # For session_id if needed from chat_state

# Import the LangGraph app and state management functions
try:
    from src.chatbot_engine import chatbot_app, create_initial_state, log_interaction
except ImportError as e:
    st.error(f"Failed to import chatbot engine. Ensure 'src' directory is accessible: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error initializing chatbot components: {e}")
    st.stop()


st.set_page_config(page_title="RealtyFlow", page_icon="🏠", layout="centered") # Centered layout often looks cleaner for chat

# --- Enhanced Styling ---
st.markdown("""
<style>
    /* Overall App Background */
    .stApp {
        background: linear-gradient(to bottom right, #E0E7FF, #F3E8FF); /* Light lavender/blue gradient */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; /* Cleaner font */
    }

    /* Chat Container - to control max width and center it more effectively */
    .main .block-container {
        max-width: 800px; /* Max width for chat content */
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Chat Input Area */
    .stChatInputContainer > div {
        background-color: rgba(255, 255, 255, 0.8); /* Slightly transparent white */
        backdrop-filter: blur(5px); /* Frosted glass effect */
        border-top: 1px solid #D1D5DB; /* Light border */
        box-shadow: 0 -2px 10px rgba(0,0,0,0.05);
    }
    .stChatInput textarea {
        background-color: transparent !important;
        color: #333 !important;
    }

    /* Message Bubbles Styling */
    [data-testid="stChatMessageContent"] {
        border-radius: 20px; /* More rounded bubbles */
        padding: 12px 18px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        line-height: 1.6;
        word-wrap: break-word; /* Ensure long words break */
    }
    [data-testid="stChatMessageContent"] p {
        margin: 0;
        line-height: 1.6;
    }

    /* AI Message Bubble */
    div[data-testid="stChatMessage"]:has(div[data-testid="stAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {
        background-color: #4F46E5; /* Indigo for AI */
        color: white;
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="stAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] a {
        color: #C7D2FE; /* Lighter link color for AI bubble */
    }


    /* User Message Bubble */
    div[data-testid="stChatMessage"]:has(div[data-testid="stAvatarIcon-user"]) [data-testid="stChatMessageContent"] {
        background-color: #F3F4F6; /* Light gray for User */
        color: #1F2937; /* Darker text for user */
    }
     div[data-testid="stChatMessage"]:has(div[data-testid="stAvatarIcon-user"]) [data-testid="stChatMessageContent"] a {
        color: #3B82F6; /* Standard blue for user links */
    }


    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.7); /* More transparent white */
        backdrop-filter: blur(10px);
        border-right: 1px solid #D1D5DB;
        padding: 20px;
    }
    [data-testid="stSidebar"] .stImage {
        margin-bottom: 1rem; /* Space below logo */
    }
    [data-testid="stSidebar"] .stButton>button {
        background-color: #4F46E5; /* Indigo */
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 15px;
        font-weight: 500;
        transition: background-color 0.3s ease;
    }
    [data-testid="stSidebar"] .stButton>button:hover {
        background-color: #4338CA; /* Darker Indigo */
    }
    [data-testid="stSidebar"] h1 {
        color: #374151; /* Dark Gray for title */
        font-size: 1.75rem;
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #4B5563; /* Medium Gray for text */
        font-size: 0.95rem;
    }


    /* Header */
    header[data-testid="stHeader"] {
        background: none; /* Remove default header background */
    }
    h1[data-testid="stHeading"] { /* Chat header */
        color: #3730A3; /* Dark Indigo */
        text-align: center;
        padding-bottom: 1rem;
    }

</style>
""", unsafe_allow_html=True)


# --- Helper function for bot "thinking" effect ---
def stream_ pensée(text, delay=0.02):
    for word in text.split():
        yield word + " "
        time.sleep(delay)
    yield "\n" # Ensure newline at the end if needed

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
            # Initial invocation to get the greeting message
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

# Store messages separately for display to avoid showing user input prematurely
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

# Sync display_messages with chat_state messages ONLY IF it's from AI or already processed human message
# This is a bit tricky; the core idea is to only add to display_messages what's confirmed.
# A simpler approach: always mirror chat_state.messages to display_messages after bot response.
# For now, we'll rely on re-rendering after bot response to update the display.

# --- Sidebar ---
with st.sidebar:
    # Placeholder for a logo image if you have one in an 'assets' folder
    # try:
    #     st.image("assets/realtyflow_logo.png", width=150)
    # except FileNotFoundError:
    st.image("https://img.icons8.com/fluency/96/000000/city-buildings.png", width=80) # A different, cleaner icon
    st.title("RealtyFlow")
    st.markdown("Your Intelligent Real Estate Assistant. Let's find your perfect property solution!")
    st.markdown("---")

    if st.button("🔄 New Conversation", use_container_width=True, key="restart_button_main"):
        if 'chatbot_app' not in globals() or chatbot_app is None:
            st.error("Cannot restart: Chatbot application is not available.")
        else:
            old_session_id = st.session_state.chat_state.get("session_id", str(uuid.uuid4()))
            old_history = st.session_state.chat_state.get("interaction_history", [])

            st.session_state.chat_state = create_initial_state()
            st.session_state.chat_state["session_id"] = old_session_id
            st.session_state.chat_state["interaction_history"] = old_history
            log_interaction(st.session_state.chat_state, "user_restart_streamlit_button")
            st.session_state.display_messages = [] # Clear display messages too

            try:
                st.session_state.chat_state = chatbot_app.invoke(st.session_state.chat_state, {"recursion_limit": 50})
            except Exception as e:
                print(f"ERROR: Restarting chatbot invoke: {e}")
                st.session_state.chat_state["messages"].append(AIMessage(content="Sorry, couldn't restart properly. Please try again."))
            st.rerun()
    
    st.markdown("---")
    st.markdown("##### Quick Info")
    st.caption(f"""
        - **Buying or Selling?** Let me know!
        - **Budget?** Have a range in mind.
        - **Location?** Postcodes help.
    """)
    st.markdown("---")
    st.caption("© 2024 RealtyFlow. All rights reserved.")


# --- Main Chat Interface ---
st.header("Chat with RealtyFlow AI 💬")

# Display chat messages from chat_state.messages
# This ensures that only processed messages (including the user's own confirmed message) are shown.
for msg in st.session_state.chat_state.get("messages", []):
    if isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="🤖"): # Can use a URL for avatar: avatar="URL_TO_BOT_AVATAR.png"
            # To simulate streaming for AI messages (if desired, and if they are not already streamed by LangGraph)
            # For now, direct display:
            st.markdown(msg.content, unsafe_allow_html=True)
    elif isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="🧑"): # Can use a URL: avatar="URL_TO_USER_AVATAR.png"
            st.markdown(msg.content, unsafe_allow_html=True)


# User input handling
is_chat_disabled = st.session_state.get("chat_state", {}).get("conversation_ended", False) or \
                   ('chatbot_app' not in globals() or chatbot_app is None)

input_prompt = "What can I help you with today?"
if is_chat_disabled:
    if not ('chatbot_app' not in globals() or chatbot_app is None):
        input_prompt = "Conversation ended. Start a new one!"
    else:
        input_prompt = "Chatbot is currently unavailable."

user_input = st.chat_input(input_prompt, disabled=is_chat_disabled, key="chat_input_main")

if user_input and not is_chat_disabled:
    # 1. Add user message to the official chat_state
    st.session_state.chat_state["messages"].append(HumanMessage(content=user_input))
    
    # 2. Immediately re-render to show the user's message (it's now in chat_state.messages)
    #    A placeholder can be shown while bot is "thinking"
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input, unsafe_allow_html=True) # Show user's latest message immediately

    with st.spinner("RealtyFlow is thinking..."): # Spinner while bot processes
        try:
            # 3. Invoke the chatbot with the new state
            st.session_state.chat_state = chatbot_app.invoke(
                st.session_state.chat_state,
                {"recursion_limit": 50}
            )
        except Exception as e:
            print(f"ERROR: Invoking chatbot with user input '{user_input}': {e}")
            error_msg = "I'm sorry, an unexpected issue occurred. Please try again or restart."
            # Add error to official messages
            st.session_state.chat_state["messages"].append(AIMessage(content=error_msg))
            st.session_state.chat_state["last_error"] = str(e)
            log_interaction(st.session_state.chat_state, "streamlit_invoke_error_user_input", details={"error": str(e)})
    
    # 4. Rerun to display the bot's response (and clear the spinner)
    st.rerun()

elif is_chat_disabled and not st.session_state.chat_state.get("messages", []):
    # This might happen if the very first initialization failed catastrophically
    st.warning("Chat is currently unavailable. Please try refreshing the page or contact support if the issue persists.")
