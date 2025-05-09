import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage # Import SystemMessage for type checking if ever needed
import time 
import uuid

# Import the LangGraph app and state management functions
try:
    from src.chatbot_engine import chatbot_app, create_initial_state, log_interaction, SYSTEM_INSTRUCTION
    if chatbot_app is None:
        raise ImportError("Chatbot graph (chatbot_app) from chatbot_engine is None. Check server logs for compilation/initialization errors.")
except ImportError as e:
    st.error(f"Critical Error: Failed to load chatbot components. {e}")
    chatbot_app = None 
except Exception as e: 
    st.error(f"Unexpected error during chatbot initialization: {e}")
    chatbot_app = None


st.set_page_config(page_title="RealtyFlow", page_icon="🏠", layout="centered")

# --- Dark Theme Styling ---
st.markdown(f"""
<style>
    /* Overall App Background & Font */
    .stApp {{
        background-color: #1E1E1E; 
        color: #E0E0E0; 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
    }}
    .main .block-container {{ max-width: 750px; padding-top: 1rem; padding-bottom: 1rem; }}
    .stChatInputContainer > div {{ background-color: #2A2A2A; border-top: 1px solid #444444; box-shadow: 0 -2px 10px rgba(0,0,0,0.2); }}
    .stChatInput textarea {{ background-color: #333333 !important; color: #F0F0F0 !important; border: 1px solid #555555 !important; border-radius: 8px !important; }}
    .stChatInput textarea::placeholder {{ color: #AAAAAA !important; }}
    .stChatInput button {{ background-color: #00A9E0 !important; color: white !important; border-radius: 8px !important; border: none !important; }}
    .stChatInput button:hover {{ background-color: #008CBA !important; }}
    .stChatInput button:disabled {{ background-color: #555 !important; color: #888 !important; }}
    [data-testid="stChatMessageContent"] {{ border-radius: 18px; padding: 14px 20px; box-shadow: 0 3px 8px rgba(0,0,0,0.25); line-height: 1.65; word-wrap: break-word; }}
    [data-testid="stChatMessageContent"] p {{ margin: 0; line-height: 1.65; }}
    [data-testid="stChatMessageContent"] strong {{ color: #FFFFFF; font-weight: 600; }}
    div[data-testid="stChatMessage"]:has(div[data-testid="stAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] {{ background: linear-gradient(135deg, #00A9E0, #0074D9); color: white; }}
    div[data-testid="stChatMessage"]:has(div[data-testid="stAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] a {{ color: #B3E5FC; text-decoration: underline; }}
    div[data-testid="stChatMessage"]:has(div[data-testid="stAvatarIcon-assistant"]) [data-testid="stChatMessageContent"] strong {{ color: #FFFFFF; }}
    div[data-testid="stChatMessage"]:has(div[data-testid="stAvatarIcon-user"]) [data-testid="stChatMessageContent"] {{ background-color: #3C3C3C; color: #E8E8E8; }}
    div[data-testid="stChatMessage"]:has(div[data-testid="stAvatarIcon-user"]) [data-testid="stChatMessageContent"] a {{ color: #60A5FA; text-decoration: underline; }}
    div[data-testid="stChatMessage"]:has(div[data-testid="stAvatarIcon-user"]) [data-testid="stChatMessageContent"] strong {{ color: #F5F5F5; }}
    [data-testid="stAvatarIcon-assistant"] span, [data-testid="stAvatarIcon-user"] span {{ font-size: 1.8rem; }}
    [data-testid="stSidebar"] {{ background-color: #252526; border-right: 1px solid #3A3A3A; padding: 25px; }}
    [data-testid="stSidebar"] .stImage {{ margin-bottom: 1.5rem; text-align: center; }}
    [data-testid="stSidebar"] .stButton>button {{ background-color: #00A9E0; color: white; border: none; border-radius: 8px; padding: 12px 18px; font-weight: 600; transition: background-color 0.2s ease-in-out, transform 0.1s ease; }}
    [data-testid="stSidebar"] .stButton>button:hover {{ background-color: #008CBA; transform: translateY(-1px); }}
    [data-testid="stSidebar"] h1 {{ color: #FFFFFF; font-size: 2rem; text-align: center; }}
    [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] .stMarkdown li {{ color: #C0C0C0; font-size: 0.98rem; }}
    [data-testid="stSidebar"] .stMarkdown h5 {{ color: #E0E0E0; margin-top: 1.5rem; margin-bottom: 0.5rem; border-bottom: 1px solid #444; padding-bottom: 0.3rem; }}
    [data-testid="stSidebar"] .stCaption {{ color: #888888; text-align: center; }}
    .chat-header {{ color: #FFFFFF; text-align: center; font-size: 2.2rem; font-weight: 600; padding-bottom: 1.5rem; margin-top: 1rem; }}
    .stSpinner > div > div {{ border-top-color: #00A9E0 !important; }}
    .stSpinner > div:nth-child(2) {{ color: #E0E0E0 !important; }}
    header[data-testid="stHeader"] {{ display: none !important; }}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='chat-header'>RealtyFlow AI 💬</h1>", unsafe_allow_html=True)

if "chat_state" not in st.session_state:
    if chatbot_app is None:
        st.session_state.chat_state = {
            "messages": [AIMessage(content="Chatbot is currently unavailable due to an initialization error. Please check server logs or contact support.")],
            "conversation_ended": True, "interaction_history": []
        }
    else:
        st.session_state.chat_state = create_initial_state()
        try:
            # The initial_greeting_node in chatbot_engine will add the SystemMessage and first AIMessage
            st.session_state.chat_state = chatbot_app.invoke(
                st.session_state.chat_state, {"recursion_limit": 50}
            )
        except Exception as e:
            print(f"ERROR: Initializing chatbot via invoke: {e}")
            if "messages" not in st.session_state.chat_state: st.session_state.chat_state["messages"] = []
            st.session_state.chat_state["messages"].append(AIMessage(content="Sorry, an error occurred during startup. Please try refreshing."))
            st.session_state.chat_state["conversation_ended"] = True
            if "interaction_history" not in st.session_state.chat_state: st.session_state.chat_state["interaction_history"] = []
            try: log_interaction(st.session_state.chat_state, "streamlit_init_invoke_error", details={"error": str(e)})
            except Exception as log_e: print(f"Error during logging init error: {log_e}")

with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/100/00A9E0/city-buildings.png", width=70)
    st.title("RealtyFlow")
    st.markdown("Your AI Real Estate Partner")
    st.markdown("---")

    if st.button("🔄 New Conversation", use_container_width=True, key="restart_button_dark_ui_final"):
        if chatbot_app is None:
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
    st.markdown("##### Quick Tips")
    st.caption(f"""
        - **Intent**: Clearly state if you're **buying** or **selling**.
        - **Budget**: Provide an approximate **budget** for purchases.
        - **Location**: Mention **postcodes** for precise area searches.
    """)
    st.markdown("---")
    st.caption("© 2024 RealtyFlow")

chat_container = st.container()
with chat_container:
    messages_list = st.session_state.get("chat_state", {}).get("messages", [])
    if not isinstance(messages_list, list): messages_list = []

    for msg_idx, msg_obj in enumerate(messages_list):
        # CRITICAL: Only display AIMessage and HumanMessage to the user.
        # SystemMessage (like SYSTEM_INSTRUCTION) is for the LLM only.
        if isinstance(msg_obj, AIMessage):
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(msg_obj.content, unsafe_allow_html=True)
        elif isinstance(msg_obj, HumanMessage):
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg_obj.content, unsafe_allow_html=True)

is_chat_disabled = st.session_state.get("chat_state", {}).get("conversation_ended", False) or (chatbot_app is None)
input_prompt = "How can I assist you with your property needs?"
if is_chat_disabled:
    input_prompt = "Conversation ended. Start a new one!" if chatbot_app is not None else "Chatbot is currently unavailable."

user_input = st.chat_input(input_prompt, disabled=is_chat_disabled, key="chat_input_dark_theme_final")

if user_input and not is_chat_disabled:
    st.session_state.chat_state["messages"].append(HumanMessage(content=user_input))
    
    # Optimistically display user message (it's now the last message in chat_state)
    # The main loop will render it properly after rerun, this is for immediate feedback.
    with chat_container: # Ensure it's rendered in the correct place
         with st.chat_message("user", avatar="👤"):
            st.markdown(user_input, unsafe_allow_html=True)

    with st.spinner("RealtyFlow is processing your request..."):
        try:
            st.session_state.chat_state = chatbot_app.invoke(
                st.session_state.chat_state, {"recursion_limit": 50}
            )
        except Exception as e:
            print(f"ERROR: Invoking chatbot with user input '{user_input}': {e}")
            error_msg = "I'm sorry, an unexpected issue occurred. Please try again or restart."
            st.session_state.chat_state["messages"].append(AIMessage(content=error_msg))
            if "last_error" in st.session_state.chat_state: 
                st.session_state.chat_state["last_error"] = str(e)
            log_interaction(st.session_state.chat_state, "streamlit_invoke_error_user_input", details={"error": str(e)})
    st.rerun()

elif is_chat_disabled and not st.session_state.get("chat_state", {}).get("messages", []):
    with chat_container:
        st.warning("Chat is currently unavailable. Please try refreshing or contact support.")
