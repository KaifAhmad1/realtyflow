# src/chatbot_engine.py
import uuid
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, TypedDict # <--- Ensure all are here

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, END

from .config import GOOGLE_API_KEY, MIN_BUDGET_NEW_HOME, COMPANY_PHONE_NUMBER, MAX_ATTEMPTS, POSTCODE_FILE_PATH
from .utils import normalize_postcode, load_eligible_postcodes, log_interaction
from .agents import (
    EnhancedIntentClassifierAgent, EnhancedInfoGathererAgent,
    EnhancedBudgetProcessorAgent, EnhancedPostcodeProcessorAgent,
    AgentIntent, AgentBuyType, AgentYesNo
)

# --- Canonical Enums and State Definition for the Graph ---
class Intent(str, Enum):
    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"

class BuyType(str, Enum):
    NEW_HOME = "new_home"
    RE_SALE = "re_sale"
    UNKNOWN = "unknown"

class YesNo(str, Enum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"

class ConversationStage(str, Enum):
    GREETING = "greeting"
    AWAITING_INTENT = "awaiting_intent"
    AWAITING_NAME = "awaiting_name"
    AWAITING_PHONE = "awaiting_phone"
    AWAITING_EMAIL = "awaiting_email"
    AWAITING_BUY_TYPE = "awaiting_buy_type"
    AWAITING_BUDGET = "awaiting_budget"
    AWAITING_POSTCODE = "awaiting_postcode"
    AWAITING_REASSISTANCE = "awaiting_reassistance"
    ENDED = "ended"

class ChatState(TypedDict): # <--- TypedDict
    messages: List[BaseMessage] # <--- List
    intent: Optional[Intent] # <--- Optional
    buy_type: Optional[BuyType] # <--- Optional
    name: Optional[str] # <--- Optional
    phone: Optional[str] # <--- Optional
    email: Optional[str] # <--- Optional
    budget: Optional[float] # <--- Optional
    postcode: Optional[str] # <--- Optional
    raw_postcode_input: Optional[str] # <--- Optional
    postcode_covered: Optional[bool] # <--- Optional
    suggested_postcode: Optional[str] # <--- Optional
    attempts: int
    conversation_ended: bool
    conversation_stage: ConversationStage
    last_error: Optional[str] # <--- Optional
    session_id: str
    interaction_history: List[Dict[str, Any]] # <--- List, Dict, Any

# --- Initialize Models and Agents ---
# (This part remains the same)
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY, convert_system_message_to_human=True)
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
except Exception as e:
    print(f"FATAL: Failed to initialize Google Generative AI models: {e}")
    raise

intent_classifier = EnhancedIntentClassifierAgent(llm)
info_gatherer = EnhancedInfoGathererAgent(llm)
budget_processor = EnhancedBudgetProcessorAgent(llm)
eligible_postcodes_set, eligible_postcodes_list_raw = load_eligible_postcodes(POSTCODE_FILE_PATH)
postcode_processor = EnhancedPostcodeProcessorAgent(eligible_postcodes_set, eligible_postcodes_list_raw, embedding_model, llm)

# --- LangGraph Node Functions ---
# (All node functions remain the same, their type hints use the now-imported types)
def create_initial_state() -> ChatState:
    return ChatState(
        messages=[], intent=None, buy_type=None, name=None, phone=None, email=None,
        budget=None, postcode=None, raw_postcode_input=None, postcode_covered=None,
        suggested_postcode=None, attempts=0, conversation_ended=False,
        conversation_stage=ConversationStage.GREETING, last_error=None,
        session_id=str(uuid.uuid4()), interaction_history=[]
    )

def initial_greeting_node(state: ChatState) -> ChatState:
    if state["conversation_stage"] == ConversationStage.GREETING and not any(isinstance(m, AIMessage) for m in state["messages"]):
        greeting_msg = "Hello! I'm RealtyFlow AI. Are you looking to buy or sell a property today?"
        state["messages"] = [
            SystemMessage(content="You are RealtyFlow AI, a friendly and professional assistant. Qualify leads by asking targeted questions. Be concise."),
            AIMessage(content=greeting_msg)
        ]
        state["conversation_stage"] = ConversationStage.AWAITING_INTENT
        state["attempts"] = 0
        log_interaction(state, "initial_greeting_sent", bot_response=greeting_msg)
    return state

def handle_intent_node(state: ChatState) -> ChatState:
    user_msg = state["messages"][-1].content
    agent_intent_val = intent_classifier.classify_intent(user_msg)
    state["intent"] = Intent(agent_intent_val.value) 

    if state["intent"] == Intent.UNKNOWN:
        state["attempts"] += 1
        bot_response = "I'm sorry, I didn't quite catch that. Are you looking to buy or sell?"
    else:
        bot_response = f"Great, you're looking to {state['intent'].value}! To get started, can I please have your full name?"
        state["conversation_stage"] = ConversationStage.AWAITING_NAME
        state["attempts"] = 0
    state["messages"].append(AIMessage(content=bot_response))
    log_interaction(state, f"intent_{state['intent'].value}", user_msg, bot_response)
    return state

def handle_name_node(state: ChatState) -> ChatState:
    user_msg = state["messages"][-1].content
    is_valid, bot_response_msg, name_val = info_gatherer.get_name(state, user_msg)
    if not is_valid:
        state["attempts"] += 1
        log_type = "get_name_invalid"
    else:
        state["name"] = name_val
        state["conversation_stage"] = ConversationStage.AWAITING_PHONE
        state["attempts"] = 0
        log_type = "get_name_valid"
    state["messages"].append(AIMessage(content=bot_response_msg))
    log_interaction(state, log_type, user_msg, bot_response_msg)
    return state

def handle_phone_node(state: ChatState) -> ChatState:
    user_msg = state["messages"][-1].content
    is_valid, bot_response_msg, phone_val = info_gatherer.get_phone(state, user_msg)
    if not is_valid:
        state["attempts"] += 1
        log_type = "get_phone_invalid"
    else:
        state["phone"] = phone_val
        state["conversation_stage"] = ConversationStage.AWAITING_EMAIL
        state["attempts"] = 0
        log_type = "get_phone_valid"
    state["messages"].append(AIMessage(content=bot_response_msg))
    log_interaction(state, log_type, user_msg, bot_response_msg)
    return state

def handle_email_node(state: ChatState) -> ChatState:
    user_msg = state["messages"][-1].content
    is_valid, reason_or_empty, email_val = info_gatherer.get_email(state, user_msg)
    if not is_valid:
        state["attempts"] += 1
        bot_response = reason_or_empty 
        log_type = "get_email_invalid"
    else:
        state["email"] = email_val
        state["attempts"] = 0
        log_type = "get_email_valid"
        if state["intent"] == Intent.BUY:
            bot_response = "Are you looking for a new home or a re-sale home?"
            state["conversation_stage"] = ConversationStage.AWAITING_BUY_TYPE
        elif state["intent"] == Intent.SELL:
            bot_response = "What is the postcode of the property you're selling?"
            state["conversation_stage"] = ConversationStage.AWAITING_POSTCODE
        else: 
            bot_response = "Thank you. Please tell me the postcode of interest."
            state["conversation_stage"] = ConversationStage.AWAITING_POSTCODE
    state["messages"].append(AIMessage(content=bot_response))
    log_interaction(state, log_type, user_msg, bot_response)
    return state

def handle_buy_type_node(state: ChatState) -> ChatState:
    user_msg = state["messages"][-1].content
    agent_buy_type_val = intent_classifier.classify_buy_type(user_msg)
    state["buy_type"] = BuyType(agent_buy_type_val.value)

    if state["buy_type"] == BuyType.UNKNOWN:
        state["attempts"] += 1
        bot_response = "Sorry, I'm not sure if that's new or re-sale. Could you clarify? (e.g., 'new build', 'existing property')"
    else:
        bt_text = state['buy_type'].value.replace('_', ' ')
        bot_response = f"Got it, a {bt_text} property. What's your approximate budget?"
        state["conversation_stage"] = ConversationStage.AWAITING_BUDGET
        state["attempts"] = 0
    state["messages"].append(AIMessage(content=bot_response))
    log_interaction(state, f"buy_type_{state['buy_type'].value}", user_msg, bot_response)
    return state

def handle_budget_node(state: ChatState) -> ChatState:
    user_msg = state["messages"][-1].content
    budget_val, reason_or_empty = budget_processor.process_budget(user_msg)
    if budget_val is None:
        state["attempts"] += 1
        bot_response = reason_or_empty 
        log_type = "get_budget_invalid"
    else:
        state["budget"] = budget_val
        state["attempts"] = 0
        log_type = "get_budget_valid"
        if state["buy_type"] == BuyType.NEW_HOME and budget_val < MIN_BUDGET_NEW_HOME:
            bot_response = (f"For new homes, our listings start at £{MIN_BUDGET_NEW_HOME:,.0f}. "
                           f"Your budget of £{budget_val:,.0f} is below this. "
                           f"Please call {COMPANY_PHONE_NUMBER} for other options. "
                           "Is there anything else I can help with? (yes/no)")
            state["conversation_stage"] = ConversationStage.AWAITING_REASSISTANCE
        else:
            bot_response = f"Understood. Budget: £{budget_val:,.0f}. What is the postcode of interest?"
            state["conversation_stage"] = ConversationStage.AWAITING_POSTCODE
    state["messages"].append(AIMessage(content=bot_response))
    log_interaction(state, log_type, user_msg, bot_response, {"parsed_budget": budget_val})
    return state

def handle_postcode_node(state: ChatState) -> ChatState:
    user_msg_raw = state["messages"][-1].content
    state["raw_postcode_input"] = user_msg_raw
    
    norm_pc, is_covered, suggestion_raw, error_msg = postcode_processor.process_postcode(user_msg_raw)

    if error_msg: 
        state["attempts"] += 1
        bot_response = error_msg
        log_type = "get_postcode_invalid_format"
    else:
        state["postcode"] = norm_pc
        state["postcode_covered"] = is_covered
        state["suggested_postcode"] = suggestion_raw
        state["attempts"] = 0
        state["conversation_stage"] = ConversationStage.AWAITING_REASSISTANCE
        log_type = "get_postcode_processed"

        if is_covered:
            bot_response = (f"Great! Postcode {user_msg_raw.upper()} is covered. "
                           "Someone will contact you within 24 hours. "
                           "Anything else I can help with? (yes/no)")
        else:
            msg_parts = [f"Sorry, we don't currently cover {user_msg_raw.upper()}."]
            if suggestion_raw:
                msg_parts.append(f"Did you perhaps mean {suggestion_raw}?")
            
            if state["intent"] == Intent.BUY and state["buy_type"] == BuyType.NEW_HOME:
                msg_parts.append(f"For new homes elsewhere, call us on {COMPANY_PHONE_NUMBER}.")
            else: 
                msg_parts.append(f"Please call {COMPANY_PHONE_NUMBER} for other options.")
            msg_parts.append("Anything else I can help with? (yes/no)")
            bot_response = " ".join(msg_parts)
    
    state["messages"].append(AIMessage(content=bot_response))
    log_interaction(state, log_type, user_msg_raw, bot_response, {"norm_pc": norm_pc, "suggestion": suggestion_raw, "covered": is_covered})
    return state

def handle_reassistance_node(state: ChatState) -> ChatState:
    user_msg = state["messages"][-1].content
    agent_choice = intent_classifier.classify_yes_no(user_msg)
    choice = YesNo(agent_choice.value)

    if choice == YesNo.YES:
        session_id_cache = state["session_id"]
        history_cache = state["interaction_history"]
        state.clear()
        state.update(create_initial_state())
        state["session_id"] = session_id_cache
        state["interaction_history"] = history_cache
        
        state = initial_greeting_node(state) 
        log_action = "reassistance_yes_restart"
        bot_response_for_log = state["messages"][-1].content if state["messages"] and isinstance(state["messages"][-1], AIMessage) else "Restarting..."

    elif choice == YesNo.NO:
        bot_response_for_log = "Thank you for chatting with RealtyFlow. Goodbye!"
        state["messages"].append(AIMessage(content=bot_response_for_log))
        state["conversation_ended"] = True
        state["conversation_stage"] = ConversationStage.ENDED
        log_action = "reassistance_no_end"
    else: 
        state["attempts"] += 1
        bot_response_for_log = "I didn't quite get that. Could you please answer with 'yes' or 'no'?"
        state["messages"].append(AIMessage(content=bot_response_for_log))
        log_action = "reassistance_unknown"
    
    log_interaction(state, log_action, user_msg, bot_response_for_log)
    return state

def handle_max_attempts_fallback_node(state: ChatState) -> ChatState:
    bot_response = f"I'm having trouble understanding. For further assistance, please call {COMPANY_PHONE_NUMBER}. Thank you."
    state["messages"].append(AIMessage(content=bot_response))
    state["conversation_ended"] = True
    state["conversation_stage"] = ConversationStage.ENDED
    log_interaction(state, "max_attempts_fallback", bot_response=bot_response)
    return state

# --- Routing Logic ---
def route_next_step(state: ChatState) -> str:
    if state.get("conversation_ended", False): return END
    if state["attempts"] >= MAX_ATTEMPTS:
        state["attempts"] = 0 
        return "max_attempts_fallback_node"

    if state["messages"] and isinstance(state["messages"][-1], AIMessage): return END

    stage = state["conversation_stage"]
    if stage == ConversationStage.GREETING: return "initial_greeting_node"
    if stage == ConversationStage.AWAITING_INTENT: return "handle_intent_node"
    if stage == ConversationStage.AWAITING_NAME: return "handle_name_node"
    if stage == ConversationStage.AWAITING_PHONE: return "handle_phone_node"
    if stage == ConversationStage.AWAITING_EMAIL: return "handle_email_node"
    if stage == ConversationStage.AWAITING_BUY_TYPE: return "handle_buy_type_node"
    if stage == ConversationStage.AWAITING_BUDGET: return "handle_budget_node"
    if stage == ConversationStage.AWAITING_POSTCODE: return "handle_postcode_node"
    if stage == ConversationStage.AWAITING_REASSISTANCE: return "handle_reassistance_node"
    
    return END

# --- Build the Graph ---
workflow = StateGraph(ChatState)
nodes = [
    ("initial_greeting_node", initial_greeting_node),
    ("handle_intent_node", handle_intent_node), ("handle_name_node", handle_name_node),
    ("handle_phone_node", handle_phone_node), ("handle_email_node", handle_email_node),
    ("handle_buy_type_node", handle_buy_type_node), ("handle_budget_node", handle_budget_node),
    ("handle_postcode_node", handle_postcode_node), ("handle_reassistance_node", handle_reassistance_node),
    ("max_attempts_fallback_node", handle_max_attempts_fallback_node)
]
for name, func in nodes: workflow.add_node(name, func)

workflow.set_entry_point("initial_greeting_node")

routing_map = {name: name for name, _ in nodes}
routing_map[END] = END

for name, _ in nodes:
    if name != "max_attempts_fallback_node": 
        workflow.add_conditional_edges(name, route_next_step, routing_map)
workflow.add_edge("max_attempts_fallback_node", END)

try:
    chatbot_app = workflow.compile()
    print("RealtyFlow LangGraph app compiled successfully.")
except Exception as e:
    print(f"Error compiling LangGraph app: {e}")
    chatbot_app = None 
    raise

import src.utils as project_utils
project_utils.ConversationStage = ConversationStage
project_utils.Intent = Intent
project_utils.ChatState = ChatState
