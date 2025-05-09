# src/utils.py
import os
import re
import pandas as pd
from typing import Set, List, Tuple, Dict, Any, Optional 
from enum import Enum

# Forward declare Enums and TypedDict for type hints.
# These will be fully defined in chatbot_engine.py and assigned here later
# to avoid circular import issues during initial module loading.
class ConversationStage(Enum):
    pass
class Intent(Enum):
    pass
class ChatState(dict): # Using dict as a placeholder
    pass


def normalize_postcode(postcode: str) -> str:
    if not postcode: return ""
    return re.sub(r'[^A-Z0-9]', '', postcode.upper())

def load_eligible_postcodes(file_path: str) -> Tuple[Set[str], List[str]]:
    sample_postcodes = ["SW1A1AA", "WC2N5DU", "EH11BQ", "M11AE", "B11HH", "L18JQ", "CF101AU"]
    try:
        if not os.path.exists(file_path):
            print(f"Warning: Postcode file '{file_path}' not found. Using sample postcodes for demonstration.")
            return set(sample_postcodes), sample_postcodes

        df = pd.read_csv(file_path)
        postcode_col_candidates = [col for col in df.columns if 'postcode' in col.lower()]
        
        if not postcode_col_candidates:
            print(f"Warning: No 'Postcode' column found in {file_path}. Attempting to use the first column. Columns: {df.columns}")
            if df.empty or not df.columns:
                print("Warning: CSV file is empty or has no columns. Using sample postcodes.")
                return set(sample_postcodes), sample_postcodes
            postcode_col = df.columns[0]
        else:
            postcode_col = postcode_col_candidates[0]
            
        postcodes_raw = [str(pc) for pc in df[postcode_col] if pd.notna(pc) and str(pc).strip()]
        normalized_postcodes = [normalize_postcode(pc) for pc in postcodes_raw]
        
        valid_normalized_postcodes = {pc for pc in normalized_postcodes if pc}
        valid_raw_postcodes = [raw for raw, norm in zip(postcodes_raw, normalized_postcodes) if norm in valid_normalized_postcodes]

        if not valid_normalized_postcodes:
            print(f"No valid postcodes found in {file_path} (column: {postcode_col}). Using sample data.")
            return set(sample_postcodes), sample_postcodes
        
        print(f"Loaded {len(valid_normalized_postcodes)} unique eligible postcodes from {file_path} (column: {postcode_col}). Using {len(valid_raw_postcodes)} raw postcodes for suggestions.")
        return valid_normalized_postcodes, valid_raw_postcodes if valid_raw_postcodes else list(valid_normalized_postcodes)
        
    except Exception as e:
        print(f"Error loading postcodes from {file_path}: {e}. Using sample data.")
        return set(sample_postcodes), sample_postcodes

def log_interaction(state: ChatState, action_type: str, user_input: Optional[str] = None, bot_response: Optional[str] = None, details: Optional[Dict] = None) -> ChatState: 
    if "interaction_history" not in state:
        state["interaction_history"] = []
    
    current_stage_val = state.get("conversation_stage")
    if isinstance(current_stage_val, Enum):
        current_stage_val = current_stage_val.value
    
    current_intent_val = state.get("intent")
    if isinstance(current_intent_val, Enum):
        current_intent_val = current_intent_val.value
    elif current_intent_val is None and hasattr(Intent, 'UNKNOWN') and isinstance(Intent.UNKNOWN, Enum):
            current_intent_val = Intent.UNKNOWN.value
    else:
        current_intent_val = str(current_intent_val) if current_intent_val is not None else "unknown"

    interaction = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "session_id": state.get("session_id", "N/A"),
        "stage": str(current_stage_val) if current_stage_val else "unknown_stage",
        "action_type": action_type,
        "attempts_at_stage": state.get("attempts", 0),
        "user_input": user_input,
        "bot_response": bot_response,
        "current_intent": current_intent_val,
        "details": details or {}
    }
    state["interaction_history"].append(interaction)
    return state
