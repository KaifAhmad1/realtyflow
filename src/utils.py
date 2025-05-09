import os
import re
import pandas as pd
from typing import Set, List, Tuple, Dict, Any
from enum import Enum
class ConversationStageBase(Enum): pass
class IntentBase(Enum): pass
ChatStateBase = Dict[str, Any]


def normalize_postcode(postcode: str) -> str:
    if not postcode: return ""
    return re.sub(r'[^A-Z0-9]', '', postcode.upper())

def load_eligible_postcodes(file_path: str) -> Tuple[Set[str], List[str]]:
    sample_postcodes = ["SW1A1AA", "WC2N5DU", "EH11BQ", "M11AE", "B11HH", "L18JQ", "CF101AU"]
    try:
        if not os.path.exists(file_path):
            print(f"Warning: Postcode file {file_path} not found. Using sample postcodes for demonstration.")
            return set(sample_postcodes), sample_postcodes

        df = pd.read_csv(file_path)
        postcode_col_candidates = [col for col in df.columns if 'postcode' in col.lower()]
        
        if not postcode_col_candidates:
            print(f"Warning: No 'Postcode' column found in {file_path}. Using first column: '{df.columns[0]}'.")
            if df.empty:
                print("Warning: Postcode CSV file is empty. Using sample postcodes.")
                return set(sample_postcodes), sample_postcodes
            postcode_col = df.columns[0]
        else:
            postcode_col = postcode_col_candidates[0]
            
        # Store original postcodes for FAISS suggestions, normalize for the set check
        original_postcodes = [str(pc).strip() for pc in df[postcode_col] if pd.notna(pc) and str(pc).strip()]
        normalized_postcodes_set = {normalize_postcode(pc) for pc in original_postcodes}
        
        if not original_postcodes:
            print(f"No valid postcodes found in {file_path} (column: {postcode_col}). Using sample data.")
            return set(sample_postcodes), sample_postcodes
        
        # print(f"Loaded {len(original_postcodes)} eligible postcodes from {file_path} (column: {postcode_col})")
        return normalized_postcodes_set, original_postcodes # Return set of normalized, list of original
    except Exception as e:
        print(f"Error loading postcodes from {file_path}: {e}. Using sample data.")
        return set(sample_postcodes), sample_postcodes

def log_interaction(state: ChatStateBase, action_type: str, user_input: str = None, bot_response: str = None, details: Dict = None) -> ChatStateBase:
    if "interaction_history" not in state: # Ensure interaction_history list exists
        state["interaction_history"] = []
    
    # Safely get and convert enum values if they are enums
    current_stage_val = state.get("conversation_stage")
    if isinstance(current_stage_val, Enum):
        current_stage_val = current_stage_val.value
    
    current_intent_val = state.get("intent")
    if isinstance(current_intent_val, Enum):
        current_intent_val = current_intent_val.value
    elif current_intent_val is None and hasattr(IntentBase, 'UNKNOWN'): # Check if UNKNOWN exists
        current_intent_val = IntentBase.UNKNOWN.value
    else:
        current_intent_val = str(current_intent_val) if current_intent_val is not None else "unknown"

    interaction = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "session_id": state.get("session_id", "N/A"),
        "stage": current_stage_val,
        "action_type": action_type,
        "attempts_at_stage": state.get("attempts", 0),
        "user_input": user_input,
        "bot_response": bot_response,
        "current_intent": current_intent_val,
        "details": details or {}
    }
    state["interaction_history"].append(interaction)
    return state
