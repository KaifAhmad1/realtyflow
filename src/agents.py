import re
import json
import numpy as np
import faiss
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from typing import Tuple, Optional, List, Set, Dict, Any
from enum import Enum

from .utils import normalize_postcode

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

# Placeholder for ChatState, actual TypedDict is in chatbot_engine.py
ChatState = Dict[str, Any]

class EnhancedIntentClassifierAgent:
    def __init__(self, llm_model: ChatGoogleGenerativeAI):
        self.llm = llm_model

    def _build_classification_chain(self, entity_name: str, allowed_options: List[str], description: str, examples: List[Dict[str, str]]):
        examples_text = "\n".join([f"User: \"{e['input']}\" -> Assistant: {e['output']}" for e in examples])
        prompt_template_str = f"""Your task is to classify the user's input for '{entity_name}'.
{description}
Allowed classifications: {', '.join(allowed_options)}.
Respond with ONLY ONE of the allowed classifications. Do not add any other text or explanation.

Examples:
{examples_text}

User: "{{user_message}}"
Assistant:"""
        prompt = ChatPromptTemplate.from_template(prompt_template_str)
        return prompt | self.llm | StrOutputParser()

    def classify_intent(self, user_message: str) -> Intent:
        examples = [
            {"input": "I want to buy a house", "output": "buy"}, {"input": "I'm looking to sell my apartment", "output": "sell"},
            {"input": "Purchase property", "output": "buy"}, {"input": "List my home", "output": "sell"},
            {"input": "Can you help me find a place?", "output": "buy"}, {"input": "Need to offload my current place", "output": "sell"}
        ]
        chain = self._build_classification_chain("User Intent", [Intent.BUY.value, Intent.SELL.value],
                                               "Determine if the user primarily wants to buy or sell a property.", examples)
        try:
            result = chain.invoke({"user_message": user_message}).strip().lower()
            # print(f"DEBUG: Intent classification for '{user_message}': {result}") # For server-side debug
            return Intent(result) if result in [Intent.BUY.value, Intent.SELL.value] else Intent.UNKNOWN
        except Exception as e:
            print(f"ERROR: Intent classification: {e}")
            return Intent.UNKNOWN

    def classify_buy_type(self, user_message: str) -> BuyType:
        examples = [
            {"input": "A new build", "output": "new_home"}, {"input": "Something pre-owned", "output": "re_sale"},
            {"input": "Newly constructed", "output": "new_home"}, {"input": "An existing house", "output": "re_sale"},
        ]
        chain = self._build_classification_chain("Property Type for Buyer", [BuyType.NEW_HOME.value, BuyType.RE_SALE.value],
                                               "Determine if the buyer is looking for a new home or a re-sale property.", examples)
        try:
            result = chain.invoke({"user_message": user_message}).strip().lower()
            # print(f"DEBUG: Buy type classification for '{user_message}': {result}")
            return BuyType(result) if result in [BuyType.NEW_HOME.value, BuyType.RE_SALE.value] else BuyType.UNKNOWN
        except Exception as e:
            print(f"ERROR: Buy type classification: {e}")
            return BuyType.UNKNOWN

    def classify_yes_no(self, user_message: str) -> YesNo:
        examples = [
            {"input": "Yes, please", "output": "yes"}, {"input": "Nope", "output": "no"},
            {"input": "Sure", "output": "yes"}, {"input": "Not right now", "output": "no"},
            {"input": "Affirmative", "output": "yes"}, {"input": "I don't think so", "output": "no"}
        ]
        chain = self._build_classification_chain("Yes/No Answer", [YesNo.YES.value, YesNo.NO.value],
                                               "Determine if the user's response is affirmative (yes) or negative (no).", examples)
        try:
            result = chain.invoke({"user_message": user_message}).strip().lower()
            # print(f"DEBUG: Yes/No classification for '{user_message}': {result}")
            return YesNo(result) if result in [YesNo.YES.value, YesNo.NO.value] else YesNo.UNKNOWN
        except Exception as e:
            print(f"ERROR: Yes/no classification: {e}")
            return YesNo.UNKNOWN

class EnhancedInfoGathererAgent:
    def __init__(self, llm_model: ChatGoogleGenerativeAI):
        self.llm = llm_model

    def _validate_with_llm(self, field_name: str, value_to_validate: str) -> Tuple[bool, str]:
        template_str = """You are a data validation assistant.
Task: Check if the provided value for the field '{field}' is plausible and correctly formatted.
Input Value: {input_value}

Instructions:
- Analyze the input value.
- For 'name', it should look like a real name (not gibberish, very short, or excessively long). Names can contain spaces and hyphens.
- For 'phone', it should resemble a phone number (mostly digits, possibly with spaces, hyphens, or a leading '+', appropriate length typically 7-15 digits).
- For 'email', it must contain '@' and a '.' in the domain part (e.g. user@example.com).
- Respond with a JSON object containing two keys:
  - "is_valid": boolean (true if valid, false otherwise)
  - "reason": string (brief explanation if invalid, or "Valid." if valid)

Example for an invalid name: {{"is_valid": false, "reason": "Name appears to be too short or not a typical name."}}
Example for a valid email: {{"is_valid": true, "reason": "Valid email format."}}

JSON Response:"""
        prompt = PromptTemplate.from_template(template_str)
        parser = JsonOutputParser()
        chain = prompt | self.llm | parser
        str_chain_for_fallback = prompt | self.llm | StrOutputParser()

        try:
            result = chain.invoke({"field": field_name, "input_value": value_to_validate})
            return result.get("is_valid", False), result.get("reason", f"Validation inconclusive for {field_name}")
        except Exception: # Catches JSON parsing errors and LLM errors
            # print(f"DEBUG: JsonOutputParser failed for {field_name} validation. Retrying with string parsing.")
            raw_output = str_chain_for_fallback.invoke({"field": field_name, "input_value": value_to_validate})
            # print(f"DEBUG: Raw LLM output for {field_name} validation: {raw_output}")
            try:
                start_index = raw_output.find('{')
                end_index = raw_output.rfind('}')
                if start_index != -1 and end_index != -1 and end_index > start_index:
                    json_str = raw_output[start_index : end_index+1]
                    result = json.loads(json_str)
                    return result.get("is_valid", False), result.get("reason", f"Validation inconclusive for {field_name} (raw parse)")
                else:
                    if "is_valid\": true" in raw_output.lower() or "valid." in raw_output.lower():
                         return True, "Valid (inferred from text)."
                    if "is_valid\": false" in raw_output.lower() or "invalid" in raw_output.lower():
                         return False, "Invalid (inferred from text)."
            except json.JSONDecodeError:
                # print(f"DEBUG: Failed to parse JSON from raw output for {field_name}: {raw_output}")
                pass # Fall through to general failure
            
            # print(f"DEBUG: LLM validation for {field_name} yielded non-JSON output and fallback failed.")
            return False, f"Could not validate {field_name}. Please ensure it's correct."


    def get_name(self, user_msg: str) -> Tuple[bool, str, Optional[str]]:
        is_valid, reason = self._validate_with_llm("name", user_msg)
        if not is_valid:
            return False, f"That doesn't seem like a valid name. {reason} Could you please provide your full name?", None
        else:
            return True, f"Thanks, {user_msg}! Can I get your phone number?", user_msg

    def get_phone(self, user_msg: str) -> Tuple[bool, str, Optional[str]]:
        if not re.search(r'\d{7,}', user_msg): # Basic check: at least 7 digits
            return False, "Phone number must contain at least 7 digits. Please enter a valid phone number.", None
        
        is_valid, reason = self._validate_with_llm("phone number", user_msg)
        if not is_valid:
            return False, f"Please enter a valid phone number. {reason}", None
        else:
            return True, "Great. And your email address?", user_msg

    def get_email(self, user_msg: str) -> Tuple[bool, str, Optional[str]]:
        if "@" not in user_msg or "." not in user_msg.split("@")[-1]: # Basic check
            return False, "Email address must include '@' and a '.' in the domain. Please provide a valid email address.", None

        is_valid, reason = self._validate_with_llm("email address", user_msg)
        if not is_valid:
            return False, f"Please provide a valid email address. {reason}", None
        else:
            return True, "", user_msg # Bot response determined by graph based on next stage

class EnhancedBudgetProcessorAgent:
    def __init__(self, llm_model: ChatGoogleGenerativeAI):
        self.llm = llm_model

    def _extract_budget_with_llm(self, text: str) -> Optional[float]:
        template_str = """Your task is to extract a numerical budget amount in pounds (£) from the user's text.
User's text: "{text_input}"
Instructions:
- Identify any monetary value mentioned. Assume pounds (£) if no currency is specified.
- Convert it to a float (e.g., "1 million" -> 1000000.0, "500k" -> 500000.0, "£1,250,000" -> 1250000.0).
- If multiple numbers, prioritize the one most likely to be the budget.
- Respond with a JSON object containing two keys:
  - "budget": float (the extracted budget amount, or null if not found/confident)
  - "confidence": float (your confidence in this extraction, 0.0 to 1.0. Return null for budget if confidence < 0.6)
Example for "around 1.5m": {{"budget": 1500000.0, "confidence": 0.9}}
Example for "I don't know yet": {{"budget": null, "confidence": 0.1}}
JSON Response:"""
        prompt = PromptTemplate.from_template(template_str)
        parser = JsonOutputParser()
        chain = prompt | self.llm | parser
        str_chain_for_fallback = prompt | self.llm | StrOutputParser()

        try:
            result = chain.invoke({"text_input": text})
            if result.get("budget") is not None and result.get("confidence", 0.0) >= 0.6:
                return float(result["budget"])
            return None
        except Exception:
            # print(f"DEBUG: JsonOutputParser failed for budget. Retrying with string parsing.")
            raw_output = str_chain_for_fallback.invoke({"text_input": text})
            # print(f"DEBUG: Raw LLM output for budget: {raw_output}")
            try:
                start_index = raw_output.find('{')
                end_index = raw_output.rfind('}')
                if start_index != -1 and end_index != -1 and end_index > start_index:
                    json_str = raw_output[start_index : end_index+1]
                    result = json.loads(json_str)
                    if result.get("budget") is not None and result.get("confidence", 0.0) >= 0.6:
                        return float(result["budget"])
            except json.JSONDecodeError:
                # print(f"DEBUG: Failed to parse JSON from raw budget output.")
                pass
            return None # Fallback if string parsing also fails

    def _parse_budget_rules(self, text: str) -> Optional[float]:
        text_lower = text.lower()
        text_cleaned = re.sub(r'[£$,]', '', text_lower)
        numbers_extracted = []

        million_matches = re.findall(r'(\d+\.?\d*)\s*(?:m\b|million\b)', text_cleaned)
        for val_str in million_matches:
            try: numbers_extracted.append(float(val_str) * 1_000_000)
            except ValueError: pass
        text_cleaned = re.sub(r'\d+\.?\d*\s*(?:m\b|million\b)', '', text_cleaned)

        thousand_matches = re.findall(r'(\d+\.?\d*)\s*(?:k\b|thousand\b)', text_cleaned)
        for val_str in thousand_matches:
            try: numbers_extracted.append(float(val_str) * 1_000)
            except ValueError: pass
        text_cleaned = re.sub(r'\d+\.?\d*\s*(?:k\b|thousand\b)', '', text_cleaned)

        plain_num_matches = re.findall(r'\b\d+\.?\d*\b', text_cleaned)
        for val_str in plain_num_matches:
            try:
                num = float(val_str)
                if num > 1000 or not numbers_extracted:
                    numbers_extracted.append(num)
            except ValueError: pass
        
        return max(numbers_extracted) if numbers_extracted else None

    def process_budget(self, user_msg: str) -> Tuple[Optional[float], str]:
        budget_val = self._extract_budget_with_llm(user_msg)
        if budget_val is None:
            # print(f"DEBUG: LLM budget extraction failed for '{user_msg}'. Trying rule-based parsing.")
            budget_val = self._parse_budget_rules(user_msg)

        if budget_val is None or budget_val <= 0 :
            return None, "I couldn't understand the budget. Please provide a clear amount (e.g., '£500,000', '1.2m')."
        else:
            # print(f"DEBUG: Parsed budget: '{user_msg}' -> {budget_val}")
            return budget_val, "" # Bot response determined by graph

class EnhancedPostcodeProcessorAgent:
    def __init__(self, eligible_set: Set[str], eligible_list_original: List[str], embedding_model_instance: GoogleGenerativeAIEmbeddings, llm_model: ChatGoogleGenerativeAI):
        self.eligible_set_normalized = eligible_set # Set of normalized postcodes for exact match
        self.eligible_list_original = eligible_list_original # List of original format postcodes for FAISS
        self.embedding_model = embedding_model_instance
        self.llm = llm_model # Kept for potential future use, not currently active in validation
        self.index = None
        self.dimension = 0
        if self.eligible_list_original:
            self._build_index()

    def _build_index(self):
        try:
            if not self.eligible_list_original:
                # print("DEBUG: No eligible postcodes for FAISS index.")
                return
            
            # Embed original format postcodes for better suggestions
            embeddings = np.array(self.embedding_model.embed_documents(self.eligible_list_original)).astype('float32')
            
            if embeddings.ndim == 1:
                 if embeddings.shape[0] == 0: return # No embeddings
                 embeddings = embeddings.reshape(1, -1) # Single postcode
            if embeddings.shape[0] == 0: return

            self.dimension = embeddings.shape[1]
            if self.dimension == 0: return

            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings)
        except Exception as e:
            print(f"ERROR: Building FAISS index: {e}")
            self.index = None

    def _validate_postcode_format(self, postcode_text: str) -> Tuple[bool, str]:
        # Regex for standard UK postcode formats (e.g., SW1A 1AA, M1 1AE, W1A 0AX, CR2 6XH, DN55 1PT)
        uk_pc_pattern = r"^[A-Z]{1,2}[0-9][A-Z0-9]?\s?[0-9][A-Z]{2}$"
        
        # Check the raw input against the regex (after uppercasing)
        if not re.fullmatch(uk_pc_pattern, postcode_text.upper()):
            # Try to insert a space if it's missing and might make it valid
            temp_pc = postcode_text.upper().replace(" ", "")
            if len(temp_pc) >= 5 and len(temp_pc) <= 7: # Common lengths for spaceless postcodes
                spaced_pc = temp_pc[:-3] + " " + temp_pc[-3:]
                if re.fullmatch(uk_pc_pattern, spaced_pc):
                    return True, "Format seems valid." # Valid after auto-spacing
            # print(f"DEBUG: Postcode '{postcode_text}' failed regex.")
            return False, "Postcode does not match the typical UK format (e.g., SW1A 1AA or M1 1AE)."
        return True, "Format seems valid."

    def _find_similar_postcodes(self, postcode_query: str, k: int = 1) -> Optional[str]:
        if not self.index or self.dimension == 0: return None
        try:
            # Use the user's raw query for embedding to catch typos
            query_embedding = np.array(self.embedding_model.embed_query(postcode_query)).astype('float32').reshape(1, -1)
            
            if query_embedding.shape[1] != self.dimension:
                print(f"ERROR: Query embedding dim ({query_embedding.shape[1]}) != index dim ({self.dimension}).")
                return None

            distances, indices = self.index.search(query_embedding, k)
            
            if indices.size > 0 and indices[0][0] != -1 and indices[0][0] < len(self.eligible_list_original):
                # Return the original format postcode from the list used to build the index
                return self.eligible_list_original[indices[0][0]]
            return None
        except Exception as e:
            print(f"ERROR: FAISS similarity search for '{postcode_query}': {e}")
            return None

    def process_postcode(self, user_msg_raw: str) -> Tuple[str, bool, Optional[str], str]:
        is_format_valid, reason = self._validate_postcode_format(user_msg_raw)
        if not is_format_valid:
            return "", False, None, f"That postcode doesn't look quite right. {reason} Please try again (e.g., 'SW1A 1AA')."

        norm_pc_input = normalize_postcode(user_msg_raw)
        is_covered = norm_pc_input in self.eligible_set_normalized
        suggestion = None

        if is_covered:
            return norm_pc_input, True, None, "" # Bot response determined by graph
        else:
            # Find suggestion using original user input for better typo handling
            suggestion = self._find_similar_postcodes(user_msg_raw)
            # Ensure suggestion is genuinely different and not just a case/space variant of input
            if suggestion and normalize_postcode(suggestion) == norm_pc_input:
                suggestion = None 
            
            return norm_pc_input, False, suggestion, "" # Bot response determined by graph
