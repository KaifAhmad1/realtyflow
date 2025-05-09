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

class AgentIntent(str, Enum):
    BUY = "buy"
    SELL = "sell"
    UNKNOWN = "unknown"

class AgentBuyType(str, Enum):
    NEW_HOME = "new_home"
    RE_SALE = "re_sale"
    UNKNOWN = "unknown"

class AgentYesNo(str, Enum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"

AgentChatState = Dict[str, Any] # Placeholder for type hinting

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

    def classify_intent(self, user_message: str) -> AgentIntent:
        examples = [
            {"input": "I want to buy a house", "output": "buy"}, {"input": "I'm looking to sell my apartment", "output": "sell"},
            {"input": "Purchase property", "output": "buy"}, {"input": "List my home", "output": "sell"},
        ]
        chain = self._build_classification_chain("User Intent", [AgentIntent.BUY.value, AgentIntent.SELL.value],
                                            "Determine if the user primarily wants to buy or sell a property.", examples)
        try:
            result = chain.invoke({"user_message": user_message}).strip().lower()
            return AgentIntent(result) if result in [AgentIntent.BUY.value, AgentIntent.SELL.value] else AgentIntent.UNKNOWN
        except Exception as e:
            print(f"Error in intent classification for '{user_message}': {e}")
            return AgentIntent.UNKNOWN

    def classify_buy_type(self, user_message: str) -> AgentBuyType:
        examples = [
            {"input": "A new build", "output": "new_home"}, {"input": "Something pre-owned", "output": "re_sale"},
        ]
        chain = self._build_classification_chain("Property Type for Buyer", [AgentBuyType.NEW_HOME.value, AgentBuyType.RE_SALE.value],
                                            "Determine if the buyer is looking for a new home or a re-sale property.", examples)
        try:
            result = chain.invoke({"user_message": user_message}).strip().lower()
            return AgentBuyType(result) if result in [AgentBuyType.NEW_HOME.value, AgentBuyType.RE_SALE.value] else AgentBuyType.UNKNOWN
        except Exception as e:
            print(f"Error in buy type classification for '{user_message}': {e}")
            return AgentBuyType.UNKNOWN

    def classify_yes_no(self, user_message: str) -> AgentYesNo:
        examples = [
            {"input": "Yes, please", "output": "yes"}, {"input": "Nope", "output": "no"},
            {"input": "Sure", "output": "yes"}, {"input": "Not right now", "output": "no"},
        ]
        chain = self._build_classification_chain("Yes/No Answer", [AgentYesNo.YES.value, AgentYesNo.NO.value],
                                            "Determine if the user's response is affirmative (yes) or negative (no).", examples)
        try:
            result = chain.invoke({"user_message": user_message}).strip().lower()
            return AgentYesNo(result) if result in [AgentYesNo.YES.value, AgentYesNo.NO.value] else AgentYesNo.UNKNOWN
        except Exception as e:
            print(f"Error in yes/no classification for '{user_message}': {e}")
            return AgentYesNo.UNKNOWN

class EnhancedInfoGathererAgent:
    def __init__(self, llm_model: ChatGoogleGenerativeAI):
        self.llm = llm_model

    def _validate_with_llm(self, field_name: str, value_to_validate: str) -> Tuple[bool, str]:
        template_str = """You are a data validation assistant.
Task: Check if the provided value for the field '{field}' is plausible and correctly formatted.
Input Value: {input_value}

Instructions:
- Analyze the input value.
- For 'name', it should look like a real name (not gibberish, very short, or excessively long).
- For 'phone', it should resemble a phone number (mostly digits, appropriate length, can include common symbols like + or spaces).
- For 'email', it must contain '@' and a '.' in the domain part.
- Respond with a JSON object containing two keys:
- "is_valid": boolean (true if valid, false otherwise)
- "reason": string (brief explanation if invalid, or "Valid." if valid)

Example for an invalid name: {{"is_valid": false, "reason": "Name appears to be too short or not a typical name."}}
Example for a valid email: {{"is_valid": true, "reason": "Valid email format."}}

JSON Response:"""
        prompt = PromptTemplate.from_template(template_str)
        parser = JsonOutputParser(pydantic_object=None)
        chain = prompt | self.llm | parser
        str_chain_for_fallback = prompt | self.llm | StrOutputParser()

        try:
            result = chain.invoke({"field": field_name, "input_value": value_to_validate})
            return result.get("is_valid", False), result.get("reason", f"Validation inconclusive for {field_name}")
        except Exception: # Fallback for any JSON parsing error or LLM error
            raw_output = str_chain_for_fallback.invoke({"field": field_name, "input_value": value_to_validate})
            # Try to infer from raw output if specific keywords are present
            if "\"is_valid\": true" in raw_output.lower() or "valid." in raw_output.lower():
                return True, "Valid (inferred from text)."
            if "\"is_valid\": false" in raw_output.lower() or "invalid" in raw_output.lower():
                reason_match = re.search(r"\"reason\":\s*\"(.*?)\"", raw_output, re.IGNORECASE)
                reason = reason_match.group(1) if reason_match else "Invalid (inferred from text)."
                return False, reason
            return False, f"Error validating {field_name}: LLM response format unclear. Raw: {raw_output[:100]}"

    def get_name(self, state: AgentChatState, user_msg: str) -> Tuple[bool, str, Optional[str]]: # is_valid, response_msg, name_value
        is_valid, reason = self._validate_with_llm("name", user_msg)
        if not is_valid:
            return False, f"That doesn't seem like a valid name. {reason} Could you please provide your full name?", None
        else:
            return True, f"Thanks, {user_msg}! Can I get your phone number?", user_msg

    def get_phone(self, state: AgentChatState, user_msg: str) -> Tuple[bool, str, Optional[str]]:
        if not re.search(r'\d{7,}', user_msg): # Basic check for at least 7 digits
            return False, "Phone number must contain at least 7 digits. Please enter a valid phone number.", None
        
        is_valid, reason = self._validate_with_llm("phone number", user_msg)
        if not is_valid:
            return False, f"Please enter a valid phone number. {reason}", None
        else:
            return True, "Great. And your email address?", user_msg

    def get_email(self, state: AgentChatState, user_msg: str) -> Tuple[bool, str, Optional[str]]:
        if "@" not in user_msg or "." not in user_msg.split("@")[-1]: # Basic format check
            return False, "Email address must include '@' and a '.' in the domain. Please provide a valid email address.", None

        is_valid, reason = self._validate_with_llm("email address", user_msg)
        if not is_valid:
            return False, f"Please provide a valid email address. {reason}", None
        else:
            return True, "", user_msg # Bot message determined by graph based on intent

class EnhancedBudgetProcessorAgent:
    def __init__(self, llm_model: ChatGoogleGenerativeAI):
        self.llm = llm_model

    def _extract_budget_with_llm(self, text: str) -> Optional[float]:
        template_str = """Your task is to extract a numerical budget amount in pounds (£) from the user's text.
User's text: "{text_input}"
Instructions:
- Identify any monetary value mentioned.
- Convert it to a float (e.g., "1 million" -> 1000000.0, "500k" -> 500000.0, "£1,250,000" -> 1250000.0).
- If multiple numbers, prioritize the one most likely to be the budget.
- Respond with a JSON object containing two keys:
- "budget": float (the extracted budget amount, or null if not found)
- "confidence": float (your confidence in this extraction, 0.0 to 1.0)
Example for "around 1.5m": {{"budget": 1500000.0, "confidence": 0.9}}
JSON Response:"""
        prompt = PromptTemplate.from_template(template_str)
        parser = JsonOutputParser()
        chain = prompt | self.llm | parser
        try:
            result = chain.invoke({"text_input": text})
            if result.get("budget") is not None and result.get("confidence", 0.0) >= 0.5: # Confidence threshold
                return float(result["budget"])
        except Exception as e:
            print(f"LLM budget extraction (JSON) failed for '{text}': {e}")
        return None # Fallback if LLM fails or confidence is low

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
                if num > 1000 or not numbers_extracted: numbers_extracted.append(num)
            except ValueError: pass
        
        return max(numbers_extracted) if numbers_extracted else None

    def process_budget(self, user_msg: str) -> Tuple[Optional[float], str]: # budget_val, response_msg
        budget_val = self._extract_budget_with_llm(user_msg)
        if budget_val is None:
            budget_val = self._parse_budget_rules(user_msg)

        if budget_val is None or budget_val <= 0:
            return None, "I couldn't understand the budget. Please provide a clear amount (e.g., '£500,000', '1.2m')."
        return budget_val, "" # Bot message determined by graph

class EnhancedPostcodeProcessorAgent:
    def __init__(self, eligible_set: Set[str], eligible_list_raw: List[str], embedding_model_instance: GoogleGenerativeAIEmbeddings, llm_model: ChatGoogleGenerativeAI):
        self.eligible_set_normalized = eligible_set # Set of normalized postcodes for exact match
        self.eligible_list_raw = eligible_list_raw # List of original format postcodes for FAISS suggestions
        self.embedding_model = embedding_model_instance
        self.llm = llm_model # Not used for validation in this version, but available
        self.index = None
        self.dimension = 0
        if self.eligible_list_raw: # Build index from raw list
            self._build_index()

    def _build_index(self):
        try:
            # Embed the original (non-normalized) eligible postcodes for better suggestions
            str_eligible_list_raw = [str(pc) for pc in self.eligible_list_raw]
            embeddings = np.array(self.embedding_model.embed_documents(str_eligible_list_raw)).astype('float32')
            
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1) if embeddings.shape[0] > 0 else np.array([])

            if embeddings.shape[0] == 0 or embeddings.shape[1] == 0:
                print("FAISS: No valid embeddings generated for postcodes. Index not built.")
                return

            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings)
            print(f"FAISS index built with {len(self.eligible_list_raw)} raw postcodes, dimension {self.dimension}.")
        except Exception as e:
            print(f"Error building FAISS index: {e}")
            self.index = None

    def _validate_postcode_format(self, postcode_text: str) -> Tuple[bool, str]:
        # UK Postcode Regex (allows for space, case-insensitive matching)
        uk_pc_pattern = r"^[A-Z]{1,2}[0-9][A-Z0-9]?\s?[0-9][A-Z]{2}$"
        if not re.fullmatch(uk_pc_pattern, postcode_text.upper()):
            return False, "Postcode does not match the typical UK format (e.g., SW1A 1AA or M1 1AE)."
        return True, "Format seems valid."

    def _find_similar_postcodes(self, postcode_query: str, k: int = 1) -> Optional[str]:
        if not self.index or self.dimension == 0: return None
        try:
            # Query with user's raw input for better typo correction via embeddings
            query_embedding = np.array(self.embedding_model.embed_query(postcode_query)).astype('float32').reshape(1, -1)
            if query_embedding.shape[1] != self.dimension: return None # Dimension mismatch

            distances, indices = self.index.search(query_embedding, k)
            if indices.size > 0 and indices[0][0] != -1 and indices[0][0] < len(self.eligible_list_raw):
                return self.eligible_list_raw[indices[0][0]] # Return original format from raw list
        except Exception as e:
            print(f"Error during FAISS similarity search for '{postcode_query}': {e}")
        return None

    def process_postcode(self, user_msg_raw: str) -> Tuple[str, bool, Optional[str], str]:
        # Returns: normalized_pc, is_covered, suggestion (raw format), error_or_response_msg
        is_format_valid, reason = self._validate_postcode_format(user_msg_raw)
        if not is_format_valid:
            return "", False, None, f"That postcode doesn't look quite right. {reason} Please try again."

        norm_pc = normalize_postcode(user_msg_raw)
        is_covered = norm_pc in self.eligible_set_normalized # Check normalized against normalized set
        suggestion = None

        if not is_covered:
            suggestion = self._find_similar_postcodes(user_msg_raw)
            # Avoid suggesting the same (differently formatted) input
            if suggestion and normalize_postcode(suggestion) == norm_pc:
                suggestion = None
        
        return norm_pc, is_covered, suggestion, "" # Bot message determined by graph
