# RealtyFlow AI Agent

RealtyFlow AI is an intelligent conversational agent designed to streamline initial customer interactions for real estate businesses. It qualifies leads by collecting necessary information based on whether they want to buy or sell property.

Built with Python, this application leverages LangGraph for state management, Google Gemini for natural language understanding and generation, and FAISS for efficient postcode similarity searching.

## Features

*   **Lead Qualification:** Automates the initial conversation to qualify leads for buying or selling.
*   **Intent Recognition:** Understands user intent (buy/sell, new home/re-sale, yes/no).
*   **Information Gathering:** Collects contact details (name, phone, email), budget, and location preferences (postcode).
*   **Postcode Validation & Suggestion:** Checks postcode coverage against an eligible list and suggests similar valid postcodes using FAISS for typo correction.
*   **Structured Dialogue:** Follows a predefined decision tree for consistent information gathering.
*   **Error Handling:** Gracefully handles misunderstandings and offers to restart or refer to the office.
*   **Dark Themed UI:** Interactive and visually appealing user interface built with Streamlit.

## Core Technologies

*   **Orchestration:** LangGraph
*   **LLM:** Google Gemini (via `langchain-google-genai`)
*   **Embeddings:** Google Generative AI Embeddings
*   **Vector Search:** FAISS (for postcode similarity)
*   **UI:** Streamlit
*   **Data Handling:** Pandas

## Project Structure


```
realtyflow/
├── .env # For API keys (MUST NOT be committed to Git)
├── .gitignore
├── README.md # This file
├── requirements.txt # Python dependencies
├── assets/
│ └── uk_postcodes.csv # List of eligible postcodes
├── app.py # Main Streamlit application
└── src/
├── init.py
├── config.py # Configuration variables (API keys, constants)
├── utils.py # Utility functions (data loading, normalization)
├── agents.py # Agent classes for specific tasks (classification, validation)
└── chatbot_engine.py # LangGraph setup, state definitions, conversational flow logic
```
