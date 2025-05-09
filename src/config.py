import os
from dotenv import load_dotenv

load_dotenv() # Load variables from .env file in the project root

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found. Ensure .env file is in project root or secrets are set.")

MIN_BUDGET_NEW_HOME = 1_000_000
COMPANY_PHONE_NUMBER = "1800 111 222"
MAX_ATTEMPTS = 3
# Path relative to the project root where app.py is located
POSTCODE_FILE_PATH = "uk_postcodes 1.csv"
