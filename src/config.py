import os
from dotenv import load_dotenv

# Load environment variables from .env file in the project root
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env') 
load_dotenv(dotenv_path=dotenv_path)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not found. Please set it in the .env file in the project root.")

MIN_BUDGET_NEW_HOME = 1_000_000
COMPANY_PHONE_NUMBER = "1800 111 222" 
MAX_ATTEMPTS = 3
POSTCODE_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets', 'uk_postcodes 1.csv')
