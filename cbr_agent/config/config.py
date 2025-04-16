from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    DEFAULT_HEADERS: dict = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    @classmethod
    def validate(cls):
        """Validate required environment variables are set"""
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        if not cls.API_BASE_URL:
            raise ValueError("API_BASE_URL environment variable is not set") 