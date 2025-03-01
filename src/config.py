import os
from dotenv import load_dotenv
from functools import lru_cache
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    # Required settings
    OPENAI_API_KEY: str
    ETNOSUR_ASSISTANT_ID: str | None = None  # Make it optional
    WHATSAPP_TOKEN: str
    PHONE_NUMBER_ID: str
    WEBHOOK_VERIFY_TOKEN: str
    
    # Optional settings with defaults
    BASE_URL: str = "http://localhost:5000/api"
    PORT: int = 8080

    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings():
    return Settings()

# Create a global settings instance
settings = get_settings()