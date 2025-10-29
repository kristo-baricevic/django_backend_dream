# app/config.py
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str
    
    # Backend APIs
    DJANGO_API_URL: str = "http://localhost:8000/api"
    DREAM_ANALYZER_URL: str = "http://localhost:8000"
    
    # CORS - ADD YOUR PRODUCTION DOMAINS HERE
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "https://dream-app-nu.vercel.app",
        "https://www.dream-app-nu.vercel.app",
        "https://dream-journal-app.com",
        "https://www.dream-journal-app.com",
    ]
    
    # OpenAI Settings
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 1000
    
    # Chat Settings
    MAX_CONVERSATION_HISTORY: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()