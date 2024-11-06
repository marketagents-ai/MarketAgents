from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):
    # Database
    DATABASE_URI: str = "sqlite:///./chat.db"
    DB_ECHO: bool = True
    
    # API
    API_V1_STR: str = "/api/v1"
    
    # OpenAI
    OPENAI_KEY: Optional[str] = None
    OPENAI_MODEL: Optional[str] = None
    OPENAI_CONTEXT_LENGTH: Optional[str] = None
    
    # Anthropic
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: Optional[str] = None
    ANTHROPIC_CONTEXT_LENGTH: Optional[str] = None
    
    # VLLM
    VLLM_API_KEY: Optional[str] = None
    VLLM_ENDPOINT: Optional[str] = None
    VLLM_MODEL: Optional[str] = None
    
    # LiteLLM
    LITELLM_API_KEY: Optional[str] = None
    LITELLM_ENDPOINT: Optional[str] = None
    LITELLM_MODEL: Optional[str] = None

    # Default Model Settings
    DEFAULT_MAX_TOKENS: int = 2500
    DEFAULT_TEMPERATURE: float = 0

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "allow"

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()