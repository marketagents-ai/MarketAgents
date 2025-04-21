from typing import Dict, List
from pydantic_settings import BaseSettings
from pydantic import Field

class WebSearchHeaders(BaseSettings):
    """Headers configuration for web search"""
    User_Agent: str = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        alias="User-Agent"
    )
    Accept: str = Field(
        default="text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    )
    Accept_Language: str = Field(
        default="en-US,en;q=0.5",
        alias="Accept-Language"
    )

    def to_dict(self) -> Dict[str, str]:
        """Convert headers to dictionary format"""
        return {
            "User-Agent": self.User_Agent,
            "Accept": self.Accept,
            "Accept-Language": self.Accept_Language
        }

class WebSearchConfig(BaseSettings):
    """Configuration for web search operations"""
    max_concurrent_requests: int = Field(default=50)
    rate_limit: float = Field(default=0.1)
    content_max_length: int = Field(default=4000)
    request_timeout: int = Field(default=30)
    urls_per_query: int = Field(default=5)
    use_ai_summary: bool = Field(default=True)
    methods: List[str] = Field(default=[
        "selenium",
        "playwright",
        "beautifulsoup",
        "newspaper3k",
        "scrapy",
        "requests_html",
        "mechanicalsoup",
        "httpx"
    ])
    default_method: str = Field(default="newspaper3k")
    headers: WebSearchHeaders = Field(default_factory=WebSearchHeaders)

    class Config:
        env_prefix = "web_search_"
        case_sensitive = False
        populate_by_name = True

    @classmethod
    def from_yaml(cls, config_dict: Dict) -> 'WebSearchConfig':
        """Create config from YAML dictionary"""
        if "headers" in config_dict:
            config_dict["headers"] = WebSearchHeaders(**config_dict["headers"])
        return cls(**config_dict)