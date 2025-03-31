import yaml
from pydantic_settings import BaseSettings
from pydantic import Field

class AgentStorageConfig(BaseSettings):
    storage_api_url: str = Field(default="http://localhost:8001")
    db_name: str = Field(default="market_agents")
    user: str = Field(default="db_user")
    password: str = Field(default="db_pwd@123")
    host: str = Field(default="localhost")
    port: str = Field(default="5433")
    pool_min: int = Field(default=1)
    pool_max: int = Field(default=10)
    index_method: str = Field(default="ivfflat")
    lists: int = Field(default=100)
    embedding_api_url: str = Field(default="http://38.128.232.35:8080/embed")
    model: str = Field(default="jinaai/jina-embeddings-v2-base-en")
    batch_size: int = Field(default=32)
    timeout: int = Field(default=10)
    retry_attempts: int = Field(default=3)
    retry_delay: float = Field(default=1.0)
    retry_max_delay: int = Field(default=60)
    retry_backoff_factor: float = Field(default=2.0)
    retry_jitter: float = Field(default=0.1)
    min_chunk_size: int = Field(default=512)
    max_chunk_size: int = Field(default=1024)
    vector_dim: int = Field(default=768, description="Options: 768, 1536, etc.")
    max_input: int = Field(default=4096)
    top_k: int = Field(default=3)
    similarity_threshold: float = Field(default=0.7)
    encoding_format: str = Field(default="float")
    embedding_provider: str = Field(default="tei", description="Options: tei, openai, etc.")
    stm_top_k: int = Field(default=2)
    ltm_top_k: int = Field(default=1)
    kb_top_k: int = Field(default=3)

def load_config_from_yaml(yaml_path: str = "config.yaml") -> AgentStorageConfig:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return AgentStorageConfig(**data)
