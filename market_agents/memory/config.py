import yaml
from pydantic_settings import BaseSettings
from pydantic import Field

class MarketMemoryConfig(BaseSettings):
    dbname: str = Field(default="my_first_table")
    user: str = Field(default="db_user")
    password: str = Field(default="password")
    host: str = Field(default="localhost")
    port: str = Field(default="0000")
    index_method: str = Field(default="ivfflat") 
    lists: int = Field(default=100)
    embedding_api_url: str = Field(default="http://0.0.0.0:8080/embed")
    model: str = Field(default="jinaai/jina-embeddings-v2-base-en")
    batch_size: int = Field(default=32)
    timeout: int = Field(default=10)
    retry_attempts: int = Field(default=3)
    retry_delay: float = Field(default=1.0)
    min_chunk_size: int = Field(default=64)
    max_chunk_size: int = Field(default=256)
    vector_dim: int = Field(default=768)
    context_window: int = Field(default=512)
    top_k: int = Field(default=3)
    similarity_threshold: float = Field(default=0.7)
    encoding_format: str = Field(default="float")
    model_type: str = Field(default="local")

def load_config_from_yaml(yaml_path: str = "config.yaml") -> MarketMemoryConfig:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return MarketMemoryConfig(**data)
