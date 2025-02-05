# config.py

from pydantic import BaseModel, Field
from typing import List, Dict, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
from pathlib import Path

class AgentConfig(BaseModel):
    knowledge_base: str
    use_llm: bool

class GroupChatConfig(BaseModel):
    name: str
    initial_topic: str
    sub_rounds: int = Field(default=3)
    group_size: int = Field(default=100)
    api_url: str = Field(default="http://localhost:8001")

class ResearchConfig(BaseModel):
    name: str
    initial_topic: str
    sub_rounds: int = Field(default=3)
    group_size: int = Field(default=100)
    schema_model: str = Field(..., description="Name of Pydantic model class from research_schemas.py")
    api_url: str = Field(default="http://localhost:8002")

class LLMConfigModel(BaseModel):
    name: str
    client: str
    model: str
    temperature: float
    max_tokens: int
    use_cache: bool

class DatabaseConfig(BaseSettings):
    db_type: str = "postgres"
    db_name: str = "market_simulation"
    db_user: str = Field(..., env='DB_USER')
    db_password: str = Field(..., env='DB_PASSWORD')
    db_host: str = Field('localhost', env='DB_HOST')
    db_port: str = Field('5432', env='DB_PORT')

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

class OrchestratorConfig(BaseSettings):
    num_agents: int
    max_rounds: int
    agent_config: AgentConfig
    llm_configs: List[LLMConfigModel]
    environment_configs: Dict[str, Union[GroupChatConfig, ResearchConfig]]
    environment_order: List[str]
    protocol: str
    database_config: DatabaseConfig = DatabaseConfig()
    tool_mode: bool
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

def load_config(config_path: Path) -> OrchestratorConfig:
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return OrchestratorConfig(**config_dict)
