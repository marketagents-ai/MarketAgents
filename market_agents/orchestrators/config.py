# config.py

from pydantic import BaseModel, Field
from typing import List, Dict, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
from pathlib import Path

class AgentConfig(BaseModel):
    num_units: int
    buyer_base_value: float
    seller_base_value: float
    use_llm: bool
    buyer_initial_cash: float
    buyer_initial_goods: int
    seller_initial_cash: float
    seller_initial_goods: int
    good_name: str
    noise_factor: float
    max_relative_spread: float

class AuctionConfig(BaseModel):
    name: str
    address: str
    max_rounds: int
    good_name: str

class GroupChatConfig(BaseModel):
    name: str
    address: str
    max_rounds: int
    initial_topic: str
    sub_rounds: int = Field(default=3)
    group_size: int = Field(default=100)

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
    environment_configs: Dict[str, Union[AuctionConfig, GroupChatConfig]]
    environment_order: List[str]
    protocol: str
    database_config: DatabaseConfig = DatabaseConfig()

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

def load_config(config_path: Path) -> OrchestratorConfig:
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return OrchestratorConfig(**config_dict)
