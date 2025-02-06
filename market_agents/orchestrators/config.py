# config.py

from pydantic import BaseModel, Field
from typing import List, Dict, Union
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
from pathlib import Path

class AgentConfig(BaseModel):
    knowledge_base: str = Field(
        ...,
        description="Knowledge base used by the agent"
    )
    use_llm: bool = Field(
        ...,
        description="Flag to indicate if LLM should be used"
    )

class GroupChatConfig(BaseModel):
    name: str = Field(
        ...,
        description="Name of the group chat environment"
    )
    initial_topic: str = Field(
        ...,
        description="Initial topic for the group chat"
    )
    sub_rounds: int = Field(
        default=3,
        description="Number of sub-rounds within each main round"
    )
    group_size: int = Field(
        default=100,
        description="Number of agents in the group chat"
    )
    api_url: str = Field(
        default="http://localhost:8001",
        description="API endpoint for group chat environment"
    )

class ResearchConfig(BaseModel):
    """Configuration for research environment orchestration"""
    name: str = Field(
        default="research",
        description="Name of the research environment"
    )
    api_url: str = Field(
        default="http://localhost:8003",
        description="API endpoint for research environment"
    )
    sub_rounds: int = Field(
        default=2,
        description="Number of sub-rounds within each main round"
    )
    initial_topic: str = Field(
        default="Market Analysis",
        description="Initial research topic"
    )
    group_size: int = Field(
        default=4,
        description="Number of agents in research group"
    )
    schema_model: str = Field(
        default="LiteraryAnalysis",
        description="Name of Pydantic model defining research output schema"
    )

class LLMConfigModel(BaseModel):
    name: str = Field(
        ...,
        description="Name of the LLM configuration"
    )
    client: str = Field(
        ...,
        description="Client used for the LLM"
    )
    model: str = Field(
        ...,
        description="Model name for the LLM"
    )
    temperature: float = Field(
        ...,
        description="Temperature setting for the LLM"
    )
    max_tokens: int = Field(
        ...,
        description="Maximum number of tokens for the LLM"
    )
    use_cache: bool = Field(
        ...,
        description="Flag to indicate if caching should be used"
    )

class DatabaseConfig(BaseSettings):
    db_type: str = Field(
        default="postgres",
        description="Type of the database"
    )
    db_name: str = Field(
        default="market_simulation",
        description="Name of the database"
    )
    db_user: str = Field(
        ...,
        env='DB_USER',
        description="Database user"
    )
    db_password: str = Field(
        ...,
        env='DB_PASSWORD',
        description="Database password"
    )
    db_host: str = Field(
        default='localhost',
        env='DB_HOST',
        description="Database host"
    )
    db_port: str = Field(
        default='5432',
        env='DB_PORT',
        description="Database port"
    )

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

class OrchestratorConfig(BaseSettings):
    num_agents: int = Field(
        ...,
        description="Number of agents in the orchestrator"
    )
    max_rounds: int = Field(
        ...,
        description="Maximum number of rounds in the orchestrator"
    )
    agent_config: AgentConfig = Field(
        ...,
        description="Configuration for the agent"
    )
    llm_configs: List[LLMConfigModel] = Field(
        ...,
        description="List of LLM configurations"
    )
    environment_configs: Dict[str, Union[GroupChatConfig, ResearchConfig]] = Field(
        ...,
        description="Configurations for different environments"
    )
    environment_order: List[str] = Field(
        ...,
        description="Order of environments"
    )
    protocol: str = Field(
        ...,
        description="Protocol used by the orchestrator"
    )
    database_config: DatabaseConfig = Field(
        default_factory=DatabaseConfig,
        description="Database configuration"
    )
    tool_mode: bool = Field(
        ...,
        description="Flag to indicate if tool mode is enabled"
    )

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

def load_config(config_path: Path) -> OrchestratorConfig:
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return OrchestratorConfig(**config_dict)
