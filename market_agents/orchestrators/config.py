# config.py

from enum import Enum
from pydantic import BaseModel, Field, validator
from typing import Any, List, Dict, Optional, Type, Union
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

class EnvironmentConfig(BaseSettings):
    name: str = Field(
        ...,
        description="Name of the group chat environment"
    )
    api_url: Optional[str] = Field(
        default=None,
        description="API endpoint for the environment"
    )
    model_config = {
        "extra": "allow"
    }

class GroupChatConfig(EnvironmentConfig):
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
        default="http://localhost:8002",
        description="API endpoint for group chat environment"
    )

class ResearchConfig(EnvironmentConfig):
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

class WebResearchConfig(EnvironmentConfig):
    """Configuration for web research environment"""
    name: str = Field(
        default="web_research",
        description="Name of the web research environment"
    )
    initial_query: str = Field(
        ...,
        description="Initial search query to start the research with"
    )
    mechanism: str = Field(
        default="web_research",
        description="Mechanism type for web research"
    )
    urls_per_query: int = Field(
        default=3,
        description="Number of URLs to fetch per query"
    )
    summary_model: Optional[Type[BaseModel]] = Field(
        default=None,
        description="Pydantic model class for structuring research summaries"
    )
    max_rounds: int = Field(
        default=3,
        description="Maximum number of rounds"
    )
    sub_rounds: int = Field(
        default=1,
        description="Number of sub-rounds within each main round"
    )

    model_config = {
        "extra": "allow"
    }

class MCPServerConfig(EnvironmentConfig):
    """Configuration for MCP Server environment"""
    name: str = Field(
        ..., 
        description="Name of the MCP server environment"
    )
    mcp_server_module: str = Field(
        ..., 
        description="Module path to the MCP server"
    )
    mcp_server_class: str = Field(
        default="mcp", 
        description="Variable name of the MCP server instance"
    )
    max_rounds: int = Field(
        default=3,
        description="Maximum number of rounds for the MCP server"
    )
    sub_rounds: int = Field(
        default=2, 
        description="Number of sub-rounds per main round"
    )
    task_prompt: str = Field(
        default="", 
        description="Initial task prompt for the MCP server interaction"
    )
    api_url: str = Field(
        default="local://mcp_server",
        description="Placeholder API endpoint for MCP server (not used)"
    )
    
    model_config = {
        "extra": "allow"
    }

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
    environment_configs: Dict[str, EnvironmentConfig] = Field(
        ...,
        description="Configurations for different environments"
    )
    environment_order: List[str] = Field(
        ...,
        description="Order of environments"
    )
    tool_mode: bool = Field(
        ...,
        description="Flag to indicate if tool mode is enabled"
    )
    agent_config: Optional[AgentConfig] = Field(
        None,
        description="Optional configuration for the agent"
    )
    llm_configs: Optional[List[LLMConfigModel]] = Field(
        None,
        description="Optional list of LLM configurations"
    )
    protocol: Optional[str] = Field(
        None,
        description="Optional protocol used by the orchestrator"
    )
    database_config: DatabaseConfig = Field(
        default_factory=DatabaseConfig,
        description="Database configuration"
    )

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

def deep_update(base_dict: dict, update_dict: dict) -> dict:
    """Recursively update a dictionary."""
    for key, value in update_dict.items():
        if (
            isinstance(value, dict) 
            and key in base_dict 
            and isinstance(base_dict[key], dict)
        ):
            deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict

def load_config(config_path: Path, overrides: dict = None) -> OrchestratorConfig:
    """
    Load config from YAML and apply any overrides.
    
    Args:
        config_path: Path to the YAML config file
        overrides: Dictionary of overrides that can update any part of the config
    """
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    
    # Apply overrides if provided
    if overrides:
        config_dict = deep_update(config_dict, overrides)
        
    try:
        return OrchestratorConfig(**config_dict)
    except Exception as e:
        raise
