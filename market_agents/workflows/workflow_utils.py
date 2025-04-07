"""Utility functions for working with MCP server environments."""

from importlib import import_module, util
from typing import Type, Any
import logging
from market_agents.environments.environment import MultiAgentEnvironment
from market_agents.environments.mechanisms.mcp_server import (
    MCPServerEnvironment,
    MCPServerEnvironmentConfig,
    MCPServerMechanism
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_environment_class(mechanism_type: str) -> Type[MultiAgentEnvironment]:
    """Get the environment class for a given mechanism type.
    
    Args:
        mechanism_type: The type of mechanism (e.g. 'mcp_server')
        
    Returns:
        The environment class for the mechanism
        
    Raises:
        ValueError: If the mechanism module cannot be loaded or no environment class is found
    """
    try:
        module = import_module(f"market_agents.environments.mechanisms.{mechanism_type}")
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, MultiAgentEnvironment) and 
                attr != MultiAgentEnvironment):
                return attr
        raise ValueError(f"No environment class found for mechanism {mechanism_type}")
    except ImportError as e:
        raise ValueError(f"Could not load mechanism: {mechanism_type}") from e

def initialize_environment(config: MCPServerEnvironmentConfig) -> MCPServerEnvironment:
    """Initialize an environment from config.
    
    Args:
        config: The environment configuration
        
    Returns:
        The initialized environment instance
        
    Raises:
        ValueError: If environment initialization fails
    """
    try:
        # Get the environment class
        environment_class = get_environment_class(config.mechanism)
        
        # Convert config to dict if needed
        env_params = config.model_dump() if hasattr(config, 'model_dump') else dict(config)
        
        # Create the environment instance
        return environment_class(**env_params)
        
    except Exception as e:
        raise ValueError(f"Failed to initialize environment: {e}") from e

async def setup_mcp_environment(config: MCPServerEnvironmentConfig) -> MCPServerEnvironment:
    """Set up and initialize an MCP server environment.
    
    Args:
        config: The MCP server environment configuration
        
    Returns:
        The initialized MCP server environment
        
    Raises:
        ValueError: If environment setup fails
    """
    logger.info("Setting up MCP Server Environment...")
    
    try:
        # Initialize the environment
        environment = initialize_environment(config)
        
        # Initialize the environment (sets up mechanism and action space)
        logger.info("Initializing environment...")
        await environment.initialize()
        
        # Verify tools were loaded
        if environment.action_space:
            tool_count = len(environment.action_space.allowed_actions)
            logger.info(f"Environment initialized with {tool_count} tools")
            logger.info("Available tools: %s", 
                       [tool.name for tool in environment.action_space.allowed_actions])
        else:
            logger.error("Action space not initialized!")
            
        return environment
        
    except Exception as e:
        logger.error(f"Failed to set up MCP environment: {e}")
        raise