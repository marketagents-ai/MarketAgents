# MCP Server Configuration

## Overview

Configuring MCP servers in the MarketAgents framework involves setting up the server, defining the connection parameters, and integrating the server with agents. This document explains how to configure MCP servers for different use cases.

## MCP Server Environment Configuration

To configure an MCP server environment, you need to specify the server URL, name, and other parameters:

```python
from market_agents.environments.mechanisms.mcp_server import MCPServer, MCPServerEnvironment

# Create MCP server mechanism
mcp_mechanism = MCPServer(
    server_url="http://localhost:8000",  # URL of the MCP server
    server_name="FinanceDataServer",     # Name of the MCP server
    timeout=30,                          # Request timeout in seconds
    retry_attempts=3                     # Number of retry attempts
)

# Create MCP server environment
mcp_env = MCPServerEnvironment(
    name="finance_tools",                # Environment name
    address="finance_tools_env",         # Environment address
    max_steps=10,                        # Maximum number of steps
    mechanism=mcp_mechanism              # MCP server mechanism
)
```

## Connection Parameters

The MCPServer mechanism supports several connection parameters:

```python
mcp_mechanism = MCPServer(
    server_url="http://localhost:8000",  # Server URL
    server_name="FinanceDataServer",     # Server name
    timeout=30,                          # Request timeout in seconds
    retry_attempts=3,                    # Number of retry attempts
    retry_delay=2,                       # Delay between retries in seconds
    headers={                            # Optional HTTP headers
        "Authorization": "Bearer your-api-key"
    }
)
```

## Multiple MCP Servers

You can configure multiple MCP servers for different purposes:

```python
# Finance MCP server
finance_mcp = MCPServer(
    server_url="http://localhost:8000",
    server_name="FinanceDataServer"
)
finance_env = MCPServerEnvironment(
    name="finance_tools",
    mechanism=finance_mcp
)

# Research MCP server
research_mcp = MCPServer(
    server_url="http://localhost:8001",
    server_name="ResearchDataServer"
)
research_env = MCPServerEnvironment(
    name="research_tools",
    mechanism=research_mcp
)

# Add both environments to the agent
agent.environments["finance_tools"] = finance_env
agent.environments["research_tools"] = research_env
```

## MCP Server in Orchestrator Configuration

You can include MCP server configuration in the orchestrator configuration:

```python
from market_agents.orchestrators.config import OrchestratorConfig

# Create orchestrator configuration with MCP server
config = OrchestratorConfig(
    environment_order=["finance_tools", "research"],
    num_agents=3,
    max_steps=5,
    mcp_servers=[
        {
            "name": "finance_tools",
            "server_url": "http://localhost:8000",
            "server_name": "FinanceDataServer"
        },
        {
            "name": "research_tools",
            "server_url": "http://localhost:8001",
            "server_name": "ResearchDataServer"
        }
    ]
)
```

## Starting MCP Servers

MCP servers need to be running before agents can interact with them. You can start them manually or use the provided scripts:

```bash
# Start the Finance MCP Server
python market_agents/orchestrators/mcp_server/finance_mcp_server.py

# Start multiple servers using the convenience script
./start_market_agents.sh
```

## MCP Server Health Check

You can check if an MCP server is running and healthy:

```python
import aiohttp
import asyncio

async def check_mcp_server_health(server_url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{server_url}/health") as response:
                if response.status == 200:
                    return True
                else:
                    return False
    except Exception as e:
        print(f"Error checking MCP server health: {e}")
        return False

# Check if the Finance MCP Server is healthy
is_healthy = asyncio.run(check_mcp_server_health("http://localhost:8000"))
print(f"Finance MCP Server is {'healthy' if is_healthy else 'unhealthy'}")
```

## Complete Example: MCP Server Configuration

Here's a complete example of configuring and using multiple MCP servers:

```python
import asyncio
from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.agents.personas.persona import Persona
from market_agents.environments.mechanisms.mcp_server import MCPServer, MCPServerEnvironment
from minference.lite.models import LLMConfig

async def configure_mcp_servers():
    # Create storage utilities
    storage_utils = AgentStorageAPIUtils(config=storage_config)
    
    # Create persona
    persona = Persona(
        role="Investment Analyst",
        persona="I am an investment analyst specializing in technology and finance.",
        objectives=["Analyze investment opportunities", "Provide market insights"]
    )
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="investment_analyst",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4",
            client="openai",
            temperature=0.7
        ),
        persona=persona
    )
    
    # Configure Finance MCP server
    finance_mcp = MCPServer(
        server_url="http://localhost:8000",
        server_name="FinanceDataServer",
        timeout=30,
        retry_attempts=3
    )
    finance_env = MCPServerEnvironment(
        name="finance_tools",
        address="finance_tools_env",
        mechanism=finance_mcp
    )
    
    # Configure Research MCP server
    research_mcp = MCPServer(
        server_url="http://localhost:8001",
        server_name="ResearchDataServer",
        timeout=30,
        retry_attempts=3
    )
    research_env = MCPServerEnvironment(
        name="research_tools",
        address="research_tools_env",
        mechanism=research_mcp
    )
    
    # Add environments to agent
    agent.environments["finance_tools"] = finance_env
    agent.environments["research_tools"] = research_env
    
    # Check if MCP servers are healthy
    async def check_server_health(server_url):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{server_url}/health") as response:
                    return response.status == 200
        except Exception:
            return False
    
    finance_healthy = await check_server_health("http://localhost:8000")
    research_healthy = await check_server_health("http://localhost:8001")
    
    print(f"Finance MCP Server: {'Healthy' if finance_healthy else 'Unhealthy'}")
    print(f"Research MCP Server: {'Healthy' if research_healthy else 'Unhealthy'}")
    
    return agent, finance_env, research_env

# Run the configuration
agent, finance_env, research_env = asyncio.run(configure_mcp_servers())
```

In the next section, we'll explore how agents interact with MCP servers to access tools and data.
