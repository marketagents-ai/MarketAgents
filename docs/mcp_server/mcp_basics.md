# MCP Server Basics

## Overview

Model-Centric Programs (MCP) servers provide a standardized way for language models to interact with external tools and data sources. In the MarketAgents framework, MCP servers enable agents to access specialized capabilities beyond their built-in functions.

## MCP Architecture

The MCP architecture consists of several key components:

1. **MCP Server**: A service that exposes tools via a standardized API
2. **MCP Tools**: Functions that perform specific tasks (e.g., fetching stock data)
3. **MCP Client**: Interface for agents to call tools on the server
4. **MCP Orchestrator**: Component that manages communication between agents and MCP servers

## How MCP Servers Work

MCP servers operate on a simple request-response model:

1. An agent sends a request to call a specific tool with parameters
2. The MCP server executes the tool function with the provided parameters
3. The server returns the result to the agent
4. The agent processes the result and incorporates it into its reasoning

## MCP Server in MarketAgents

In the MarketAgents framework, MCP servers are implemented using the `FastMCP` library, which provides a simple way to create and deploy MCP servers:

```python
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("ExampleServer")

# Define a tool
@mcp.tool()
def hello_world(name: str) -> str:
    """
    Return a greeting message.
    """
    return f"Hello, {name}!"

# Run the server
if __name__ == "__main__":
    mcp.run()
```

## MCP Tool Definition

MCP tools are defined as Python functions with type annotations:

```python
@mcp.tool()
def calculate_compound_interest(
    principal: float,
    rate: float,
    time: float,
    compounds_per_year: int = 1
) -> float:
    """
    Calculate compound interest.
    
    Args:
        principal: Initial investment amount
        rate: Annual interest rate (decimal)
        time: Time period in years
        compounds_per_year: Number of times interest is compounded per year
    
    Returns:
        The final amount after compound interest
    """
    return principal * (1 + rate/compounds_per_year)**(compounds_per_year*time)
```

## MCP Server Deployment

MCP servers can be deployed as standalone services:

```bash
# Run an MCP server
python market_agents/orchestrators/mcp_server/finance_mcp_server.py
```

By default, the server runs on `localhost:8000`, but this can be configured.

## MCP Integration in MarketAgents

The MarketAgents framework provides built-in support for MCP servers through the `MCPServerEnvironment`:

```python
from market_agents.environments.mechanisms.mcp_server import MCPServer, MCPServerEnvironment

# Create MCP server mechanism
mcp_mechanism = MCPServer(
    server_url="http://localhost:8000",
    server_name="FinanceDataServer"
)

# Create MCP server environment
mcp_env = MCPServerEnvironment(
    name="finance_tools",
    address="finance_tools_env",
    mechanism=mcp_mechanism
)
```

## Available MCP Servers

The MarketAgents framework includes several built-in MCP servers:

1. **Finance Tools MCP Server**: Provides financial data and analysis tools
2. **Research Tools MCP Server**: Provides research and information gathering tools
3. **Data Analysis MCP Server**: Provides data processing and visualization tools

In the next section, we'll explore the Finance Tools MCP Server in detail.
