# Agent-MCP Interaction

## Overview

Agents in the MarketAgents framework interact with MCP servers to access specialized tools and data sources. This document explains how agents call MCP tools, process the results, and incorporate them into their reasoning.

## Basic MCP Tool Calling

Agents call MCP tools by sending actions to the MCP server environment:

```python
# Define an action to call an MCP tool
action = {
    "tool_name": "get_current_stock_price",  # Name of the tool to call
    "parameters": {                          # Parameters for the tool
        "symbol": "AAPL"
    }
}

# Execute the action in the MCP environment
result = agent.environments["finance_tools"].step(action)

# Process the result
stock_price = result.get("result")
print(f"Current price of AAPL: ${stock_price}")
```

## MCP Tool Calling in Cognitive Steps

MCP tool calling is typically integrated into cognitive steps:

```python
from market_agents.agents.cognitive_steps import CognitiveStep
from pydantic import Field

class FinancialAnalysisStep(CognitiveStep):
    step_name: str = Field(default="financial_analysis", description="Financial analysis step")
    symbol: str = Field(default="AAPL", description="Stock symbol to analyze")
    
    async def execute(self, agent):
        # Call MCP tool to get stock price
        price_action = {
            "tool_name": "get_current_stock_price",
            "parameters": {"symbol": self.symbol}
        }
        price_result = agent.environments["finance_tools"].step(price_action)
        
        # Call MCP tool to get fundamentals
        fundamentals_action = {
            "tool_name": "get_stock_fundamentals",
            "parameters": {"symbol": self.symbol}
        }
        fundamentals_result = agent.environments["finance_tools"].step(fundamentals_action)
        
        # Generate analysis based on the data
        prompt = f"""
        You are {agent.role}. {agent.persona}
        
        Analyze the following financial data for {self.symbol}:
        
        Current Price: ${price_result.get('result', 'N/A')}
        
        Fundamentals:
        {self._format_dict(fundamentals_result.get('result', {}))}
        
        Based on this data, provide a financial analysis of {self.symbol}.
        """
        
        response = await agent.llm_orchestrator.generate(
            model=agent.llm_config.model,
            messages=[{"role": "system", "content": prompt}]
        )
        
        return response.content
    
    def _format_dict(self, data):
        if not data:
            return "No data available"
        
        formatted = ""
        for key, value in data.items():
            formatted += f"- {key}: {value}\n"
        return formatted
```

## Handling MCP Tool Results

MCP tool results are returned as dictionaries with the following structure:

```python
{
    "success": True,                # Whether the tool call was successful
    "result": {...},                # The result of the tool call
    "error": None,                  # Error message if the call failed
    "tool_name": "get_stock_fundamentals",  # Name of the tool that was called
    "parameters": {"symbol": "AAPL"}  # Parameters that were passed to the tool
}
```

You should always check if the tool call was successful before using the result:

```python
# Call MCP tool
result = agent.environments["finance_tools"].step(action)

# Check if the call was successful
if result.get("success", False):
    # Process the result
    data = result.get("result", {})
    # ...
else:
    # Handle the error
    error_message = result.get("error", "Unknown error")
    print(f"Tool call failed: {error_message}")
```

## Chaining MCP Tool Calls

You can chain multiple MCP tool calls to build complex workflows:

```python
# Get a list of stocks in a sector
sector_stocks_action = {
    "tool_name": "get_sector_stocks",
    "parameters": {"sector": "Technology"}
}
sector_result = agent.environments["finance_tools"].step(sector_stocks_action)

# For each stock, get the current price
if sector_result.get("success", False):
    stocks = sector_result.get("result", [])
    prices = {}
    
    for symbol in stocks:
        price_action = {
            "tool_name": "get_current_stock_price",
            "parameters": {"symbol": symbol}
        }
        price_result = agent.environments["finance_tools"].step(price_action)
        
        if price_result.get("success", False):
            prices[symbol] = price_result.get("result", 0.0)
    
    # Find the stocks with the highest prices
    top_stocks = sorted(prices.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 highest-priced technology stocks:")
    for symbol, price in top_stocks:
        print(f"{symbol}: ${price}")
```

## Error Handling

When calling MCP tools, it's important to handle errors gracefully:

```python
try:
    # Call MCP tool
    result = agent.environments["finance_tools"].step(action)
    
    # Check if the call was successful
    if result.get("success", False):
        # Process the result
        data = result.get("result", {})
        # ...
    else:
        # Handle the error
        error_message = result.get("error", "Unknown error")
        print(f"Tool call failed: {error_message}")
        
        # Try an alternative approach
        alternative_action = {
            "tool_name": "get_company_profile",
            "parameters": {"symbol": symbol}
        }
        alternative_result = agent.environments["finance_tools"].step(alternative_action)
        # ...
except Exception as e:
    print(f"An error occurred: {str(e)}")
    # Fallback to a default value or alternative approach
```

## MCP Tool Discovery

Agents can discover available MCP tools:

```python
# Get available tools from the MCP server
tools_action = {
    "tool_name": "__get_available_tools",
    "parameters": {}
}
tools_result = agent.environments["finance_tools"].step(tools_action)

# Process the result
if tools_result.get("success", False):
    available_tools = tools_result.get("result", [])
    print("Available tools:")
    for tool in available_tools:
        print(f"- {tool['name']}: {tool['description']}")
```

## Complete Example: Agent-MCP Interaction

Here's a complete example demonstrating agent interaction with MCP tools:

```python
import asyncio
from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.agents.personas.persona import Persona
from market_agents.environments.mechanisms.mcp_server import MCPServer, MCPServerEnvironment
from market_agents.agents.cognitive_steps import CognitiveStep, CognitiveEpisode
from minference.lite.models import LLMConfig

class StockAnalysisStep(CognitiveStep):
    step_name: str = Field(default="stock_analysis", description="Stock analysis step")
    symbols: list = Field(default=["AAPL", "MSFT", "GOOGL"], description="Stock symbols to analyze")
    
    async def execute(self, agent):
        analysis_results = {}
        
        for symbol in self.symbols:
            # Get current price
            price_action = {
                "tool_name": "get_current_stock_price",
                "parameters": {"symbol": symbol}
            }
            price_result = agent.environments["finance_tools"].step(price_action)
            
            # Get fundamentals
            fundamentals_action = {
                "tool_name": "get_stock_fundamentals",
                "parameters": {"symbol": symbol}
            }
            fundamentals_result = agent.environments["finance_tools"].step(fundamentals_action)
            
            # Store results
            analysis_results[symbol] = {
                "price": price_result.get("result", "N/A"),
                "fundamentals": fundamentals_result.get("result", {})
            }
        
        # Generate comparative analysis
        prompt = f"""
        You are {agent.role}. {agent.persona}
        
        Analyze and compare the following stocks:
        
        {self._format_analysis_results(analysis_results)}
        
        Provide a comparative analysis of these stocks, including:
        1. Financial health comparison
        2. Valuation metrics comparison
        3. Investment recommendation for each stock
        4. Overall portfolio recommendation
        """
        
        response = await agent.llm_orchestrator.generate(
            model=agent.llm_config.model,
            messages=[{"role": "system", "content": prompt}]
        )
        
        return response.content
    
    def _format_analysis_results(self, results):
        formatted = ""
        for symbol, data in results.items():
            formatted += f"## {symbol}\n"
            formatted += f"Current Price: ${data['price']}\n\n"
            formatted += "Fundamentals:\n"
            for key, value in data['fundamentals'].items():
                formatted += f"- {key}: {value}\n"
            formatted += "\n"
        return formatted

async def run_stock_analysis():
    # Create storage utilities
    storage_utils = AgentStorageAPIUtils(config=storage_config)
    
    # Create persona
    persona = Persona(
        role="Investment Analyst",
        persona="I am an investment analyst specializing in technology stocks.",
        objectives=["Analyze tech stocks", "Provide investment recommendations"]
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
        server_name="FinanceDataServer"
    )
    finance_env = MCPServerEnvironment(
        name="finance_tools",
        mechanism=finance_mcp
    )
    
    # Add environment to agent
    agent.environments["finance_tools"] = finance_env
    
    # Run stock analysis
    analysis = await agent.run_step(
        step=StockAnalysisStep(symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "META"])
    )
    
    return analysis

# Run the analysis
analysis = asyncio.run(run_stock_analysis())
print(analysis)
```

In the next section, we'll explore how to create custom MCP servers for specialized use cases.
