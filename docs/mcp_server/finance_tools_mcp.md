# Finance Tools MCP Server

## Overview

The Finance Tools MCP Server is a specialized toolkit included in the MarketAgents framework that provides agents with access to financial data, market information, and analysis capabilities. This server enables agents to retrieve real-time stock prices, company fundamentals, financial statements, and other market data.

## Finance MCP Server Architecture

The Finance MCP Server is built using the `FastMCP` library and integrates with financial data sources like Yahoo Finance. It exposes a set of tools that agents can call to access financial information.

## Available Finance Tools

The Finance MCP Server provides the following tools:

### 1. Get Current Stock Price

```python
@mcp.tool()
def get_current_stock_price(symbol: str) -> float:
    """
    Get the current stock price for a given symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
    
    Returns:
        Current stock price
    """
```

### 2. Get Stock Fundamentals

```python
@mcp.tool()
def get_stock_fundamentals(symbol: str) -> dict:
    """
    Get fundamental data for a given stock symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
    
    Returns:
        Dictionary containing fundamental data including:
        - company_name
        - sector
        - industry
        - market_cap
        - pe_ratio
        - pb_ratio
        - dividend_yield
        - eps
        - beta
        - 52_week_high
        - 52_week_low
    """
```

### 3. Get Financial Statements

```python
@mcp.tool()
def get_financial_statements(symbol: str) -> dict:
    """
    Get financial statements for a given stock symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
    
    Returns:
        Dictionary containing financial statement data
    """
```

### 4. Get Key Financial Ratios

```python
@mcp.tool()
def get_key_financial_ratios(symbol: str) -> dict:
    """
    Get key financial ratios for a given stock symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
    
    Returns:
        Dictionary containing various financial ratios
    """
```

### 5. Get Analyst Recommendations

```python
@mcp.tool()
def get_analyst_recommendations(symbol: str) -> dict:
    """
    Get analyst recommendations for a given stock symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
    
    Returns:
        Dictionary containing analyst recommendations
    """
```

### 6. Get Dividend Data

```python
@mcp.tool()
def get_dividend_data(symbol: str) -> dict:
    """
    Get dividend data for a given stock symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
    
    Returns:
        Dictionary containing dividend history
    """
```

### 7. Get Company News

```python
@mcp.tool()
def get_company_news(symbol: str) -> dict:
    """
    Get company news for a given stock symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
    
    Returns:
        Dictionary containing recent news articles
    """
```

### 8. Get Technical Indicators

```python
@mcp.tool()
def get_technical_indicators(symbol: str) -> dict:
    """
    Get technical indicators for a given stock symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
    
    Returns:
        Dictionary containing technical indicators
    """
```

### 9. Get Company Profile

```python
@mcp.tool()
def get_company_profile(symbol: str) -> dict:
    """
    Get company profile and overview for a given stock symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
    
    Returns:
        Dictionary containing company profile information
    """
```

## Setting Up the Finance MCP Server

To use the Finance MCP Server, you need to start it as a separate service:

```bash
# Start the Finance MCP Server
python market_agents/orchestrators/mcp_server/finance_mcp_server.py
```

By default, the server runs on `localhost:8000`.

## Finance MCP Server Implementation

Here's the implementation of the Finance MCP Server:

```python
# finance_mcp_server.py
from mcp.server.fastmcp import FastMCP
import yfinance as yf
import pandas as pd

# Create an MCP server instance with the name "FinanceDataServer"
mcp = FastMCP("FinanceDataServer")

@mcp.tool()
def get_current_stock_price(symbol: str) -> float:
    """
    Get the current stock price for a given symbol.
    """
    try:
        stock = yf.Ticker(symbol)
        # Try to fetch the regular market price; fall back to currentPrice
        price = stock.info.get("regularMarketPrice") or stock.info.get("currentPrice")
        return price if price is not None else 0.0
    except Exception as e:
        return 0.0

@mcp.tool()
def get_stock_fundamentals(symbol: str) -> dict:
    """
    Get fundamental data for a given stock symbol.
    """
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        fundamentals = {
            'symbol': symbol,
            'company_name': info.get('longName', ''),
            'sector': info.get('sector', ''),
            'industry': info.get('industry', ''),
            'market_cap': info.get('marketCap', None),
            'pe_ratio': info.get('forwardPE', None),
            'pb_ratio': info.get('priceToBook', None),
            'dividend_yield': info.get('dividendYield', None),
            'eps': info.get('trailingEps', None),
            'beta': info.get('beta', None),
            '52_week_high': info.get('fiftyTwoWeekHigh', None),
            '52_week_low': info.get('fiftyTwoWeekLow', None)
        }
        return fundamentals
    except Exception as e:
        return {}

# Additional tool implementations...

if __name__ == "__main__":
    mcp.run()
```

## Complete Example: Using the Finance MCP Server

Here's a complete example of setting up and using the Finance MCP Server:

```python
import asyncio
from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.agents.personas.persona import Persona
from market_agents.environments.mechanisms.mcp_server import MCPServer, MCPServerEnvironment
from market_agents.agents.cognitive_steps import CognitiveStep
from minference.lite.models import LLMConfig

async def use_finance_mcp_server():
    # Create storage utilities
    storage_utils = AgentStorageAPIUtils(config=storage_config)
    
    # Create persona
    persona = Persona(
        role="Financial Analyst",
        persona="I am a financial analyst specializing in technology stocks.",
        objectives=["Analyze tech company financials", "Provide investment recommendations"]
    )
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="financial_analyst",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4",
            client="openai",
            temperature=0.7
        ),
        persona=persona
    )
    
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
    
    # Add environment to agent
    agent.environments["finance_tools"] = mcp_env
    
    # Define a step to use finance tools
    class FinanceAnalysisStep(CognitiveStep):
        step_name: str = Field(default="finance_analysis", description="Finance analysis step")
        symbol: str = Field(default="AAPL", description="Stock symbol to analyze")
        
        async def execute(self, agent):
            # Call MCP tools to get financial data
            price_action = {
                "tool_name": "get_current_stock_price",
                "parameters": {"symbol": self.symbol}
            }
            price_result = agent.environments["finance_tools"].step(price_action)
            
            fundamentals_action = {
                "tool_name": "get_stock_fundamentals",
                "parameters": {"symbol": self.symbol}
            }
            fundamentals_result = agent.environments["finance_tools"].step(fundamentals_action)
            
            news_action = {
                "tool_name": "get_company_news",
                "parameters": {"symbol": self.symbol}
            }
            news_result = agent.environments["finance_tools"].step(news_action)
            
            # Generate analysis based on the data
            prompt = f"""
            You are {agent.role}. {agent.persona}
            
            Analyze the following financial data for {self.symbol}:
            
            Current Price: ${price_result.get('result', 'N/A')}
            
            Fundamentals:
            {self._format_dict(fundamentals_result.get('result', {}))}
            
            Recent News:
            {self._format_news(news_result.get('result', []))}
            
            Based on this data, provide a comprehensive analysis of {self.symbol} including:
            1. Current financial health
            2. Market position and competitive advantages
            3. Recent developments and their impact
            4. Investment recommendation (Buy, Hold, or Sell)
            5. Price target and rationale
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
        
        def _format_news(self, news):
            if not news:
                return "No recent news available"
            
            formatted = ""
            for i, article in enumerate(news[:5]):  # Show up to 5 recent news items
                formatted += f"- {article.get('title', 'No title')}\n"
            return formatted
    
    # Run the finance analysis step
    analysis = await agent.run_step(
        step=FinanceAnalysisStep(symbol="AAPL")
    )
    
    return analysis

# Run the example
analysis = asyncio.run(use_finance_mcp_server())
print(analysis)
```

In the next section, we'll explore how to configure MCP servers and integrate them with agents.
