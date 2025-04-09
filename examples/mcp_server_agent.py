import asyncio
import json
from typing import Dict, Any, Optional, List
from uuid import UUID
from datetime import datetime

from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.config import AgentStorageConfig
from market_agents.agents.personas.persona import Persona
from market_agents.environments.mechanisms.mcp_server import MCPServerEnvironmentConfig, MCPServerActionSpace
from market_agents.workflows.workflow_utils import setup_mcp_environment
from minference.lite.models import LLMConfig, ResponseFormat

async def main():
    """
    Example of a MarketAgent using MCP server tools for investment research.
    """
    # Initialize storage
    storage_config = AgentStorageConfig(
        model="text-embedding-3-small",
        embedding_provider="openai",
        vector_dim=256
    )
    storage_utils = AgentStorageAPIUtils(config=storage_config)
    
    # Configure and set up the MCP finance server environment
    print("Setting up MCP finance server environment...")
    config = MCPServerEnvironmentConfig(
        name="mcp_finance",
        mechanism="mcp_server",
        mcp_server_module="market_agents.orchestrators.mcp_server.finance_mcp_server"
    )
    
    # Set up MCP environment
    finance_mcp = await setup_mcp_environment(config)
    tools = finance_mcp.get_tools()
    
    selected_tool_names = [
        "get_stock_fundamentals", 
        "get_financial_statements",
        "get_technical_indicators",
        "get_analyst_recommendations"
    ]
    
    # Extract the specific tools we want
    selected_mcp_tools = [tools[name] for name in selected_tool_names if name in tools]

    finance_mcp.action_space = MCPServerActionSpace(
        mechanism=finance_mcp.mechanism,
        selected_tools=selected_mcp_tools,
        workflow=True
    )
    
    # Create investment analyst persona
    investment_analyst_persona = Persona(
        role="Investment Analyst",
        persona="You are an investment analyst specializing in security analysis and market research. I conduct in-depth research on investment opportunities to identify attractive investments.",
        objectives=[
            "Research and analyze investment opportunities",
            "Evaluate securities based on fundamental and technical factors",
            "Monitor market trends and economic indicators",
            "Generate investment ideas and recommendations"
        ]
    )

    # Create agent with MCP server environment
    print("Creating investment analyst agent...")
    agent = await MarketAgent.create(
        persona=investment_analyst_persona,
        llm_config=LLMConfig(
            model="gpt-4o-mini",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.workflow
        ),
        environments={"finance": finance_mcp},
        tools=selected_mcp_tools,
        storage_utils=storage_utils,
    )
    
    print(f"Created agent with ID: {agent.id}")
    print(f"Role: {agent.role}")
    
    # Run analysis for a specific ticker
    ticker = "NVDA"
    agent.task = f"""
    Conduct a comprehensive investment analysis for {ticker} covering:
    
    1. Fundamental analysis - Review key financial metrics and valuation
    2. Technical analysis - Examine price trends and technical indicators
    3. Analyst sentiment - Consider expert recommendations and targets
    
    Conclude with an investment recommendation (Buy, Hold, or Sell).
    """
    
    print(f"\nAnalyzing ticker: {ticker}")
    print("-" * 80)
    
    # Run a step or an entire episode
    results = await agent.run_step()
    #results = await agent.run_episode()
    
    # Display results
    print(f"\nInvestment Analysis Results for {ticker}:")
    print("-" * 80)
    
    # Print results
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    asyncio.run(main())