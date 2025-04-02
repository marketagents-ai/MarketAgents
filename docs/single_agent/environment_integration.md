# Environment Integration

## Overview

A key feature of the MarketAgents framework is the ability to integrate agents with various environment mechanisms. Environments provide the context in which agents operate, defining the available actions, observations, and interaction rules.

This document explains how to integrate a single agent with different environment types and how to configure environment-specific behavior.

## Environment Basics

In the MarketAgents framework, environments are instances of the `MultiAgentEnvironment` class or its derivatives. Each environment has:

- An action space defining valid agent actions
- An observation space defining the structure of environment observations
- A mechanism that implements the environment's specific logic

## Adding Environments to an Agent

You can add environments to an agent either during creation or after initialization:

```python
# During agent creation
agent = await MarketAgent.create(
    # ... other parameters
    environments={
        "stock_market": stock_market_env,
        "auction": auction_env
    }
)

# After initialization
agent.environments["group_chat"] = group_chat_env
```

## Common Environment Types

The MarketAgents framework includes several built-in environment mechanisms:

1. **Stock Market**: For financial market simulations
2. **Auction**: For auction-based interactions
3. **Group Chat**: For multi-agent communication
4. **Information Board**: For shared information access
5. **Research**: For collaborative research tasks
6. **Beauty Contest**: For economic game theory simulations

### Stock Market Environment

```python
from market_agents.environments.mechanisms.stock_market import StockMarketEnvironment

# Create a stock market environment
stock_market = StockMarketEnvironment(
    name="us_market",
    symbols=["AAPL", "MSFT", "GOOGL", "AMZN"],
    data_source="yahoo_finance",
    initial_cash=10000.0,
    commission_rate=0.001
)

# Add to agent
agent.environments["stock_market"] = stock_market

# Run an episode in this environment
results = await agent.run_episode(environment_name="stock_market")
```

### Auction Environment

```python
from market_agents.environments.mechanisms.auction import AuctionEnvironment

# Create an auction environment
auction = AuctionEnvironment(
    name="item_auction",
    auction_type="english",  # Options: english, dutch, sealed_bid
    items=[
        {"id": "item_1", "name": "Rare Collectible", "starting_price": 100.0},
        {"id": "item_2", "name": "Vintage Watch", "starting_price": 500.0}
    ],
    max_rounds=10
)

# Add to agent
agent.environments["auction"] = auction

# Run an episode in this environment
results = await agent.run_episode(environment_name="auction")
```

### Information Board Environment

```python
from market_agents.environments.mechanisms.information_board import InformationBoardEnvironment

# Create an information board environment
info_board = InformationBoardEnvironment(
    name="market_news",
    categories=["economic_indicators", "company_news", "market_trends"],
    max_posts_per_category=20
)

# Add to agent
agent.environments["info_board"] = info_board

# Run an episode in this environment
results = await agent.run_episode(environment_name="info_board")
```

## Environment Actions

Each environment defines its own action space, which determines what actions agents can take:

```python
# Stock market action example
stock_action = {
    "action_type": "trade",
    "symbol": "AAPL",
    "order_type": "market",
    "side": "buy",
    "quantity": 10
}

# Execute action in environment
observation = stock_market.step(stock_action)

# Information board action example
post_action = {
    "action_type": "post",
    "category": "company_news",
    "content": "Apple announces new product line with revolutionary features.",
    "metadata": {
        "source": "company_press_release",
        "timestamp": "2025-04-02T14:30:00Z"
    }
}

# Execute action in environment
observation = info_board.step(post_action)
```

## Environment State

You can access the current state of an environment:

```python
# Get global state (all information)
global_state = stock_market.get_global_state()

# Get agent-specific state (filtered information)
agent_state = stock_market.get_global_state(agent_id="agent_1")
```

## Practical Example: Agent in Multiple Environments

Here's a complete example of an agent operating in multiple environments:

```python
import asyncio
from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.agents.personas.persona import Persona
from market_agents.environments.mechanisms.stock_market import StockMarketEnvironment
from market_agents.environments.mechanisms.information_board import InformationBoardEnvironment
from minference.lite.models import LLMConfig

async def multi_environment_agent():
    # Create storage utilities
    storage_utils = AgentStorageAPIUtils(config=storage_config)
    
    # Create persona
    persona = Persona(
        role="Investment Analyst",
        persona="I am an investment analyst who researches market trends and makes investment decisions.",
        objectives=["Research market trends", "Make profitable investment decisions"]
    )
    
    # Create environments
    stock_market = StockMarketEnvironment(
        name="us_market",
        symbols=["AAPL", "MSFT", "GOOGL", "AMZN"],
        data_source="yahoo_finance"
    )
    
    info_board = InformationBoardEnvironment(
        name="market_news",
        categories=["economic_indicators", "company_news", "market_trends"]
    )
    
    # Create the agent with multiple environments
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="investment_analyst_1",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4",
            client="openai",
            temperature=0.7
        ),
        persona=persona,
        environments={
            "stock_market": stock_market,
            "info_board": info_board
        }
    )
    
    # First, gather information from the information board
    info_results = await agent.run_episode(environment_name="info_board")
    
    # Then, make investment decisions in the stock market
    market_results = await agent.run_episode(environment_name="stock_market")
    
    # Process and return combined results
    return {
        "information_gathering": info_results,
        "investment_decisions": market_results
    }

# Run the example
results = asyncio.run(multi_environment_agent())
```

This concludes our documentation on single agent creation and usage in the MarketAgents framework. In the next section, we'll explore multi-agent orchestration with group chat mechanisms.
