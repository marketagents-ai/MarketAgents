# Basic Agent Creation

## Overview

Creating a `MarketAgent` instance is the first step in building applications with the MarketAgents framework. The `MarketAgent` class is the core component that represents an intelligent agent with cognitive capabilities, economic incentives, and memory systems.

## Prerequisites

Before creating a market agent, ensure you have:

1. Installed the MarketAgents framework
2. Set up the required services (if using storage and memory features)
3. Configured your API keys in the `.env` file

## Creating a Basic Agent

The `MarketAgent` class provides a factory method `create()` that handles the initialization of all required components. This is the recommended way to create a new agent.

```python
import asyncio
from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.agents.personas.persona import Persona
from market_agents.economics.econ_agent import EconomicAgent
from minference.lite.models import LLMConfig

async def create_basic_agent():
    # Initialize storage utilities
    storage_utils = AgentStorageAPIUtils(
        config=storage_config  # Load from your config file
    )
    
    # Create a simple persona
    persona = Persona(
        role="Financial Analyst",
        persona="I am a financial analyst specializing in market research and investment strategies.",
        objectives=["Analyze market trends", "Provide investment recommendations"]
    )
    
    # Create an economic agent component
    econ_agent = EconomicAgent(
        generate_wallet=True,
        initial_holdings={
            "ETH": 1.0,
            "USDC": 1000.0
        }
    )
    
    # Create the market agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="financial_analyst_1",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4",
            client="openai",
            temperature=0.7,
            max_tokens=1000,
            use_cache=True
        ),
        persona=persona,
        econ_agent=econ_agent
    )
    
    return agent

# Run the async function
agent = asyncio.run(create_basic_agent())
```

## Required Parameters

The `MarketAgent.create()` method requires several parameters:

- `storage_utils`: An instance of `AgentStorageAPIUtils` for memory operations
- `agent_id`: A unique identifier for the agent
- `use_llm`: Boolean indicating whether to use language models
- `llm_config`: Configuration for the language model

## Optional Parameters

You can customize your agent with these optional parameters:

- `ai_utils`: Custom inference orchestrator (defaults to a new instance)
- `environments`: Dictionary of environments the agent can interact with
- `protocol`: Communication protocol class (e.g., ACLMessage)
- `persona`: Persona object defining the agent's role and objectives
- `econ_agent`: Economic agent component for handling transactions
- `knowledge_agent`: Knowledge base agent for information retrieval
- `reward_function`: Custom reward function for reinforcement learning

## What Happens During Creation

When you call `MarketAgent.create()`, the following steps occur:

1. Short-term and long-term memory systems are initialized
2. The agent is instantiated with the provided configuration
3. Components like economic agent and knowledge agent are linked
4. The agent is ready to interact with environments and execute cognitive steps

## Example: Creating an Agent with Knowledge Base

```python
import asyncio
from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.knowledge_base import MarketKnowledgeBase
from market_agents.memory.knowledge_base_agent import KnowledgeBaseAgent
from market_agents.memory.config import AgentStorageConfig

async def create_agent_with_kb():
    # Load storage configuration
    storage_config = AgentStorageConfig(
        api_url="http://localhost:8001",
        stm_top_k=5,
        ltm_top_k=10
    )
    
    # Initialize storage utilities
    storage_utils = AgentStorageAPIUtils(
        config=storage_config
    )
    
    # Create and initialize knowledge base
    market_kb = MarketKnowledgeBase(
        config=storage_config,
        table_prefix="finance_kb"
    )
    await market_kb.initialize()
    
    # Create knowledge base agent
    kb_agent = KnowledgeBaseAgent(market_kb=market_kb)
    kb_agent.id = "finance_kb_agent"
    
    # Create the market agent with knowledge base
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="financial_analyst_with_kb",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4",
            client="openai",
            temperature=0.7,
            max_tokens=1000
        ),
        knowledge_agent=kb_agent
    )
    
    return agent

# Run the async function
agent = asyncio.run(create_agent_with_kb())
```

In the next section, we'll explore how to configure agent behavior in more detail.
