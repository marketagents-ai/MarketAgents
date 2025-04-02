# Agent Configuration Options

## Overview

The MarketAgents framework provides extensive configuration options to customize agent behavior, capabilities, and interactions. This document covers the various configuration options available for a single market agent.

## LLM Configuration

Language Model configuration is a critical aspect of agent behavior, determining how the agent processes information and generates responses.

### LLMConfig Parameters

```python
from minference.lite.models import LLMConfig, ResponseFormat

llm_config = LLMConfig(
    model="gpt-4",              # Model name
    client="openai",            # Client provider (openai, anthropic, etc.)
    temperature=0.7,            # Creativity level (0.0-1.0)
    max_tokens=1000,            # Maximum response length
    use_cache=True,             # Enable response caching
    response_format=ResponseFormat.json_object  # Response format (json_object or tool)
)
```

### Available Response Formats

- `ResponseFormat.json_object`: Returns structured JSON responses
- `ResponseFormat.tool`: Enables tool-calling capabilities for agents

### Tool Mode

When using tool mode, agents can interact with predefined tools:

```python
# Enable tool mode in configuration
config.tool_mode = True

# This will automatically set response_format to ResponseFormat.tool
```

## Persona Configuration

Personas define an agent's role, personality, and objectives, significantly influencing its behavior.

```python
from market_agents.agents.personas.persona import Persona

persona = Persona(
    role="Investment Advisor",
    persona="I am an experienced investment advisor with expertise in portfolio management and risk assessment.",
    objectives=[
        "Analyze client financial situations",
        "Recommend appropriate investment strategies",
        "Monitor market trends and adjust recommendations accordingly"
    ]
)
```

### Weighted Personas

For more nuanced agent behavior, you can use weighted personas:

```python
from market_agents.agents.personas.weighted_personas.persona_weighted import WeightedPersona

weighted_persona = WeightedPersona(
    role="Financial Analyst",
    persona="I am a financial analyst with a balanced approach to investment.",
    objectives=["Provide balanced financial advice"],
    traits={
        "risk_tolerance": 0.6,  # 0.0 (risk-averse) to 1.0 (risk-seeking)
        "analytical": 0.8,      # 0.0 (intuitive) to 1.0 (analytical)
        "long_term": 0.7        # 0.0 (short-term focus) to 1.0 (long-term focus)
    }
)
```

## Economic Agent Configuration

The economic agent component handles transactions, wallet management, and economic incentives.

```python
from market_agents.economics.econ_agent import EconomicAgent

econ_agent = EconomicAgent(
    generate_wallet=True,  # Automatically generate a wallet
    initial_holdings={
        "ETH": 1.0,
        "USDC": 1000.0,
        "BTC": 0.05
    },
    risk_preference=0.5,  # 0.0 (risk-averse) to 1.0 (risk-seeking)
    discount_factor=0.9   # Time preference for future rewards
)
```

## Memory Configuration

Memory systems determine how agents store and retrieve information.

```python
from market_agents.memory.config import AgentStorageConfig

storage_config = AgentStorageConfig(
    api_url="http://localhost:8001",  # Storage API endpoint
    stm_top_k=5,                      # Number of short-term memories to retrieve
    ltm_top_k=10,                     # Number of long-term memories to retrieve
    embedding_model="text-embedding-ada-002",  # Model for embedding generation
    vector_dimension=1536,            # Dimension of embedding vectors
    similarity_threshold=0.75         # Threshold for memory similarity
)
```

## Protocol Configuration

Protocols define how agents communicate with each other and with environments.

```python
from market_agents.agents.protocols.acl_message import ACLMessage

# When creating an agent
agent = await MarketAgent.create(
    # ... other parameters
    protocol=ACLMessage,
    # ...
)
```

## Reward Function Configuration

For agents that learn from experience, you can configure custom reward functions.

```python
from market_agents.verbal_rl.rl_models import BaseRewardFunction

class CustomRewardFunction(BaseRewardFunction):
    def calculate_reward(self, state, action, next_state):
        # Custom reward calculation logic
        return reward_value

# When creating an agent
agent = await MarketAgent.create(
    # ... other parameters
    reward_function=CustomRewardFunction(),
    # ...
)
```

## Complete Configuration Example

Here's a comprehensive example combining various configuration options:

```python
import asyncio
from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.agents.personas.persona import Persona
from market_agents.economics.econ_agent import EconomicAgent
from market_agents.memory.config import AgentStorageConfig
from market_agents.agents.protocols.acl_message import ACLMessage
from minference.lite.models import LLMConfig, ResponseFormat

async def create_fully_configured_agent():
    # Storage configuration
    storage_config = AgentStorageConfig(
        api_url="http://localhost:8001",
        stm_top_k=5,
        ltm_top_k=10,
        embedding_model="text-embedding-ada-002",
        vector_dimension=1536
    )
    
    # Storage utilities
    storage_utils = AgentStorageAPIUtils(
        config=storage_config
    )
    
    # Persona
    persona = Persona(
        role="Investment Advisor",
        persona="I am an experienced investment advisor with expertise in portfolio management.",
        objectives=[
            "Analyze client financial situations",
            "Recommend appropriate investment strategies"
        ]
    )
    
    # Economic agent
    econ_agent = EconomicAgent(
        generate_wallet=True,
        initial_holdings={
            "ETH": 1.0,
            "USDC": 1000.0
        },
        risk_preference=0.6
    )
    
    # Create the market agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="investment_advisor_1",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4",
            client="openai",
            temperature=0.7,
            max_tokens=1000,
            use_cache=True,
            response_format=ResponseFormat.json_object
        ),
        protocol=ACLMessage,
        persona=persona,
        econ_agent=econ_agent
    )
    
    return agent

# Run the async function
agent = asyncio.run(create_fully_configured_agent())
```

In the next section, we'll explore how to run agent episodes and cognitive steps.
