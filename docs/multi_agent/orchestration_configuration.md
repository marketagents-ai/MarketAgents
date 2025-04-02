# GroupChat Orchestration Configuration

## Overview

The MarketAgents framework provides a powerful orchestration system for managing multi-agent group chat interactions. The orchestrator coordinates agent communication, manages conversation flow, and ensures proper execution of the group discussion.

This document explains how to configure and use the GroupChat orchestrator to facilitate complex multi-agent discussions.

## Orchestrator Components

The GroupChat orchestration system consists of several key components:

1. **GroupChat API Service**: Backend service for managing chat sessions
2. **MarketAgentTeam**: Manages a collection of agents as a team
3. **Orchestrator**: Coordinates agent interactions and environment steps
4. **MetaOrchestrator**: Manages multiple orchestrators across different environments

## GroupChat API Service

The GroupChat API service provides a persistent backend for managing chat sessions:

```python
# The GroupChat API service should be running on http://localhost:8002
# You can start it using:
# python market_agents/orchestrators/group_chat/groupchat_api.py
```

## Configuring the Orchestrator

To configure a GroupChat orchestrator, you'll need to:

1. Create a configuration object
2. Initialize the orchestrator with agents and environments
3. Set up the orchestration parameters

### Orchestrator Configuration

```python
from market_agents.orchestrators.config import OrchestratorConfig
from market_agents.orchestrators.orchestrator import Orchestrator
from market_agents.orchestrators.market_agent_team import MarketAgentTeam

# Create orchestrator configuration
config = OrchestratorConfig(
    environment_order=["group_chat"],  # List of environments to run in sequence
    num_agents=3,                      # Number of agents to create
    max_steps=5,                       # Maximum steps per environment
    tool_mode=False,                   # Whether to use tool mode for agents
    agent_config={
        "knowledge_base": "finance_kb"  # Optional knowledge base name
    },
    llm_configs=[                       # LLM configurations for agents
        {
            "model": "gpt-4",
            "client": "openai",
            "temperature": 0.7,
            "max_tokens": 1000,
            "use_cache": True
        }
    ]
)
```

### Creating the Orchestrator

```python
# Create a team of agents
team = MarketAgentTeam(agents=agents)

# Create the orchestrator
orchestrator = Orchestrator(
    config=config,
    team=team,
    environment=group_chat_env
)
```

## MetaOrchestrator for Multiple Environments

For more complex scenarios involving multiple environments, you can use the MetaOrchestrator:

```python
from market_agents.orchestrators.meta_orchestrator import MetaOrchestrator

# Create environments dictionary
environments = {
    "group_chat": group_chat_env,
    "stock_market": stock_market_env
}

# Create the meta orchestrator
meta_orchestrator = MetaOrchestrator(
    config=config,
    agents=agents,
    environments=environments
)
```

## Configuration from YAML

You can load orchestrator configuration from a YAML file:

```python
from market_agents.environments.config import load_config_from_yaml

# Load configuration from YAML file
config = load_config_from_yaml("market_agents/orchestrators/orchestrator_config.yaml")
```

Example `orchestrator_config.yaml`:

```yaml
environment_order:
  - group_chat
  - stock_market

num_agents: 3
max_steps: 5
tool_mode: false

agent_config:
  knowledge_base: finance_kb

llm_configs:
  - model: gpt-4
    client: openai
    temperature: 0.7
    max_tokens: 1000
    use_cache: true
```

## Cohort Management

For larger agent groups, you can organize agents into cohorts (smaller discussion groups):

```python
# Create GroupChat with cohort support
group_chat_mechanism = GroupChat(
    max_rounds=3,
    sequential=True,
    form_cohorts=True,     # Enable cohort formation
    group_size=3           # Size of each cohort
)

# Create GroupChat environment
group_chat_env = GroupChatEnvironment(
    name="investment_discussion",
    mechanism=group_chat_mechanism
)
```

## Complete Example: GroupChat Orchestration Configuration

Here's a complete example of configuring a GroupChat orchestration:

```python
import asyncio
from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.agents.personas.persona import Persona
from market_agents.environments.mechanisms.group_chat import GroupChat, GroupChatEnvironment
from market_agents.orchestrators.config import OrchestratorConfig
from market_agents.orchestrators.orchestrator import Orchestrator
from market_agents.orchestrators.market_agent_team import MarketAgentTeam
from minference.lite.models import LLMConfig

async def configure_group_chat_orchestration():
    # Create storage utilities
    storage_utils = AgentStorageAPIUtils(config=storage_config)
    
    # Create personas for different agents
    personas = [
        Persona(
            role="Financial Analyst",
            persona="I am a financial analyst specializing in tech stocks.",
            objectives=["Analyze tech company financials", "Identify investment opportunities"]
        ),
        Persona(
            role="Market Strategist",
            persona="I am a market strategist focusing on macroeconomic trends.",
            objectives=["Identify market trends", "Develop investment strategies"]
        ),
        Persona(
            role="Risk Manager",
            persona="I am a risk manager specializing in portfolio risk assessment.",
            objectives=["Evaluate investment risks", "Recommend risk mitigation strategies"]
        )
    ]
    
    # Create agents
    agents = []
    for i, persona in enumerate(personas):
        agent = await MarketAgent.create(
            storage_utils=storage_utils,
            agent_id=f"agent_{i}",
            use_llm=True,
            llm_config=LLMConfig(
                model="gpt-4",
                client="openai",
                temperature=0.7
            ),
            persona=persona
        )
        agents.append(agent)
    
    # Create GroupChat mechanism
    group_chat_mechanism = GroupChat(
        max_rounds=3,
        sequential=True,
        initial_topic="Investment opportunities in the tech sector for 2025",
        api_url="http://localhost:8002"  # GroupChat API endpoint
    )
    
    # Create GroupChat environment
    group_chat_env = GroupChatEnvironment(
        name="investment_discussion",
        address="investment_discussion_room",
        max_steps=3,
        mechanism=group_chat_mechanism
    )
    
    # Add environment to each agent
    for agent in agents:
        agent.environments["group_chat"] = group_chat_env
    
    # Create orchestrator configuration
    config = OrchestratorConfig(
        environment_order=["group_chat"],
        num_agents=len(agents),
        max_steps=3,
        tool_mode=False
    )
    
    # Create team and orchestrator
    team = MarketAgentTeam(agents=agents)
    orchestrator = Orchestrator(
        config=config,
        team=team,
        environment=group_chat_env
    )
    
    return orchestrator, agents, group_chat_env

# Run the configuration
orchestrator, agents, group_chat_env = asyncio.run(configure_group_chat_orchestration())
```

In the next section, we'll explore how to run group discussions using the configured orchestrator.
