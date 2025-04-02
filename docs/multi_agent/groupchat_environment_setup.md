# GroupChat Environment Setup

## Overview

The GroupChat environment in MarketAgents provides a structured mechanism for agents to communicate with each other. It simulates a chat room where multiple agents can exchange messages, respond to each other, and collaborate on tasks.

## GroupChat Mechanism

The core of multi-agent communication is the `GroupChat` mechanism, which manages message exchange, tracks conversation history, and maintains the state of discussions.

### Key Features

- **Sequential or Parallel Discussions**: Support for both turn-based and simultaneous agent interactions
- **Cohort Formation**: Ability to organize agents into smaller discussion groups
- **Topic Management**: Structured conversations around specific topics
- **Round-Based Interactions**: Organized conversation flow with defined rounds
- **API Integration**: Optional integration with the GroupChat API service

## Creating a GroupChat Environment

To set up a GroupChat environment, you'll need to:

1. Create a GroupChat mechanism
2. Initialize a GroupChatEnvironment with this mechanism
3. Add agents to the environment

```python
from market_agents.environments.mechanisms.group_chat import GroupChat, GroupChatEnvironment

# Create the GroupChat mechanism
group_chat_mechanism = GroupChat(
    max_rounds=5,                # Maximum number of discussion rounds
    sequential=True,             # Whether agents take turns (True) or speak simultaneously (False)
    form_cohorts=False,          # Whether to organize agents into smaller groups
    initial_topic="Market trends for technology stocks in 2025"  # Initial discussion topic
)

# Create the GroupChat environment
group_chat_env = GroupChatEnvironment(
    name="tech_discussion",
    address="tech_discussion_room",
    max_steps=5,                 # Maximum number of environment steps
    mechanism=group_chat_mechanism
)
```

### Configuration Options

The GroupChat mechanism supports several configuration options:

```python
group_chat_mechanism = GroupChat(
    max_rounds=5,                # Maximum conversation rounds
    sequential=True,             # Turn-based (True) or simultaneous (False) discussion
    form_cohorts=True,           # Whether to organize agents into cohorts
    group_size=3,                # Size of each cohort (if form_cohorts is True)
    api_url="http://localhost:8002",  # GroupChat API endpoint (optional)
    initial_topic="Discuss the impact of rising interest rates on tech stocks"  # Starting topic
)
```

## Adding Agents to the Environment

Once you've created the GroupChat environment, you need to add it to your agents:

```python
# Add the environment to each agent
for agent in agents:
    agent.environments["group_chat"] = group_chat_env
```

## GroupChat API Integration

For more advanced features and persistent storage of conversations, you can integrate with the GroupChat API service:

```python
# Create GroupChat with API integration
group_chat_mechanism = GroupChat(
    max_rounds=5,
    sequential=False,
    api_url="http://localhost:8002"  # GroupChat API endpoint
)
```

The GroupChat API service should be running before you start the orchestration. You can start it using the provided script:

```bash
python market_agents/orchestrators/group_chat/groupchat_api.py
```

Or use the convenience script:

```bash
./start_market_agents.sh
```

## Action and Observation Spaces

The GroupChat environment defines specific action and observation spaces:

### GroupChatActionSpace

```python
from market_agents.environments.mechanisms.group_chat import GroupChatActionSpace

action_space = GroupChatActionSpace()

# Example valid action
action = {
    "message": "I believe tech stocks will face challenges due to rising interest rates.",
    "recipient_id": "all",  # or specific agent ID
    "metadata": {
        "sentiment": "cautious",
        "confidence": 0.8
    }
}
```

### GroupChatObservationSpace

```python
from market_agents.environments.mechanisms.group_chat import GroupChatObservationSpace

observation_space = GroupChatObservationSpace()

# Example observation structure
observation = {
    "messages": [
        {
            "sender_id": "agent_1",
            "content": "What are your thoughts on tech stocks?",
            "timestamp": "2025-04-02T10:30:00Z"
        },
        {
            "sender_id": "agent_2",
            "content": "I'm bullish on AI-focused companies.",
            "timestamp": "2025-04-02T10:31:00Z"
        }
    ],
    "current_topic": "Tech stock outlook for 2025",
    "current_round": 1,
    "max_rounds": 5
}
```

## Complete Example: Setting Up a GroupChat Environment

Here's a complete example of setting up a GroupChat environment with multiple agents:

```python
import asyncio
from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.agents.personas.persona import Persona
from market_agents.environments.mechanisms.group_chat import GroupChat, GroupChatEnvironment
from minference.lite.models import LLMConfig

async def setup_group_chat():
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
        initial_topic="Investment opportunities in the tech sector for 2025"
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
    
    return agents, group_chat_env

# Run the setup
agents, group_chat_env = asyncio.run(setup_group_chat())
```

In the next section, we'll explore how to orchestrate group discussions using the GroupChat environment.
