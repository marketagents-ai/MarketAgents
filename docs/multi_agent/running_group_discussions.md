# Running Group Discussions

## Overview

Once you've set up your GroupChat environment and configured the orchestrator, you can run multi-agent group discussions. This document explains how to initiate, manage, and extract results from group discussions using the MarketAgents framework.

## Starting a Group Discussion

To start a group discussion, you run the orchestrator with your configured agents and environment:

```python
import asyncio
from market_agents.orchestrators.orchestrator import Orchestrator

async def run_group_discussion(orchestrator):
    # Run the orchestration
    results = await orchestrator.run_orchestration()
    return results

# Execute the orchestration
results = asyncio.run(run_group_discussion(orchestrator))
```

## Orchestration Process

When you run the orchestration, the following steps occur:

1. The orchestrator initializes the environment
2. Agents are added to the environment
3. The orchestrator runs the specified number of steps or rounds
4. For each round, each agent processes the current state and generates a response
5. The orchestrator collects and processes the results

## Sequential vs. Parallel Discussions

The GroupChat mechanism supports two modes of discussion:

### Sequential Mode

In sequential mode, agents take turns speaking in a predefined order:

```python
# Create GroupChat with sequential mode
group_chat_mechanism = GroupChat(
    max_rounds=3,
    sequential=True,  # Agents take turns
    initial_topic="Investment strategies for tech stocks"
)
```

### Parallel Mode

In parallel mode, all agents respond simultaneously in each round:

```python
# Create GroupChat with parallel mode
group_chat_mechanism = GroupChat(
    max_rounds=3,
    sequential=False,  # All agents respond simultaneously
    initial_topic="Investment strategies for tech stocks"
)
```

## Managing Discussion Rounds

The GroupChat mechanism organizes discussions into rounds. You can control the number of rounds and monitor the current round:

```python
# Set maximum number of rounds
group_chat_mechanism = GroupChat(max_rounds=5)

# Check current round during orchestration
current_round = group_chat_env.mechanism.current_round

# Reset the discussion to start a new one
group_chat_env.reset()
```

## Cohort Management

For larger agent groups, you can organize agents into cohorts (smaller discussion groups):

```python
# Create GroupChat with cohort support
group_chat_mechanism = GroupChat(
    max_rounds=3,
    form_cohorts=True,  # Enable cohort formation
    group_size=3        # Size of each cohort
)
```

When using cohorts, the orchestrator will:

1. Divide agents into groups of the specified size
2. Run separate discussions within each cohort
3. Manage the state for each cohort independently

### Accessing Cohort Information

```python
# Get all cohorts information
cohorts_info = group_chat_env.mechanism.get_global_state()["cohorts"]

# Get specific cohort information
cohort_id = "cohort_1"
cohort_agents = group_chat_env.mechanism.cohorts[cohort_id]
cohort_messages = group_chat_env.mechanism.round_messages[cohort_id]
```

## Topic Management

You can set and update discussion topics:

```python
# Set initial topic during creation
group_chat_mechanism = GroupChat(
    initial_topic="Investment strategies for tech stocks in 2025"
)

# Update topic during orchestration
group_chat_mechanism.initial_topic = "Impact of AI advancements on tech stock valuations"
```

## Extracting Discussion Results

After running the orchestration, you can extract and analyze the discussion results:

```python
# Get all messages from the discussion
all_messages = group_chat_env.mechanism.round_messages.get("default", [])

# Format messages for analysis
formatted_messages = []
for message in all_messages:
    formatted_messages.append({
        "round": message.get("round", 0),
        "sender": message.get("sender_id", "unknown"),
        "content": message.get("content", ""),
        "timestamp": message.get("timestamp", "")
    })

# Analyze discussion content
# ...
```

## Integration with GroupChat API

When using the GroupChat API service, you can retrieve persistent chat history:

```python
import aiohttp
import json

async def get_chat_history(chat_id):
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://localhost:8002/chats/{chat_id}/messages") as response:
            if response.status == 200:
                return await response.json()
            else:
                return []

# Get chat history
chat_id = "investment_discussion_123"
history = asyncio.run(get_chat_history(chat_id))
```

## Complete Example: Running a Group Discussion

Here's a complete example of running a group discussion:

```python
import asyncio
import json
from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.agents.personas.persona import Persona
from market_agents.environments.mechanisms.group_chat import GroupChat, GroupChatEnvironment
from market_agents.orchestrators.config import OrchestratorConfig
from market_agents.orchestrators.orchestrator import Orchestrator
from market_agents.orchestrators.market_agent_team import MarketAgentTeam
from minference.lite.models import LLMConfig

async def run_investment_discussion():
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
    
    # Run the orchestration
    results = await orchestrator.run_orchestration()
    
    # Extract discussion messages
    all_messages = group_chat_env.mechanism.round_messages.get("default", [])
    
    # Format and return results
    discussion_summary = {
        "topic": group_chat_mechanism.initial_topic,
        "rounds": group_chat_mechanism.current_round,
        "participants": [agent.id for agent in agents],
        "messages": [
            {
                "round": msg.get("round", 0),
                "sender": msg.get("sender_id", "unknown"),
                "sender_role": next((a.role for a in agents if a.id == msg.get("sender_id")), "unknown"),
                "content": msg.get("content", ""),
                "timestamp": msg.get("timestamp", "")
            }
            for msg in all_messages
        ]
    }
    
    # Save discussion summary to file
    with open("investment_discussion_results.json", "w") as f:
        json.dump(discussion_summary, f, indent=2)
    
    return discussion_summary

# Run the discussion
discussion_results = asyncio.run(run_investment_discussion())
```

## Practical Applications

Group discussions can be used for various applications:

1. **Collaborative Problem Solving**: Agents with different expertise collaborate to solve complex problems
2. **Market Analysis**: Financial agents discuss market trends and investment opportunities
3. **Scenario Planning**: Agents explore different scenarios and their implications
4. **Decision Making**: Agents debate options and reach consensus on decisions
5. **Knowledge Sharing**: Agents share and integrate information from different sources

This concludes our documentation on multi-agent GroupChat orchestration. In the next section, we'll explore multi-agent research capabilities in the MarketAgents framework.
