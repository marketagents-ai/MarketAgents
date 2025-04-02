# Research Environment Setup

## Overview

The Research environment in MarketAgents provides a structured mechanism for agents to collaboratively conduct research tasks. It enables agents to gather information, analyze data, and synthesize findings into comprehensive research outputs.

## Research Mechanism

The core of multi-agent research is the `Research` mechanism, which manages the research workflow, tracks progress, and coordinates agent contributions.

### Key Features

- **Task Distribution**: Assign specific research tasks to specialized agents
- **Information Gathering**: Collect data from various sources
- **Collaborative Analysis**: Combine insights from multiple agents
- **Knowledge Integration**: Incorporate findings into shared knowledge bases
- **Research Synthesis**: Compile individual contributions into cohesive outputs

## Creating a Research Environment

To set up a Research environment, you'll need to:

1. Create a Research mechanism
2. Initialize a ResearchEnvironment with this mechanism
3. Add agents to the environment

```python
from market_agents.environments.mechanisms.research import Research, ResearchEnvironment

# Create the Research mechanism
research_mechanism = Research(
    topic="Impact of AI on financial markets",  # Main research topic
    max_rounds=3,                              # Maximum research rounds
    subtopics=[                                # Optional research subtopics
        "AI adoption in trading",
        "Algorithmic trading impact",
        "Regulatory considerations"
    ]
)

# Create the Research environment
research_env = ResearchEnvironment(
    name="ai_finance_research",
    address="ai_finance_research_env",
    max_steps=3,                              # Maximum environment steps
    mechanism=research_mechanism
)
```

### Configuration Options

The Research mechanism supports several configuration options:

```python
research_mechanism = Research(
    topic="Impact of AI on financial markets",  # Main research topic
    max_rounds=3,                              # Maximum research rounds
    subtopics=[                                # Research subtopics
        "AI adoption in trading",
        "Algorithmic trading impact",
        "Regulatory considerations"
    ],
    assign_subtopics=True,                     # Whether to assign subtopics to specific agents
    collaborative_synthesis=True,              # Whether to collaboratively synthesize findings
    knowledge_base_name="ai_finance_kb"        # Optional knowledge base for storing findings
)
```

## Adding Agents to the Environment

Once you've created the Research environment, you need to add it to your agents:

```python
# Add the environment to each agent
for agent in agents:
    agent.environments["research"] = research_env
```

## Action and Observation Spaces

The Research environment defines specific action and observation spaces:

### ResearchActionSpace

```python
from market_agents.environments.mechanisms.research import ResearchActionSpace

action_space = ResearchActionSpace()

# Example valid action
action = {
    "action_type": "research_finding",
    "content": "AI-driven trading algorithms now account for over 70% of daily trading volume in major exchanges.",
    "subtopic": "AI adoption in trading",
    "sources": ["Financial Times report", "Market analysis data"],
    "confidence": 0.85
}
```

### ResearchObservationSpace

```python
from market_agents.environments.mechanisms.research import ResearchObservationSpace

observation_space = ResearchObservationSpace()

# Example observation structure
observation = {
    "topic": "Impact of AI on financial markets",
    "subtopics": [
        "AI adoption in trading",
        "Algorithmic trading impact",
        "Regulatory considerations"
    ],
    "current_round": 1,
    "max_rounds": 3,
    "findings": [
        {
            "agent_id": "agent_1",
            "content": "AI-driven trading algorithms now account for over 70% of daily trading volume in major exchanges.",
            "subtopic": "AI adoption in trading",
            "sources": ["Financial Times report", "Market analysis data"],
            "confidence": 0.85,
            "timestamp": "2025-04-02T10:30:00Z"
        }
    ]
}
```

## Complete Example: Setting Up a Research Environment

Here's a complete example of setting up a Research environment with multiple agents:

```python
import asyncio
from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.agents.personas.persona import Persona
from market_agents.environments.mechanisms.research import Research, ResearchEnvironment
from minference.lite.models import LLMConfig

async def setup_research_environment():
    # Create storage utilities
    storage_utils = AgentStorageAPIUtils(config=storage_config)
    
    # Create personas for different research agents
    personas = [
        Persona(
            role="Financial Technology Analyst",
            persona="I am a financial technology analyst specializing in AI applications in finance.",
            objectives=["Research AI adoption in trading", "Analyze impact of AI on market efficiency"]
        ),
        Persona(
            role="Market Structure Researcher",
            persona="I am a market structure researcher focusing on algorithmic trading impacts.",
            objectives=["Study algorithmic trading effects", "Analyze market microstructure changes"]
        ),
        Persona(
            role="Regulatory Expert",
            persona="I am a regulatory expert specializing in financial technology regulation.",
            objectives=["Research regulatory frameworks", "Identify regulatory challenges"]
        )
    ]
    
    # Create agents
    agents = []
    for i, persona in enumerate(personas):
        agent = await MarketAgent.create(
            storage_utils=storage_utils,
            agent_id=f"researcher_{i}",
            use_llm=True,
            llm_config=LLMConfig(
                model="gpt-4",
                client="openai",
                temperature=0.7
            ),
            persona=persona
        )
        agents.append(agent)
    
    # Create Research mechanism
    research_mechanism = Research(
        topic="Impact of AI on financial markets",
        max_rounds=3,
        subtopics=[
            "AI adoption in trading",
            "Algorithmic trading impact",
            "Regulatory considerations"
        ],
        assign_subtopics=True,
        collaborative_synthesis=True
    )
    
    # Create Research environment
    research_env = ResearchEnvironment(
        name="ai_finance_research",
        address="ai_finance_research_env",
        max_steps=3,
        mechanism=research_mechanism
    )
    
    # Add environment to each agent
    for agent in agents:
        agent.environments["research"] = research_env
    
    return agents, research_env

# Run the setup
agents, research_env = asyncio.run(setup_research_environment())
```

In the next section, we'll explore the research workflow and how agents collaborate on research tasks.
