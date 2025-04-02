# Research Orchestration

## Overview

Research orchestration in the MarketAgents framework coordinates the activities of multiple agents working together on research tasks. The orchestrator manages the research workflow, distributes tasks, and ensures that agents collaborate effectively to produce comprehensive research outputs.

## Research Orchestrator Components

The research orchestration system consists of several key components:

1. **Research Environment**: Provides the context and mechanism for research
2. **MarketAgentTeam**: Manages a collection of agents as a research team
3. **Orchestrator**: Coordinates agent interactions and research steps
4. **MetaOrchestrator**: Manages multiple orchestrators across different environments

## Configuring Research Orchestration

To configure research orchestration, you'll need to:

1. Create a configuration object
2. Initialize the orchestrator with agents and the research environment
3. Set up the orchestration parameters

### Orchestrator Configuration

```python
from market_agents.orchestrators.config import OrchestratorConfig
from market_agents.orchestrators.orchestrator import Orchestrator
from market_agents.orchestrators.market_agent_team import MarketAgentTeam

# Create orchestrator configuration
config = OrchestratorConfig(
    environment_order=["research"],  # List of environments to run in sequence
    num_agents=3,                    # Number of agents to create
    max_steps=3,                     # Maximum steps per environment
    tool_mode=False,                 # Whether to use tool mode for agents
    agent_config={
        "knowledge_base": "research_kb"  # Optional knowledge base name
    },
    llm_configs=[                     # LLM configurations for agents
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

### Creating the Research Orchestrator

```python
# Create a team of agents
team = MarketAgentTeam(agents=agents)

# Create the orchestrator
orchestrator = Orchestrator(
    config=config,
    team=team,
    environment=research_env
)
```

## Running Research Orchestration

To run the research orchestration:

```python
import asyncio

async def run_research_orchestration(orchestrator):
    # Run the orchestration
    results = await orchestrator.run_orchestration()
    return results

# Execute the orchestration
results = asyncio.run(run_research_orchestration(orchestrator))
```

## Orchestration Process

When you run the research orchestration, the following steps occur:

1. The orchestrator initializes the research environment
2. Agents are assigned to specific research subtopics
3. The orchestrator runs the specified number of research rounds
4. For each round, each agent conducts research on their assigned subtopic
5. The orchestrator collects and processes the research findings
6. A final research output is synthesized from all findings

## Research Task Distribution

The orchestrator can distribute research tasks among agents based on their expertise:

```python
# Automatic task distribution based on agent expertise
research_mechanism = Research(
    topic="Impact of AI on financial markets",
    subtopics=[
        "AI adoption in trading",
        "Algorithmic trading impact",
        "Regulatory considerations"
    ],
    assign_subtopics=True  # Automatically assign subtopics to agents
)

# The orchestrator will match agents to subtopics based on their personas
```

## Parallel Research Execution

The orchestrator can execute research tasks in parallel:

```python
from market_agents.orchestrators.parallel_cognitive_steps import ParallelCognitiveProcessor

# Create parallel cognitive processor
parallel_processor = ParallelCognitiveProcessor(
    ai_utils=inference_orchestrator,
    storage_service=storage_service,
    tool_mode=False
)

# Configure research environment with parallel processing
research_env = ResearchEnvironment(
    name="parallel_research",
    mechanism=research_mechanism,
    ai_utils=inference_orchestrator,
    storage_service=storage_service,
    tool_mode=False
)
```

## Research Output Collection

The orchestrator collects and processes research outputs:

```python
# Get research output after orchestration
research_output = research_env.mechanism.get_research_output()

# Format and save the output
formatted_output = {
    "topic": research_output.get("topic"),
    "subtopics": research_output.get("subtopics"),
    "findings": research_output.get("findings"),
    "synthesis": research_output.get("synthesis"),
    "contributors": [agent.id for agent in agents]
}

# Save to file
import json
with open("research_output.json", "w") as f:
    json.dump(formatted_output, f, indent=2)
```

## MetaOrchestrator for Complex Research

For more complex research involving multiple environments, you can use the MetaOrchestrator:

```python
from market_agents.orchestrators.meta_orchestrator import MetaOrchestrator

# Create environments dictionary
environments = {
    "research": research_env,
    "group_chat": group_chat_env  # For collaborative discussion of research
}

# Create the meta orchestrator
meta_orchestrator = MetaOrchestrator(
    config=config,
    agents=agents,
    environments=environments
)

# Run the meta orchestration
await meta_orchestrator.run_orchestration()
```

## Complete Example: Research Orchestration

Here's a complete example of configuring and running research orchestration:

```python
import asyncio
import json
from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.agents.personas.persona import Persona
from market_agents.environments.mechanisms.research import Research, ResearchEnvironment
from market_agents.orchestrators.config import OrchestratorConfig
from market_agents.orchestrators.orchestrator import Orchestrator
from market_agents.orchestrators.market_agent_team import MarketAgentTeam
from minference.lite.models import LLMConfig

async def run_orchestrated_research():
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
    
    # Create orchestrator configuration
    config = OrchestratorConfig(
        environment_order=["research"],
        num_agents=len(agents),
        max_steps=3,
        tool_mode=False
    )
    
    # Create team and orchestrator
    team = MarketAgentTeam(agents=agents)
    orchestrator = Orchestrator(
        config=config,
        team=team,
        environment=research_env
    )
    
    # Run the orchestration
    results = await orchestrator.run_orchestration()
    
    # Get final research output
    research_output = research_env.mechanism.get_research_output()
    
    # Format and save the output
    formatted_output = {
        "topic": research_output.get("topic"),
        "subtopics": research_output.get("subtopics"),
        "findings": [
            {
                "content": finding.get("content"),
                "subtopic": finding.get("subtopic"),
                "agent_id": finding.get("agent_id"),
                "agent_role": next((a.role for a in agents if a.id == finding.get("agent_id")), "unknown"),
                "sources": finding.get("sources", []),
                "confidence": finding.get("confidence", 0.0)
            }
            for finding in research_output.get("findings", [])
        ],
        "synthesis": research_output.get("synthesis", ""),
        "contributors": [
            {
                "agent_id": agent.id,
                "role": agent.role,
                "assigned_subtopic": research_env.mechanism.get_agent_subtopic(agent.id)
            }
            for agent in agents
        ]
    }
    
    # Save to file
    with open("ai_finance_research_output.json", "w") as f:
        json.dump(formatted_output, f, indent=2)
    
    return formatted_output

# Run the orchestrated research
research_results = asyncio.run(run_orchestrated_research())
```

## Practical Applications

Research orchestration can be used for various applications:

1. **Market Analysis**: Agents research different aspects of market trends and opportunities
2. **Competitive Intelligence**: Agents gather and analyze information about competitors
3. **Technology Assessment**: Agents evaluate emerging technologies and their potential impacts
4. **Policy Research**: Agents analyze policy implications and regulatory considerations
5. **Investment Research**: Agents conduct comprehensive research on investment opportunities

This concludes our documentation on multi-agent research capabilities in the MarketAgents framework. In the next section, we'll explore MCP server integration for enhanced agent capabilities.
