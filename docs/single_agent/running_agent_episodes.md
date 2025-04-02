# Running Agent Episodes

## Overview

In the MarketAgents framework, agent behavior is organized around the concept of cognitive episodes and steps. This structure allows agents to process information, make decisions, and learn from experiences in a systematic way.

## Cognitive Steps

A cognitive step represents a single unit of agent processing. The framework provides several built-in cognitive steps:

- **PerceptionStep**: Processes observations from the environment
- **ActionStep**: Determines and executes actions based on perceptions
- **ReflectionStep**: Analyzes the results of actions and updates internal state
- **PlanningStep**: Creates plans for achieving objectives
- **EvaluationStep**: Assesses the outcomes of actions against objectives

### Running a Single Cognitive Step

You can run a single cognitive step using the `run_step()` method:

```python
import asyncio
from market_agents.agents.cognitive_steps import ActionStep

async def run_single_step(agent):
    # Run an action step
    result = await agent.run_step(
        step=ActionStep,
        environment_name="stock_market",  # Optional: specify environment
        additional_context={"market_data": market_data}  # Optional: provide context
    )
    
    return result

# Execute the function
result = asyncio.run(run_single_step(agent))
```

### Custom Cognitive Steps

You can create custom cognitive steps by extending the base classes:

```python
from market_agents.agents.cognitive_steps import CognitiveStep
from pydantic import Field

class MarketAnalysisStep(CognitiveStep):
    step_name: str = Field(default="market_analysis", description="Market analysis step")
    market_data: dict = Field(default_factory=dict, description="Market data to analyze")
    
    async def execute(self, agent):
        # Step-specific prompt construction
        prompt = self.construct_prompt(agent)
        
        # Execute LLM call
        response = await agent.llm_orchestrator.generate(
            model=agent.llm_config.model,
            messages=[{"role": "system", "content": prompt}],
            temperature=agent.llm_config.temperature
        )
        
        # Process and return results
        return response.content

# Using the custom step
result = await agent.run_step(
    step=MarketAnalysisStep(market_data=current_market_data)
)
```

## Cognitive Episodes

A cognitive episode is a sequence of cognitive steps that form a complete processing cycle. The default episode consists of Perception → Action → Reflection.

### Running a Complete Episode

```python
import asyncio
from market_agents.agents.cognitive_steps import CognitiveEpisode, PerceptionStep, ActionStep, ReflectionStep

async def run_episode(agent):
    # Define an episode with specific steps
    episode = CognitiveEpisode(
        steps=[PerceptionStep, ActionStep, ReflectionStep],
        environment_name="stock_market"
    )
    
    # Run the episode
    results = await agent.run_episode(episode)
    
    return results

# Execute the function
results = asyncio.run(run_episode(agent))
```

### Default Episode

If you don't specify an episode, the agent will use the default Perception → Action → Reflection sequence:

```python
# Run with default episode
results = await agent.run_episode(environment_name="stock_market")
```

### Custom Episodes

You can create custom episodes by defining your own sequence of steps:

```python
from market_agents.agents.cognitive_steps import CognitiveEpisode, PerceptionStep, PlanningStep, ActionStep, EvaluationStep

# Create a custom episode for strategic decision making
strategic_episode = CognitiveEpisode(
    steps=[PerceptionStep, PlanningStep, ActionStep, EvaluationStep],
    environment_name="investment_environment"
)

# Run the custom episode
results = await agent.run_episode(strategic_episode)
```

## Environment Integration

Episodes and steps are typically executed within the context of an environment. The environment provides the agent with observations and processes its actions.

```python
from market_agents.environments.mechanisms.stock_market import StockMarketEnvironment

# Create a stock market environment
stock_market = StockMarketEnvironment(
    name="nyse",
    symbols=["AAPL", "MSFT", "GOOGL"],
    data_source="yahoo_finance"
)

# Add the environment to the agent
agent.environments["stock_market"] = stock_market

# Run an episode in this environment
results = await agent.run_episode(environment_name="stock_market")
```

## Practical Example: Market Analysis Agent

Here's a complete example of creating and running an agent that analyzes market data:

```python
import asyncio
from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.agents.personas.persona import Persona
from market_agents.environments.mechanisms.stock_market import StockMarketEnvironment
from market_agents.agents.cognitive_steps import PerceptionStep, ActionStep, ReflectionStep
from minference.lite.models import LLMConfig

async def run_market_analysis():
    # Create storage utilities
    storage_utils = AgentStorageAPIUtils(config=storage_config)
    
    # Create persona
    persona = Persona(
        role="Market Analyst",
        persona="I am a market analyst specializing in technical analysis and trend identification.",
        objectives=["Identify market trends", "Recommend trading strategies"]
    )
    
    # Create the agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="market_analyst_1",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4",
            client="openai",
            temperature=0.5,
            max_tokens=1000
        ),
        persona=persona
    )
    
    # Create and add stock market environment
    stock_market = StockMarketEnvironment(
        name="us_market",
        symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
        data_source="yahoo_finance"
    )
    agent.environments["stock_market"] = stock_market
    
    # Run a complete episode
    results = await agent.run_episode(environment_name="stock_market")
    
    # Process and return results
    analysis = results[1]  # Action step result contains the analysis
    return analysis

# Execute the function
analysis = asyncio.run(run_market_analysis())
print(analysis)
```

In the next section, we'll explore the agent's memory systems in detail.
