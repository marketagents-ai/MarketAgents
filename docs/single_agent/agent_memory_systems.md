# Agent Memory Systems

## Overview

The MarketAgents framework implements a sophisticated memory architecture that enables agents to store, retrieve, and utilize information effectively. This memory system is divided into two main components:

1. **Short-Term Memory (STM)**: For recent interactions and immediate context
2. **Long-Term Memory (LTM)**: For persistent knowledge and experiences

This document explains how these memory systems work and how to use them effectively with your agents.

## Memory Architecture

Each MarketAgent is equipped with both short-term and long-term memory systems that are initialized during agent creation:

```python
# During agent creation, memory systems are initialized
stm = ShortTermMemory(
    agent_id=agent_id,
    agent_storage_utils=storage_utils,
    default_top_k=storage_utils.config.stm_top_k
)
await stm.initialize()

ltm = LongTermMemory(
    agent_id=agent_id,
    agent_storage_utils=storage_utils,
    default_top_k=storage_utils.config.ltm_top_k
)
await ltm.initialize()
```

## Storage Backend

The memory systems rely on a vector database backend accessed through the Agent Storage API. This service must be running for memory operations to work correctly.

### Storage Configuration

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

### Storage API Utilities

The `AgentStorageAPIUtils` class provides methods for interacting with the storage backend:

```python
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils

storage_utils = AgentStorageAPIUtils(
    config=storage_config,
    logger=logging.getLogger("storage_api")
)

# Check if the storage API is healthy
is_healthy = await storage_utils.check_api_health()
```

## Short-Term Memory

Short-term memory stores recent interactions, observations, and immediate context. It's designed for quick access to recent information.

### Adding to Short-Term Memory

```python
# Add an observation to short-term memory
await agent.short_term_memory.add_memory(
    content="The stock price of AAPL increased by 2.5% today.",
    memory_type="observation",
    metadata={
        "source": "market_data",
        "timestamp": "2025-04-02T10:30:00Z",
        "symbol": "AAPL"
    }
)

# Add an action to short-term memory
await agent.short_term_memory.add_memory(
    content="Recommended buying 10 shares of AAPL based on positive momentum.",
    memory_type="action",
    metadata={
        "action_type": "recommendation",
        "timestamp": "2025-04-02T10:35:00Z",
        "symbol": "AAPL"
    }
)
```

### Retrieving from Short-Term Memory

```python
# Retrieve recent memories
recent_memories = await agent.short_term_memory.get_recent_memories(limit=5)

# Retrieve memories by similarity to a query
similar_memories = await agent.short_term_memory.get_memories_by_similarity(
    query="Apple stock performance",
    top_k=3
)

# Retrieve memories by type
observation_memories = await agent.short_term_memory.get_memories_by_type(
    memory_type="observation",
    limit=5
)
```

## Long-Term Memory

Long-term memory stores persistent knowledge, learned patterns, and important experiences. It's designed for retaining information over extended periods.

### Adding to Long-Term Memory

```python
# Add knowledge to long-term memory
await agent.long_term_memory.add_memory(
    content="Apple Inc. typically announces new iPhone models in September each year.",
    memory_type="knowledge",
    metadata={
        "source": "historical_analysis",
        "category": "product_cycles",
        "company": "Apple"
    }
)

# Add a learned pattern to long-term memory
await agent.long_term_memory.add_memory(
    content="Tech stocks tend to experience increased volatility during earnings season.",
    memory_type="pattern",
    metadata={
        "category": "market_behavior",
        "sector": "technology"
    }
)
```

### Retrieving from Long-Term Memory

```python
# Retrieve memories by similarity to a query
relevant_knowledge = await agent.long_term_memory.get_memories_by_similarity(
    query="Apple product release cycles",
    top_k=5
)

# Retrieve memories by type
pattern_memories = await agent.long_term_memory.get_memories_by_type(
    memory_type="pattern",
    limit=10
)

# Retrieve memories with specific metadata
tech_memories = await agent.long_term_memory.get_memories_by_metadata(
    metadata_filter={"sector": "technology"},
    limit=10
)
```

## Memory in Cognitive Steps

Memory systems are automatically integrated into cognitive steps. For example, during a perception step, relevant memories are retrieved to provide context:

```python
class PerceptionStep(CognitiveStep):
    # ...
    
    async def execute(self, agent):
        # Retrieve relevant memories
        recent_memories = await agent.short_term_memory.get_recent_memories(limit=5)
        relevant_knowledge = await agent.long_term_memory.get_memories_by_similarity(
            query=self.environment_info.get("current_state", ""),
            top_k=3
        )
        
        # Include memories in prompt context
        context = {
            "recent_memories": recent_memories,
            "relevant_knowledge": relevant_knowledge,
            "environment_info": self.environment_info
        }
        
        # Generate perception based on context
        # ...
```

## Knowledge Base Integration

For more advanced information retrieval, agents can be equipped with a Knowledge Base Agent:

```python
from market_agents.memory.knowledge_base import MarketKnowledgeBase
from market_agents.memory.knowledge_base_agent import KnowledgeBaseAgent

# Create and initialize knowledge base
market_kb = MarketKnowledgeBase(
    config=storage_config,
    table_prefix="finance_kb"
)
await market_kb.initialize()

# Create knowledge base agent
kb_agent = KnowledgeBaseAgent(market_kb=market_kb)
kb_agent.id = "finance_kb_agent"

# Add to market agent during creation
agent = await MarketAgent.create(
    # ... other parameters
    knowledge_agent=kb_agent
)

# Query the knowledge base
response = await agent.knowledge_agent.query(
    query="What are the key factors affecting Apple's stock price?",
    top_k=5
)
```

## Practical Example: Agent with Memory

Here's a complete example demonstrating memory usage:

```python
import asyncio
from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.config import AgentStorageConfig
from minference.lite.models import LLMConfig

async def agent_with_memory_example():
    # Storage configuration
    storage_config = AgentStorageConfig(
        api_url="http://localhost:8001",
        stm_top_k=5,
        ltm_top_k=10
    )
    
    # Storage utilities
    storage_utils = AgentStorageAPIUtils(config=storage_config)
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="financial_analyst_with_memory",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4",
            client="openai",
            temperature=0.7
        )
    )
    
    # Add memories
    await agent.short_term_memory.add_memory(
        content="AAPL closed at $198.45, up 2.3% for the day.",
        memory_type="observation"
    )
    
    await agent.long_term_memory.add_memory(
        content="Apple typically releases new iPhone models in September, which often impacts stock price.",
        memory_type="knowledge"
    )
    
    # Retrieve memories for a task
    query = "Apple stock performance and product releases"
    
    stm_results = await agent.short_term_memory.get_memories_by_similarity(query, top_k=3)
    ltm_results = await agent.long_term_memory.get_memories_by_similarity(query, top_k=3)
    
    # Combine memories to form context for agent reasoning
    context = {
        "recent_observations": [mem.content for mem in stm_results],
        "relevant_knowledge": [mem.content for mem in ltm_results]
    }
    
    # Use context in agent reasoning
    # ...
    
    return context

# Run the example
context = asyncio.run(agent_with_memory_example())
print(context)
```

In the next section, we'll explore how to integrate agents with different environment mechanisms.
