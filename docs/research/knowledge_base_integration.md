# Knowledge Base Integration

## Overview

A powerful feature of the MarketAgents research capabilities is the integration with knowledge bases. This allows agents to store, retrieve, and utilize structured information during the research process, enhancing their ability to conduct comprehensive and informed research.

## Knowledge Base Architecture

The knowledge base system in MarketAgents consists of several components:

1. **MarketKnowledgeBase**: Core component for storing and retrieving knowledge
2. **KnowledgeBaseAgent**: Agent interface for interacting with knowledge bases
3. **Storage Backend**: Vector database for efficient knowledge storage and retrieval

## Setting Up a Knowledge Base

To set up a knowledge base for research, you'll need to:

1. Configure the storage backend
2. Initialize a MarketKnowledgeBase
3. Create a KnowledgeBaseAgent
4. Integrate with research agents

```python
from market_agents.memory.config import AgentStorageConfig
from market_agents.memory.knowledge_base import MarketKnowledgeBase
from market_agents.memory.knowledge_base_agent import KnowledgeBaseAgent

# Configure storage
storage_config = AgentStorageConfig(
    api_url="http://localhost:8001",
    embedding_model="text-embedding-ada-002",
    vector_dimension=1536
)

# Initialize knowledge base
market_kb = MarketKnowledgeBase(
    config=storage_config,
    table_prefix="finance_research_kb"
)
await market_kb.initialize()

# Create knowledge base agent
kb_agent = KnowledgeBaseAgent(market_kb=market_kb)
kb_agent.id = "finance_kb_agent"
```

## Integrating Knowledge Base with Research

You can integrate the knowledge base with the research environment:

```python
# Create Research mechanism with knowledge base integration
research_mechanism = Research(
    topic="Impact of AI on financial markets",
    knowledge_base_name="finance_research_kb",
    store_findings_in_kb=True  # Automatically store findings in knowledge base
)
```

## Adding Information to Knowledge Base

During research, agents can add information to the knowledge base:

```python
# Add a research finding to the knowledge base
await kb_agent.add_document(
    content="AI-driven trading algorithms now account for over 70% of daily trading volume in major exchanges.",
    metadata={
        "type": "research_finding",
        "topic": "Impact of AI on financial markets",
        "subtopic": "AI adoption in trading",
        "sources": ["Financial Times report", "Market analysis data"],
        "confidence": 0.85
    }
)
```

## Querying the Knowledge Base

Agents can query the knowledge base during research:

```python
# Query the knowledge base
results = await kb_agent.query(
    query="What is the adoption rate of AI in trading?",
    top_k=3
)

# Process query results
for result in results:
    print(f"Content: {result.content}")
    print(f"Similarity: {result.similarity}")
    print(f"Metadata: {result.metadata}")
```

## Knowledge Base in Research Workflow

The knowledge base can be integrated into the research workflow:

```python
class KnowledgeEnhancedResearchStep(CognitiveStep):
    step_name: str = Field(default="knowledge_research", description="Knowledge-enhanced research step")
    
    async def execute(self, agent):
        # Query knowledge base for relevant information
        kb_results = await agent.knowledge_agent.query(
            query=f"Information about {self.environment_info.get('assigned_subtopic')}",
            top_k=5
        )
        
        # Incorporate knowledge base results into research
        kb_context = "\n".join([
            f"- {result.content} (Confidence: {result.metadata.get('confidence', 'N/A')})"
            for result in kb_results
        ])
        
        # Generate research findings with knowledge base context
        prompt = f"""
        You are {agent.role}. {agent.persona}
        
        Research topic: {self.environment_info.get('topic')}
        Your assigned subtopic: {self.environment_info.get('assigned_subtopic')}
        
        Relevant information from knowledge base:
        {kb_context}
        
        Based on your knowledge and the information from the knowledge base, provide 3-5 key findings about your assigned subtopic.
        """
        
        response = await agent.llm_orchestrator.generate(
            model=agent.llm_config.model,
            messages=[{"role": "system", "content": prompt}]
        )
        
        # Process and submit findings
        # ...
        
        return response.content
```

## Collaborative Knowledge Building

Multiple agents can contribute to the same knowledge base, creating a collaborative knowledge building process:

```python
# Each agent contributes to the shared knowledge base
for agent in agents:
    # Create a research finding
    finding = {
        "content": f"Finding from {agent.id}: ...",
        "subtopic": agent.assigned_subtopic,
        "sources": ["..."],
        "confidence": 0.8
    }
    
    # Add to knowledge base
    await kb_agent.add_document(
        content=finding["content"],
        metadata={
            "type": "research_finding",
            "agent_id": agent.id,
            "subtopic": finding["subtopic"],
            "sources": finding["sources"],
            "confidence": finding["confidence"]
        }
    )
```

## Knowledge Base Persistence

Knowledge bases persist beyond individual research sessions, allowing for cumulative knowledge building:

```python
# Check if knowledge base exists
exists = await market_kb.check_table_exists()

if exists:
    # Use existing knowledge base
    print("Using existing knowledge base")
else:
    # Initialize new knowledge base
    print("Creating new knowledge base")
    await market_kb.create_tables()
```

## Complete Example: Research with Knowledge Base

Here's a complete example of integrating a knowledge base with the research process:

```python
import asyncio
from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.config import AgentStorageConfig
from market_agents.memory.knowledge_base import MarketKnowledgeBase
from market_agents.memory.knowledge_base_agent import KnowledgeBaseAgent
from market_agents.environments.mechanisms.research import Research, ResearchEnvironment
from market_agents.agents.cognitive_steps import CognitiveStep, CognitiveEpisode
from minference.lite.models import LLMConfig

async def research_with_knowledge_base():
    # Configure storage
    storage_config = AgentStorageConfig(
        api_url="http://localhost:8001",
        embedding_model="text-embedding-ada-002",
        vector_dimension=1536
    )
    
    # Initialize storage utilities
    storage_utils = AgentStorageAPIUtils(config=storage_config)
    
    # Initialize knowledge base
    market_kb = MarketKnowledgeBase(
        config=storage_config,
        table_prefix="finance_research_kb"
    )
    await market_kb.initialize()
    
    # Check if knowledge base exists, create if not
    exists = await market_kb.check_table_exists()
    if not exists:
        await market_kb.create_tables()
    
    # Create knowledge base agent
    kb_agent = KnowledgeBaseAgent(market_kb=market_kb)
    kb_agent.id = "finance_kb_agent"
    
    # Add some initial knowledge
    await kb_agent.add_document(
        content="AI trading systems now account for approximately 70% of all US equity trading volume.",
        metadata={
            "type": "fact",
            "topic": "AI in trading",
            "source": "Financial Times, 2024",
            "confidence": 0.9
        }
    )
    
    # Create research agents with knowledge base
    agents = []
    for i, role in enumerate(["Financial Analyst", "Market Researcher", "Regulatory Expert"]):
        agent = await MarketAgent.create(
            storage_utils=storage_utils,
            agent_id=f"researcher_{i}",
            use_llm=True,
            llm_config=LLMConfig(
                model="gpt-4",
                client="openai",
                temperature=0.7
            ),
            knowledge_agent=kb_agent  # Attach knowledge base agent
        )
        agents.append(agent)
    
    # Create Research mechanism
    research_mechanism = Research(
        topic="Impact of AI on financial markets",
        max_rounds=2,
        subtopics=[
            "AI adoption in trading",
            "Market efficiency impacts",
            "Regulatory considerations"
        ],
        assign_subtopics=True,
        store_findings_in_kb=True  # Store findings in knowledge base
    )
    
    # Create Research environment
    research_env = ResearchEnvironment(
        name="ai_finance_research",
        mechanism=research_mechanism
    )
    
    # Add environment to each agent
    for agent in agents:
        agent.environments["research"] = research_env
    
    # Define knowledge-enhanced research step
    class KnowledgeEnhancedResearchStep(CognitiveStep):
        step_name: str = Field(default="knowledge_research", description="Knowledge-enhanced research step")
        
        async def execute(self, agent):
            # Get current research state
            research_state = agent.environments[self.environment_name].get_global_state(agent_id=agent.id)
            
            # Query knowledge base for relevant information
            kb_results = await agent.knowledge_agent.query(
                query=f"Information about {research_state.get('assigned_subtopic', 'AI in finance')}",
                top_k=3
            )
            
            # Incorporate knowledge base results into research
            kb_context = "\n".join([
                f"- {result.content} (Source: {result.metadata.get('source', 'Unknown')})"
                for result in kb_results
            ])
            
            # Generate research findings
            prompt = f"""
            You are {agent.role}. Your task is to research {research_state.get('topic')}.
            
            Your assigned subtopic: {research_state.get('assigned_subtopic', 'AI in finance')}
            
            Relevant information from knowledge base:
            {kb_context}
            
            Based on your knowledge and the information from the knowledge base, provide 3 key findings about your assigned subtopic.
            """
            
            response = await agent.llm_orchestrator.generate(
                model=agent.llm_config.model,
                messages=[{"role": "system", "content": prompt}]
            )
            
            # Submit finding to environment and knowledge base
            finding = {
                "action_type": "research_finding",
                "content": response.content,
                "subtopic": research_state.get('assigned_subtopic', 'AI in finance'),
                "sources": ["Knowledge base", "Agent analysis"],
                "confidence": 0.85
            }
            
            agent.environments[self.environment_name].step(finding)
            
            # Also add to knowledge base directly
            await agent.knowledge_agent.add_document(
                content=response.content,
                metadata={
                    "type": "research_finding",
                    "agent_id": agent.id,
                    "topic": research_state.get('topic'),
                    "subtopic": research_state.get('assigned_subtopic', 'AI in finance'),
                    "confidence": 0.85
                }
            )
            
            return response.content
    
    # Run research with knowledge enhancement
    research_episode = CognitiveEpisode(
        steps=[KnowledgeEnhancedResearchStep],
        environment_name="research"
    )
    
    # Run for each agent
    results = []
    for agent in agents:
        agent_result = await agent.run_episode(research_episode)
        results.append({
            "agent_id": agent.id,
            "result": agent_result
        })
    
    # Get final research output
    final_output = research_env.mechanism.get_research_output()
    
    # Query knowledge base for all research findings
    all_findings = await kb_agent.query(
        query="Research findings about AI in financial markets",
        filter_metadata={"type": "research_finding"},
        top_k=10
    )
    
    return {
        "agent_results": results,
        "final_output": final_output,
        "knowledge_base_findings": [
            {"content": finding.content, "metadata": finding.metadata}
            for finding in all_findings
        ]
    }

# Run the research with knowledge base
results = asyncio.run(research_with_knowledge_base())
```

In the next section, we'll explore how to orchestrate multi-agent research tasks.
