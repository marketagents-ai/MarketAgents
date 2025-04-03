import asyncio
import os
from pathlib import Path

from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.config import AgentStorageConfig
from market_agents.agents.personas.persona import Persona
from minference.lite.models import LLMConfig, ResponseFormat

async def create_basic_agent():
    """
    Create a basic market agent with a simple persona.
    """
    # Configure storage
    storage_config = AgentStorageConfig(
        api_url="http://localhost:8001",
        embedding_model="text-embedding-ada-002",
        vector_dimension=1536
    )
    
    # Initialize storage utilities
    storage_utils = AgentStorageAPIUtils(config=storage_config)
    
    # Create persona
    persona = Persona(
        role="Research Analyst",
        persona="I am a research analyst specializing in technology trends and market analysis.",
        objectives=[
            "Analyze emerging technology trends",
            "Provide insights on market developments",
            "Identify investment opportunities in the tech sector"
        ]
    )
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="tech_analyst",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=persona
    )
    
    print(f"Created agent with ID: {agent.id}")
    print(f"Role: {agent.role}")
    print(f"Persona: {agent.persona}")
    print(f"Objectives: {agent.objectives}")
    
    return agent

async def ask_agent_question(agent, question):
    """
    Ask the agent a question and get a response.
    """
    print(f"\nQuestion: {question}")
    print("-" * 80)
    
    # Generate response using the agent's LLM
    response = await agent.llm_orchestrator.generate(
        model=agent.llm_config.model,
        messages=[
            {"role": "system", "content": f"You are {agent.role}. {agent.persona}"},
            {"role": "user", "content": question}
        ]
    )
    
    print(f"Response: {response.content}")
    return response.content

async def main():
    # Create a basic agent
    agent = await create_basic_agent()
    
    # Ask the agent some questions
    questions = [
        "What are the most significant AI trends to watch in 2025?",
        "How might quantum computing impact the technology landscape in the next 5 years?",
        "What are the key challenges facing semiconductor companies today?"
    ]
    
    for question in questions:
        await ask_agent_question(agent, question)

if __name__ == "__main__":
    asyncio.run(main())
