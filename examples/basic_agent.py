import asyncio
import os
from pathlib import Path

from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.config import AgentStorageConfig
from market_agents.agents.personas.persona import Persona
from market_agents.agents.cognitive_steps import ActionStep
from market_agents.environments.mechanisms.chat import ChatEnvironment
from minference.lite.models import LLMConfig, ResponseFormat

async def create_basic_agent():
    """
    Create a basic market agent with a simple persona.
    """
    # Configure storage
    storage_config = AgentStorageConfig(
        model="text-embedding-3-small",
        embedding_provider="openai",
        vector_dim=256,
        stm_top_k=2,
        ltm_top_k=1,
        kb_top_k=3,
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

    chat_env = ChatEnvironment(name="market_chat")
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="tech_analyst",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o-mini",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        environments={"chat": chat_env},
        persona=persona
    )
    
    print(f"Created agent with ID: {agent.id}")
    print(f"Role: {agent.role}")
    print(f"Persona: {agent.persona}")
    print(f"Objectives: {agent.objectives}")
    
    return agent

async def execute_agent_task(agent, question):
    """
    Ask the agent a question and get a response.
    """
    print(f"\nQuestion: {question}")
    print("-" * 80)
    
    # Set the agent's task (question)
    agent.task = question
    
    # Run an action step to get the response
    result = await agent.run_step()
    
    if isinstance(result, dict):
        response = result.get('action', 'No response generated.')
    else:
        response = result
    
    print(f"Response: {response}")
    return response

async def main():
    # Create a basic agent
    agent = await create_basic_agent()
    
    # Ask the agent some questions
    tasks = [
        "Explain how network effects and market dynamics typically influence the growth of technology platforms.",
        "Analyze the general relationship between interest rates and growth stock valuations in technology sectors.",
        "What are the key factors to consider when evaluating semiconductor companies' competitive advantages?"
    ]
    
    for task in tasks:
        await execute_agent_task(agent, task)

if __name__ == "__main__":
    asyncio.run(main())