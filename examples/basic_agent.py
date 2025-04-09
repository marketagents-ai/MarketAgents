import json
import asyncio
from market_agents.agents.market_agent import MarketAgent
from market_agents.agents.personas.persona import Persona
from market_agents.environments.mechanisms.chat import ChatEnvironment
from market_agents.memory.config import AgentStorageConfig
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from minference.lite.models import LLMConfig, ResponseFormat

async def main():
    """
    Market Agent with a research analyst persona.
    """
    # Initialize storage
    storage_config = AgentStorageConfig(
        model="text-embedding-3-small",
        embedding_provider="openai",
        vector_dim=256
    )
    storage_utils = AgentStorageAPIUtils(config=storage_config)
    
    # Create chat environment
    chat_env = ChatEnvironment(name="market_chat")
    
    # Create persona
    persona = Persona(
        role="Research Analyst",
        persona="You are a market research analyst specializing in technology sector analysis.",
        objectives=[
            "Identify investment opportunities in the tech sector"
        ]
    )
    # Create agent
    agent = await MarketAgent.create(
        persona=persona,
        llm_config=LLMConfig(
            model="gpt-4o-mini",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.tool
        ),
        environments={"chat": chat_env},
        storage_utils=storage_utils,
    )  
    # Assign a task to the agent
    agent.task = "Key factors to consider when evaluating semiconductor stocks."

    # Run a single action step
    #step_result = await agent.run_step()
    #print(f"Response: {json.dumps(step_result, indent=2)}")

    # Run a full cognitive episode [Perception > Action > Reflection]
    episode_result = await agent.run_episode()
    print(f"Response: {json.dumps(episode_result, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())