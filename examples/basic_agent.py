import json
import asyncio
from market_agents.agents.market_agent import MarketAgent
from market_agents.agents.personas.persona import Persona
from market_agents.environments.mechanisms.chat import ChatEnvironment
from market_agents.memory.config import AgentStorageConfig
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from minference.lite.models import LLMConfig, ResponseFormat, LLMClient

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
    
    # Create agent persona
    persona = Persona(
        role="Research Analyst",
        persona="You are a market research analyst specializing in technology sector analysis.",
        objectives=[
            "Provide actionable insights based on market research"
        ],
        skills=[
            "Technology Sector Analysis",
            "Market Research",
        ]
    )

    # Create agent
    agent = await MarketAgent.create(
        name="tech-analyst",
        persona=persona,
        llm_config=LLMConfig(
            client=LLMClient.openai,
            model="gpt-4o-mini",
            temperature=0.7,
            response_format=ResponseFormat.tool
        ),
        task="What are the key factors to consider when evaluating semiconductor stocks.",
        environments={"chat": chat_env},
        storage_utils=storage_utils
    )

    # Run a full cognitive episode [Perception > Action > Reflection]
    episode_result = await agent.run_episode()
    
    # Pretty print the results
    print("\n=== Cognitive Episode Results ===")
    for i, result in enumerate(episode_result):
        print(f"\nStep {i+1}:")
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())