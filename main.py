# main.py

import asyncio
import logging
import random
import uuid
from pathlib import Path
from typing import List, Optional

from market_agents.agents.market_agent import MarketAgent
from market_agents.agents.personas.persona import load_or_generate_personas

from market_agents.economics.econ_agent import EconomicAgent
from market_agents.memory.knowledge_base import MarketKnowledgeBase
from market_agents.memory.knowledge_base_agent import KnowledgeBaseAgent
from market_agents.memory.config import AgentStorageConfig, load_config_from_yaml
from market_agents.orchestrators.config import load_config
from market_agents.orchestrators.meta_orchestrator import MetaOrchestrator
from market_agents.agents.protocols.acl_message import ACLMessage

from market_agents.orchestrators.groupchat_orchestrator import GroupChatOrchestrator
from market_agents.orchestrators.research_orchestrator import ResearchOrchestrator

from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils

from minference.lite.models import LLMConfig, ResponseFormat

import logging

logger = logging.getLogger(__name__)


async def create_kb_agent(
    config: AgentStorageConfig,
    kb_name: str
) -> Optional[KnowledgeBaseAgent]:
    """
    Create and initialize a KnowledgeBaseAgent if kb_name is provided and
    the storage API passes the health check. Returns None if creation fails.
    """
    if not kb_name:
        logger.info("No knowledge base specified in config")
        return None

    try:
        storage_utils = AgentStorageAPIUtils(
            config=config,
            logger=logging.getLogger("storage_api")
        )
        is_healthy = await storage_utils.check_api_health()
        if not is_healthy:
            logger.error("Storage API health check failed")
            return None

        logger.info(f"Creating MarketKnowledgeBase with prefix '{kb_name}'")

        market_kb = MarketKnowledgeBase(
            config=config,
            table_prefix=kb_name
        )
        
        logger.info(f"Initializing knowledge base '{kb_name}'")
        await market_kb.initialize()
        
        logger.info(f"Checking if knowledge base '{kb_name}' exists")
        exists = await market_kb.check_table_exists()
        if not exists:
            logger.warning(f"Knowledge base '{kb_name}' tables not found or empty")
            return None

        kb_agent = KnowledgeBaseAgent(market_kb=market_kb)
        kb_agent.id = f"{kb_name}_agent"

        logger.info(f"Successfully initialized knowledge base agent '{kb_name}_agent'")
        return kb_agent
        
    except Exception as e:
        logger.error(f"Failed to initialize knowledge base '{kb_name}': {str(e)}")
        return None


async def create_agents(
    config,
    storage_config: AgentStorageConfig,
) -> List[MarketAgent]:
    """
    Create market agents using the updated MarketAgent framework.
    Optionally associates a single KnowledgeBaseAgent with all agents if specified.
    """
    agents = []
    num_agents = config.num_agents
    storage_utils = AgentStorageAPIUtils(config=storage_config)

    kb_agent = await create_kb_agent(storage_config, config.agent_config.knowledge_base)
    if not kb_agent and config.agent_config.knowledge_base:
        logger.warning("Failed to initialize knowledge base agent, continuing without it")

    personas_dir = Path("./market_agents/agents/personas/generated_personas")
    personas = load_or_generate_personas(personas_dir, num_agents)

    if config.tool_mode:
        response_format = ResponseFormat.tool
    else:
        response_format = ResponseFormat.json_object

    llm_confs = config.llm_configs
    if not llm_confs:
        raise ValueError("No LLM configurations found in config")

    for i, persona in enumerate(personas):
        agent_id = f"agent_{i}"
        llm_c = random.choice(llm_confs)

        econ_agent = EconomicAgent(
            generate_wallet=True,
            initial_holdings={
                "ETH": 1.0,
                "USDC": 1000.0
            }
        )

        try:
            agent = await MarketAgent.create(
                storage_utils=storage_utils,
                agent_id=agent_id,
                use_llm=True,
                llm_config=LLMConfig(
                    model=llm_c.model,
                    client=llm_c.client,
                    temperature=llm_c.temperature,
                    max_tokens=llm_c.max_tokens,
                    use_cache=llm_c.use_cache,
                    response_format=response_format
                ),
                environments={},
                protocol=ACLMessage,
                persona=persona,
                econ_agent=econ_agent,
                knowledge_agent=kb_agent
            )

            logger.info(
                f"Created agent {agent_id} with LLM client={llm_c.client}, "
                f"persona={persona.role}, economic agent wallet={econ_agent.wallet}"
            )
            agents.append(agent)

        except Exception as e:
            logger.error(f"Failed to create agent {agent_id}: {str(e)}")
            continue

    if not agents:
        raise RuntimeError("Failed to create any agents")

    return agents


async def main():
    logging.basicConfig(level=logging.INFO)
    config = load_config("market_agents/orchestrators/orchestrator_config.yaml")
    storage_config = load_config_from_yaml("market_agents/memory/storage_config.yaml")

    agents = await create_agents(config, storage_config)

    orchestrator_registry = {
        "groupchat": GroupChatOrchestrator,
        "research": ResearchOrchestrator
    }

    meta_orch = MetaOrchestrator(
        config=config,
        agents=agents,
        orchestrator_registry=orchestrator_registry
    )
    await meta_orch.run_orchestration()


if __name__ == "__main__":
    asyncio.run(main())