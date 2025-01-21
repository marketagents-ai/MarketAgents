# main.py

import asyncio
import importlib
import logging
import random
import uuid
from pathlib import Path
from typing import List, Optional, Type

from market_agents.agents.market_agent import MarketAgent
from market_agents.agents.personas.persona import generate_persona
from market_agents.memory.knowledge_base import MarketKnowledgeBase
from market_agents.memory.knowledge_base_agent import KnowledgeBaseAgent
from market_agents.memory.vector_search import MemoryRetriever
from market_agents.memory.embedding import MemoryEmbedder
from market_agents.memory.config import MarketMemoryConfig, load_config_from_yaml
from market_agents.memory.setup_db import DatabaseConnection

from market_agents.orchestrators.config import load_config
from market_agents.orchestrators.meta_orchestrator import MetaOrchestrator
from market_agents.agents.protocols.acl_message import ACLMessage
from market_agents.inference.message_models import LLMConfig

from market_agents.orchestrators.groupchat_orchestrator import GroupChatOrchestrator
from market_agents.orchestrators.research_orchestrator import ResearchOrchestrator
from pydantic import BaseModel


def create_kb_agent(
    memory_config: MarketMemoryConfig,
    db_conn: DatabaseConnection,
    kb_name: str
) -> Optional[KnowledgeBaseAgent]:
    
    if not kb_name:
        return None

    embedding_service = MemoryEmbedder(config=memory_config)
    market_kb = MarketKnowledgeBase(
        config=memory_config,
        db_conn=db_conn,
        embedding_service=embedding_service,
        table_prefix=kb_name
    )
    # Check if table exists
    if not market_kb.check_table_exists(db_conn, kb_name):
        print(f"KB '{kb_name}' table not found or empty, skipping.")
        return None

    retriever = MemoryRetriever(
        config=memory_config,
        db_conn=db_conn,
        embedding_service=embedding_service
    )
    kb_agent = KnowledgeBaseAgent(
        market_kb=market_kb,
        retriever=retriever
    )
    print(f"Loaded knowledge base '{kb_name}'.")
    return kb_agent


def create_agents(
    config,
    memory_config: MarketMemoryConfig,
    db_conn: DatabaseConnection
) -> List[MarketAgent]:
    
    agents = []
    # Number of agents from config
    num_agents = config.num_agents

    # setup knowledge base
    kb_name = config.agent_config.knowledge_base
    kb_agent = create_kb_agent(memory_config, db_conn, kb_name)

    # randomly choose from config.llm_configs for each agent
    llm_confs = config.llm_configs

    for i in range(num_agents):
        persona = generate_persona()
        persona.role = "market_researcher"

        # Randomly pick an LLM config
        llm_c = random.choice(llm_confs)

        knowledge_agent = kb_agent

        # Create MarketAgent
        agent_id = str(uuid.uuid4())
        agent = MarketAgent.create(
            memory_config=memory_config,
            db_conn=db_conn,
            agent_id=agent_id,
            use_llm=True,
            llm_config=LLMConfig(
                name=llm_c.name,
                client=llm_c.client,
                model=llm_c.model,
                temperature=llm_c.temperature,
                max_tokens=llm_c.max_tokens,
                use_cache=llm_c.use_cache
            ),
            environments={},
            protocol=ACLMessage,
            persona=persona,
            econ_agent=None,
            knowledge_agent=knowledge_agent
        )

        agent.index = i
        agents.append(agent)

    return agents


async def main():
    # Load orchestrator config
    config_path = Path("market_agents/orchestrators/orchestrator_config.yaml")
    config = load_config(config_path)

    # Load memory config & DB for knowledge base usage
    memory_config_path = "market_agents/memory/memory_config.yaml"
    memory_config = load_config_from_yaml(memory_config_path)
    db_conn = DatabaseConnection(memory_config)
    db_conn._ensure_database_exists()

    # Create Agents
    agents = create_agents(config, memory_config, db_conn)

    # Build orchestrator registry
    orchestrator_registry = {
        #"group_chat": GroupChatOrchestrator,
        "research": ResearchOrchestrator
    }

    # Instantiate MetaOrchestrator with loaded config, your agents, and registry
    meta_orch = MetaOrchestrator(
        config=config,
        agents=agents,
        orchestrator_registry=orchestrator_registry,
        logger=logging.getLogger("meta_orchestrator")
    )

    await meta_orch.start()
    db_conn.close()

if __name__ == "__main__":
    asyncio.run(main())