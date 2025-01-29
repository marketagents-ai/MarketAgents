# meta_orchestrator.py

import asyncio
import logging
import warnings
from typing import List, Dict, Optional, Type

from market_agents.memory.agent_storage.setup_db import AsyncDatabase
from market_agents.memory.agent_storage.storage_service import StorageService
from market_agents.memory.embedding import MemoryEmbedder
from market_agents.orchestrators.base_orchestrator import BaseEnvironmentOrchestrator
from market_agents.orchestrators.config import OrchestratorConfig
from market_agents.orchestrators.orchestration_data_inserter import OrchestrationDataInserter
from market_agents.orchestrators.logger_utils import (
    orchestration_logger,
    log_round,
    print_ascii_art,
    log_section,
    log_completion,
    log_environment_setup,
)
from market_agents.inference.parallel_inference import ParallelAIUtilities, RequestLimits
from market_agents.memory.config import AgentStorageConfig, load_config_from_yaml
from market_agents.orchestrators.setup_orchestrator_db import setup_orchestrator_tables


warnings.filterwarnings("ignore", module="pydantic")


class MetaOrchestrator:
    """
    A top-level orchestrator that:
      - Receives a set of pre-created agents (with or without knowledge bases).
      - Sets up environment orchestrators using environment_order from the config.
      - Runs multiple simulation rounds across these environments.
    """
    def __init__(
        self,
        config: OrchestratorConfig,
        agents,
        orchestrator_registry: Dict[str, Type[BaseEnvironmentOrchestrator]],
        logger: logging.Logger = None
    ):
        self.config = config
        self.agents = agents
        self.logger = logger or orchestration_logger
        self.environment_order = config.environment_order
        self.orchestrator_registry = orchestrator_registry
        self.ai_utils = self._initialize_ai_utils()

        # Initialize storage components
        self.storage_config = self._initialize_storage_config()
        self.db = AsyncDatabase(self.storage_config)
        self.embedder = MemoryEmbedder(self.storage_config)
        self.storage_service = StorageService(
            db=self.db,
            embedding_service=self.embedder,
            config=self.storage_config
        )
        self.data_inserter = OrchestrationDataInserter(storage_service=self.storage_service)
        self.environment_orchestrators: Dict[str, BaseEnvironmentOrchestrator] = {}



    def _initialize_storage_config(self) -> AgentStorageConfig:
        # Load from your storage_config.yaml
        config_path = "market_agents/memory/storage_config.yaml"
        return load_config_from_yaml(config_path)

    def _initialize_ai_utils(self) -> ParallelAIUtilities:
        # Just an example with default large request limits
        oai_request_limits = RequestLimits(max_requests_per_minute=20000, max_tokens_per_minute=2000000)
        anthropic_request_limits = RequestLimits(max_requests_per_minute=20000, max_tokens_per_minute=2000000)
        return ParallelAIUtilities(
            oai_request_limits=oai_request_limits,
            anthropic_request_limits=anthropic_request_limits
        )

    def _initialize_environment_orchestrators(self):
        for env_name in self.environment_order:
            env_cfg = self.config.environment_configs.get(env_name)
            if not env_cfg:
                self.logger.warning(f"No config for environment '{env_name}'. Skipping.")
                continue

            orchestrator_class = self.orchestrator_registry.get(env_name)
            if not orchestrator_class:
                self.logger.warning(f"No orchestrator registry entry for '{env_name}'. Skipping.")
                continue

            kwargs = {
                "config": env_cfg,
                "orchestrator_config": self.config,
                "agents": self.agents,
                "ai_utils": self.ai_utils,
                "storage_service": self.storage_service,
                "data_inserter": self.data_inserter,
                "logger": self.logger
            }

            orchestrator = orchestrator_class(**kwargs)
            self.environment_orchestrators[env_name] = orchestrator
            self.logger.info(f"Initialized {orchestrator_class.__name__} for environment '{env_name}'")

    async def run_orchestration(self):
        """
        Main simulation loop:
        - Setup environments
        - For each round: run each environment in order
        - Print summaries
        """
        self._initialize_environment_orchestrators()

        # Store initial agent states once at the beginning
        await self._store_agent_states()

        # Setup step
        for env_name, orch in self.environment_orchestrators.items():
            self.logger.info(f"Setting up {env_name} environment...")
            await orch.setup_environment()

        # Round loop
        for round_num in range(1, self.config.max_rounds + 1):
            log_round(self.logger, round_num)

            for env_name in self.environment_order:
                orchestrator = self.environment_orchestrators.get(env_name)
                if not orchestrator:
                    continue

                log_environment_setup(self.logger, env_name)
                try:
                    await orchestrator.run_environment(round_num)
                    await orchestrator.process_round_results(round_num)
                except Exception as e:
                    self.logger.error(f"Error in '{env_name}' environment, round {round_num}: {e}")
                    raise e

        # Summaries
        for env_name, orch in self.environment_orchestrators.items():
            if orch:
                await orch.print_summary()

    async def _store_agent_states(self):
        """Store or update agent states including economic data."""
        try:
            agents_data = []
            for agent in self.agents:
                agent_data = {
                    'id': agent.id,
                    'role': getattr(agent, 'role', 'default'),
                    'persona': getattr(agent, 'persona', {}),
                    'is_llm': getattr(agent, 'use_llm', True),
                    'max_iter': getattr(agent, 'max_iter', 0),
                    'llm_config': agent.llm_config.model_dump() if hasattr(agent.llm_config, 'model_dump') 
                                else agent.llm_config,
                    'economic_agent': agent.economic_agent.serialize() if agent.economic_agent else {}
                }
                agents_data.append(agent_data)

            await self.data_inserter.insert_agents(agents_data)
            self.logger.info(f"Stored states for {len(agents_data)} agents")

        except Exception as e:
            self.logger.error(f"Error storing agent states: {e}")
            self.logger.exception("Exception details:")
            raise

    async def _setup_tables(self):
        """Initialize database tables"""
        try:
            async with self.db.pool.acquire() as conn:
                await setup_orchestrator_tables(conn)
        except Exception as e:
            self.logger.error(f"Error setting up tables: {e}")
            raise

    async def start(self):
        print_ascii_art()
        log_section(self.logger, "Simulation Starting")
        try:
            # Initialize database and create tables
            await self.db.initialize()
            await self._setup_tables()
            
            # Run simulation
            await self.run_orchestration()
            log_completion(self.logger, "Simulation completed successfully")
        finally:
            await self.db.close()