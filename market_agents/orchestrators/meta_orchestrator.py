# meta_orchestrator.py

import asyncio
import logging
import warnings
from typing import List, Dict, Type

from market_agents.orchestrators.base_orchestrator import BaseEnvironmentOrchestrator
from market_agents.orchestrators.config import OrchestratorConfig
from market_agents.orchestrators.insert_simulation_data import SimulationDataInserter
from market_agents.orchestrators.logger_utils import (
    orchestration_logger,
    log_round,
    print_ascii_art,
    log_section,
    log_completion,
    log_environment_setup,
)
from market_agents.inference.parallel_inference import ParallelAIUtilities, RequestLimits
from market_agents.memory.setup_db import DatabaseConnection
from market_agents.memory.embedding import MemoryEmbedder
from market_agents.memory.config import MarketMemoryConfig, load_config_from_yaml


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

        # If you want to keep references to memory/DB for logging, do so:
        self.memory_config = self._initialize_memory_config()
        self.db_conn = self._initialize_database()
        self.embedder = MemoryEmbedder(config=self.memory_config)
        self.ai_utils = self._initialize_ai_utils()
        self.data_inserter = self._initialize_data_inserter()

        self.environment_orchestrators: Dict[str, BaseEnvironmentOrchestrator] = {}

    def _initialize_memory_config(self) -> MarketMemoryConfig:
        # Load from your memory_config.yaml
        config_path = "market_agents/memory/memory_config.yaml"
        return load_config_from_yaml(config_path)

    def _initialize_database(self) -> DatabaseConnection:
        db_conn = DatabaseConnection(self.memory_config)
        db_conn._ensure_database_exists()
        return db_conn

    def _initialize_ai_utils(self) -> ParallelAIUtilities:
        # Just an example with default large request limits
        oai_request_limits = RequestLimits(max_requests_per_minute=20000, max_tokens_per_minute=2000000)
        anthropic_request_limits = RequestLimits(max_requests_per_minute=20000, max_tokens_per_minute=2000000)
        return ParallelAIUtilities(
            oai_request_limits=oai_request_limits,
            anthropic_request_limits=anthropic_request_limits
        )

    def _initialize_data_inserter(self) -> SimulationDataInserter:
        db_config = self.config.database_config
        db_params = {
            "dbname": db_config.db_name,
            "user": db_config.db_user,
            "password": db_config.db_password,
            "host": db_config.db_host,
            "port": db_config.db_port
        }
        return SimulationDataInserter(db_params)

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
                "data_inserter": self.data_inserter,
                "logger": self.logger
            }

            orchestrator = orchestrator_class(**kwargs)
            self.environment_orchestrators[env_name] = orchestrator
            self.logger.info(f"Initialized {orchestrator_class.__name__} for environment '{env_name}'")

    async def run_simulation(self):
        """
        Main simulation loop:
         - Setup environments
         - For each round: run each environment in order
         - Print summaries
        """
        self._initialize_environment_orchestrators()

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

    async def start(self):
        print_ascii_art()
        log_section(self.logger, "Simulation Starting")
        try:
            await self.run_simulation()
            log_completion(self.logger, "Simulation completed successfully")
        finally:
            self.db_conn.close()