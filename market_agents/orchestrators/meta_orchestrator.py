from datetime import datetime
from typing import Dict, Type
import logging
import uuid
import warnings

from market_agents.memory.agent_storage.setup_db import AsyncDatabase
from market_agents.memory.agent_storage.storage_service import StorageService
from market_agents.memory.embedding import MemoryEmbedder
from market_agents.memory.config import AgentStorageConfig, load_config_from_yaml
from market_agents.orchestrators.base_orchestrator import BaseEnvironmentOrchestrator
from market_agents.orchestrators.config import OrchestratorConfig
from market_agents.orchestrators.requests_limits_config import OrchestratorRequestsLimits
from market_agents.orchestrators.orchestration_data_inserter import OrchestrationDataInserter
from market_agents.orchestrators.setup_orchestrator_db import setup_orchestrator_tables
from market_agents.orchestrators.logger_utils import (
    orchestration_logger, print_ascii_art,
    log_environment_setup, log_section, log_round, log_completion
)

from minference.lite.inference import InferenceOrchestrator

warnings.filterwarnings("ignore", module="pydantic")

class MetaOrchestrator:
    """
    A high-level orchestrator that instantiates environment orchestrators in config.environment_order
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

        self.storage_config = self._initialize_storage_config()
        self.db = AsyncDatabase(self.storage_config)
        self.embedder = MemoryEmbedder(self.storage_config)
        self.storage_service = StorageService(
            db=self.db,
            embedding_service=self.embedder,
            config=self.storage_config
        )

        self.ai_utils = self._initialize_ai_utils()

        for agent in self.agents:
            agent.llm_orchestrator = self.ai_utils
            
        self.data_inserter = OrchestrationDataInserter(storage_service=self.storage_service)
        self.environment_orchestrators: Dict[str, BaseEnvironmentOrchestrator] = {}

    def _initialize_storage_config(self) -> AgentStorageConfig:
        config_path = "market_agents/memory/storage_config.yaml"
        return load_config_from_yaml(config_path)

    def _initialize_ai_utils(self) -> InferenceOrchestrator:
        request_limits_data = getattr(self.config, "request_limits", {}) or {}
        
        orchestrator_limits = OrchestratorRequestsLimits.from_dict(request_limits_data)
        provider_map = orchestrator_limits.to_provider_map()

        return InferenceOrchestrator(
            oai_request_limits=provider_map["openai"],
            anthropic_request_limits=provider_map["anthropic"],
            vllm_request_limits=provider_map["vllm"],
            litellm_request_limits=provider_map["litellm"],
            local_cache=True,
            cache_folder=None
        )

    def _initialize_environment_orchestrators(self):
        for env_name in self.environment_order:
            env_cfg = self.config.environment_configs.get(env_name)
            if not env_cfg:
                self.logger.warning(f"No config found for environment '{env_name}'. Skipping.")
                continue

            orchestrator_class = self.orchestrator_registry.get(env_name)
            if not orchestrator_class:
                self.logger.warning(f"Unknown orchestrator registry entry for '{env_name}'. Skipping.")
                continue

            orchestrator = orchestrator_class(
                config=env_cfg,
                orchestrator_config=self.config,
                agents=self.agents,
                storage_service=self.storage_service,
                logger=self.logger,
                ai_utils=self.ai_utils,
            )
            self.environment_orchestrators[env_name] = orchestrator
            self.logger.info(f"Initialized {orchestrator_class.__name__} for environment '{env_name}'")

    async def run_orchestration(self):
        print_ascii_art()
        log_section(self.logger, "MarketAgents Swarm Deploying...")

        await self.db.initialize()
        await self._setup_tables()

        await self._store_agent_states()

        self._initialize_environment_orchestrators()

        for i, env_name in enumerate(self.environment_order, 1):
            env_orch = self.environment_orchestrators.get(env_name)
            if not env_orch:
                continue

            log_round(self.logger, i, f"Environment: {env_name}")
            try:
                await env_orch.run()
            except Exception as e:
                self.logger.error(f"Error in '{env_name}' environment: {e}")
                continue

        log_completion(self.logger, "All Environment Orchestrators Complete")
        await self.db.close()

    async def _store_agent_states(self):
        """Store all agent states in the database before environment simulation."""
        agents_data = []
        for agent in self.agents:
            llm_config = agent.llm_config.model_dump() if agent.llm_config else {}
            llm_config = {
                k: str(v) if isinstance(v, (uuid.UUID, datetime)) else v 
                for k, v in llm_config.items()
            }

            agent_data = {
                'id': agent.id,
                'role': getattr(agent, 'role', 'participant'),
                'persona': agent.persona,
                'is_llm': True,
                'max_iter': 0,
                'llm_config': llm_config,
                'economic_agent': {}
            }
            agents_data.append(agent_data)
        
        try:
            await self.data_inserter.insert_agents(agents_data)
            self.logger.info(f"Successfully inserted {len(agents_data)} agents into database")
        except Exception as e:
            self.logger.error(f"Error inserting agents into database: {e}")
            raise

    async def _setup_tables(self):
        """Initialize orchestrator DB tables, if needed."""
        try:
            async with self.db.pool.acquire() as conn:
                await setup_orchestrator_tables(conn)
        except Exception as e:
            self.logger.error(f"Error setting up tables: {e}")
            raise