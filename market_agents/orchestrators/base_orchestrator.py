from typing import List
from market_agents.agents.market_agent import MarketAgent
from market_agents.inference.parallel_inference import ParallelAIUtilities
from market_agents.orchestrators.config import OrchestratorConfig
from market_agents.memory.agent_storage.storage_service import StorageService
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
import logging

class BaseEnvironmentOrchestrator(BaseModel, ABC):
    config: BaseModel
    orchestrator_config: OrchestratorConfig
    agents: List['MarketAgent']
    ai_utils: 'ParallelAIUtilities'
    storage_service: 'StorageService'
    logger: logging.Logger = Field(default=None)
    environment_name: str = Field(default="")

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self, **data):
        super().__init__(**data)
        if self.logger is None:
            self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def setup_environment(self):
        pass

    @abstractmethod
    async def run_environment(self, round_num: int):
        pass

    @abstractmethod
    def process_environment_state(self, env_state):
        pass

    @abstractmethod
    def get_round_summary(self, round_num: int) -> dict:
        pass

    async def process_round_results(self, round_num: int):
        pass

    async def run(self):
        pass