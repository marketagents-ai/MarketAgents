# base_environment_orchestrator.py
from typing import List, Union, Dict
from market_agents.agents.market_agent import MarketAgent
from market_agents.inference.parallel_inference import ParallelAIUtilities
from market_agents.orchestrators.insert_simulation_data import SimulationDataInserter
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from market_agents.orchestrators.config import AuctionConfig, GroupChatConfig
import logging

class BaseEnvironmentOrchestrator(BaseModel, ABC):
    config: Union[AuctionConfig, GroupChatConfig]
    agents: List['MarketAgent']
    ai_utils: 'ParallelAIUtilities'
    data_inserter: 'SimulationDataInserter'
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
    def setup_environment(self):
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
        # Implement common result processing logic if any
        pass

    async def run(self):
        # Implement common run logic if any
        pass
