from abc import ABC, abstractmethod
from typing import List, Optional
import logging

from pydantic import BaseModel, Field
from market_agents.orchestrators.config import OrchestratorConfig
from market_agents.memory.agent_storage.storage_service import StorageService
from market_agents.agents.market_agent import MarketAgent
from minference.lite.inference import InferenceOrchestrator

class BaseEnvironmentOrchestrator(BaseModel, ABC):
    """
    A base orchestrator abstraction that coordinates round-based logic for
    any environment simulation.
    """

    config: BaseModel = Field(
        ...,
        description=("Environment-specific configuration model, which may contain named API URLs"),
    )
    orchestrator_config: OrchestratorConfig = Field(
        ...,
        description=(
            "High-level orchestrator config specifying parameters such as environment_order, max_rounds, etc."
        ),
    )
    agents: List["MarketAgent"] = Field(
        ...,
        description="List of agents participating in this environment orchestrator.",
    )
    cohorts: Optional[List[List["MarketAgent"]]] = Field(
        default=None,
        description="Pre-formed agent cohorts to be used across environments"
    )
    storage_service: "StorageService" = Field(
        ...,
        description="Interface for agent memory and data storage.",
    )
    logger: logging.Logger = Field(
        default_factory=lambda: logging.getLogger(__name__),
        description=("Logger instance used for recording debug information, warnings, etc."),
    )
    environment_name: str = Field(
        default="",
        description="Human-readable label or identifier for this environment orchestrator.",
    )

    ai_utils: InferenceOrchestrator = Field(
        default=None,
        description="Instance of InferenceOrchestrator used for parallel LLM completion."
    )

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self, **data):
        super().__init__(**data)
        if not self.logger:
            self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    async def setup_environment(self) -> None:
        pass

    @abstractmethod
    async def run_environment(self, round_num: Optional[int] = None) -> None:
        pass

    @abstractmethod
    async def process_round_results(self, round_num: int) -> None:
        pass

    @abstractmethod
    async def get_round_summary(self, round_num: int) -> dict:
        pass

    async def print_summary(self) -> None:
        pass

    async def run(self) -> None:
        await self.setup_environment()
        await self.run_environment()
        await self.print_summary()