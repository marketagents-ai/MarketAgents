from typing import Dict, Any, List, Optional, Type, Union
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

class Environment(BaseModel, ABC):
    name: str = Field(..., description="Name of the environment")
    address: str = Field(..., description="Address of the environment for orchestrator linking")
    global_state: Dict[str, Any] = Field(default_factory=dict, description="Global state of the environment")

    @abstractmethod
    def update(self, agent_actions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update the global state based on agent actions.
        Returns the updated global state.
        """
        pass

    @abstractmethod
    def get_observation(self, agent_id: str) -> Dict[str, Any]:
        """
        Return the observation for a specific agent.
        This includes a summary of the global state (e.g., last trade price, average price)
        and the agent's own market execution results (success/failure, price, quantity).
        """
        pass

    @abstractmethod
    def get_global_state(self) -> Dict[str, Any]:
        """
        Return a summary of the global state, including metrics like last trade price, average price, etc.
        """
        pass

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Reset the environment and return the initial global state."""
        pass

    @abstractmethod
    def render(self):
        """Render the environment."""
        pass

    @abstractmethod
    def get_action_space(self) -> Any:
        """Return the action space of the environment."""
        pass

    @abstractmethod
    def get_action_schema(self) -> Type[BaseModel]:
        """Return the Pydantic model for the action schema of this environment."""
        pass

    @abstractmethod
    def get_observation_space(self) -> Any:
        """Return the observation space of the environment."""
        pass

    @abstractmethod
    def get_global_state(self) -> Dict[str, Any]:
        """Return the current global state of the environment."""
        pass

    @abstractmethod
    def parse_action(self, action: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse an action (either as a string or a dictionary) into a format the environment can use.
        This method should be flexible to accommodate different message protocols (e.g., ACL).
        """
        pass

    class Config:
        arbitrary_types_allowed = True