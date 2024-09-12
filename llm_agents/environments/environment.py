from typing import Dict, Any, List, Optional, Type, Union
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

class Environment(BaseModel, ABC):
    name: str = Field(..., description="Name of the environment")
    address: str = Field(..., description="Address of the environment for orchestrator linking")
    global_state: Dict[str, Any] = Field(default_factory=dict, description="Global state of the environment")
    current_step: int = Field(default=0, description="Current step/round of the simulation")
    max_steps: int = Field(..., description="Maximum number of steps/rounds for this environment")

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
    def reset(self) -> Dict[str, Any]:
        """Reset the environment and return the initial global state."""
        self.current_step = 0
        return self.get_global_state()

    @abstractmethod
    def render(self):
        """Render the environment."""
        pass

    @abstractmethod
    def parse_action(self, action: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse an action (either as a string or a dictionary) into a format the environment can use.
        This method should be flexible to accommodate different message protocols (e.g., ACL).
        """
        pass

    @abstractmethod
    def step(self) -> Dict[str, Any]:
        """
        Advance the environment by one step/round.
        This method should be called by the orchestrator to progress the simulation.
        """
        if self.current_step < self.max_steps:
            self.current_step += 1
        return self.get_global_state()

    def get_current_step(self) -> int:
        """
        Return the current step/round of the simulation.
        """
        return self.current_step

    class Config:
        arbitrary_types_allowed = True