from typing import Dict, Any, List, Optional, Type
from pydantic import BaseModel, Field, computed_field
from market_agents.environments.environment import (
    Mechanism, LocalAction, GlobalAction, LocalObservation, GlobalObservation,
    LocalEnvironmentStep, EnvironmentStep, ActionSpace, ObservationSpace,
    FloatAction
)
from statistics import mean

class BeautyContestAction(FloatAction):
    action: float = Field(..., description="Value of the guess in between 0 and 100", ge=0, le=100)

class Prize(BaseModel):
    is_winner: bool = Field(..., description="Whether this agent is the winner")
    prize_type: str = Field(default="Dollars", description="The prize type")
    quantity: int = Field(default=100, description="The prize quantity")

class BeautyContestLocalObservation(LocalObservation):
    observation: Prize

    @classmethod
    def sample(cls, agent_id: str) -> 'BeautyContestLocalObservation':
        return cls(
            agent_id=agent_id,
            observation=Prize(is_winner=False, prize_type="Dollars", quantity=100)
        )

class BeautyContestGlobalObservation(GlobalObservation):
    observations: Dict[str, BeautyContestLocalObservation]
    all_actions: Dict[str, float] = Field(..., description="All agents' actions")
    average: float = Field(..., description="Average of all guesses")
    target: float = Field(..., description="Target value (2/3 of average)")
    winner_id: str = Field(..., description="ID of the winning agent")
    winner_value: float = Field(..., description="Winning guess")

class BeautyContestActionSpace(ActionSpace):
    allowed_actions: List[Type[LocalAction]] = [BeautyContestAction]

class BeautyContestObservationSpace(ObservationSpace):
    allowed_observations: List[Type[LocalObservation]] = [BeautyContestLocalObservation]

class BeautyContestMechanism(Mechanism):
    target_factor: float = Field(default=2/3, description="The factor to multiply the average by")
    last_actions: Dict[str, float] = Field(default_factory=dict, description="The last actions taken by each agent")
    last_target: float = Field(default=0, description="The last target number")
    last_winner: str = Field(default="", description="The last winner's agent ID")
    last_winner_value: float = Field(default=0, description="The last winner's guess")
    action_space: BeautyContestActionSpace = Field(default_factory=BeautyContestActionSpace)
    observation_space: BeautyContestObservationSpace = Field(default_factory=BeautyContestObservationSpace)

    sequential: bool = Field(default=False, description="Whether the mechanism is sequential")

    def step(self, action: GlobalAction) -> EnvironmentStep:
        # Extract the float values from the actions
        self.last_actions = {agent_id: float(local_action.action) for agent_id, local_action in action.actions.items()}
        
        # Calculate the target number
        average = mean(self.last_actions.values())
        self.last_target = self.target_factor * average

        # Determine the winner
        self.last_winner = min(self.last_actions, key=lambda x: abs(self.last_actions[x] - self.last_target))
        self.last_winner_value = self.last_actions[self.last_winner]

        # Prepare observations
        local_observations = {}
        for agent_id, action_value in self.last_actions.items():
            is_winner = agent_id == self.last_winner
            local_observations[agent_id] = BeautyContestLocalObservation(
                agent_id=agent_id,
                observation=Prize(
                    is_winner=is_winner,
                    prize_type="Dollars",
                    quantity=100 if is_winner else 0
                )
            )

        global_observation = BeautyContestGlobalObservation(
            observations=local_observations,
            all_actions=self.last_actions,
            average=average,
            target=self.last_target,
            winner_id=self.last_winner,
            winner_value=self.last_winner_value
        )

        return EnvironmentStep(
            global_observation=global_observation,
            done=False,  # The beauty contest can continue indefinitely
            info={"current_round": getattr(self, 'current_round', 0) + 1}
        )

    def get_global_state(self) -> Dict[str, Any]:
        return {
            "last_actions": self.last_actions,
            "last_target": self.last_target,
            "last_winner": self.last_winner,
            "last_winner_value": self.last_winner_value
        }

    def reset(self) -> None:
        self.last_actions = {}
        self.last_target = 0
        self.last_winner = ""
        self.last_winner_value = 0