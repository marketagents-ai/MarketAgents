# research.py

from datetime import datetime
import json
import random
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel, Field

from market_agents.environments.environment import (
    EnvironmentHistory,
    Mechanism,
    LocalAction,
    GlobalAction,
    LocalObservation,
    GlobalObservation,
    EnvironmentStep,
    ActionSpace,
    ObservationSpace,
    MultiAgentEnvironment,
    LocalEnvironmentStep,
)
import logging

logger = logging.getLogger(__name__)


class ResearchAction(LocalAction):
    action: BaseModel

    @classmethod
    def sample(cls, agent_id: str, summary_model: Type[BaseModel]) -> 'ResearchAction':
        demo_instance = summary_model.construct()
        return cls(agent_id=agent_id, action=demo_instance)


class ResearchGlobalAction(GlobalAction):
    """Global container of local ResearchActions for each agent."""
    actions: Dict[str, ResearchAction]


class ResearchObservation(BaseModel):
    """Individual observation containing research summary data"""
    own_summary: Optional[BaseModel] = None
    aggregator_notes: str = ""

    def dict(self, *args, **kwargs):
        """Custom dict method to handle BaseModel serialization"""
        d = super().dict(*args, **kwargs)
        if self.own_summary:
            d['own_summary'] = self.own_summary.dict()
        return d


class ResearchLocalObservation(LocalObservation):
    """Local observation for a specific agent"""
    agent_id: str
    observation: ResearchObservation

    def dict(self, *args, **kwargs):
        """Custom dict method to handle nested observation"""
        d = super().dict(*args, **kwargs)
        if self.observation:
            d['observation'] = self.observation.dict()
        return d


class ResearchGlobalObservation(GlobalObservation):
    """Global observation containing all agent observations"""
    observations: Dict[str, ResearchLocalObservation]
    all_actions_this_round: Optional[Dict[str, Any]] = None
    final_all_summaries: Optional[List[Dict[str, Any]]] = None
    aggregator_notes: str = ""

    def dict(self, *args, **kwargs):
        """Custom dict method to handle nested observations"""
        d = super().dict(*args, **kwargs)
        if self.observations:
            d['observations'] = {
                k: v.dict() for k, v in self.observations.items()
            }
        return d

    @property
    def global_obs(self) -> Optional[Any]:
        """Get the global observation for all agents."""
        return {
            'all_actions_this_round': self.all_actions_this_round,
            'final_all_summaries': self.final_all_summaries,
            'aggregator_notes': self.aggregator_notes
        }
    
class ResearchActionSpace(ActionSpace):
    allowed_actions: List[Type[LocalAction]] = [ResearchAction]
    summary_model: Type[BaseModel]

    def get_action_schema(self) -> Dict[str, Any]:
        """
        Return JSON schema for whichever summary_model is used.
        """
        return self.summary_model.model_json_schema()

    def sample(self, agent_id: str) -> LocalAction:
        """
        For debugging or random testing; create a sample instance of the
        user-defined summary model.
        """
        return ResearchAction.sample(agent_id, self.summary_model)


class ResearchObservationSpace(ObservationSpace):
    """
    Observations revolve around local/global research data.
    """
    allowed_observations: List[Type[LocalObservation]] = [ResearchLocalObservation]


class ResearchMechanism(Mechanism):
    sequential: bool = Field(
        default=False,
        description="Whether the mechanism is sequential (one agent at a time)."
    )
    max_rounds: int = Field(default=3, description="Number of research steps allowed.")
    current_round: int = Field(default=0, description="Current step or round.")
    
    round_summaries: List[Dict[str, Any]] = Field(default_factory=list)
    last_step: Optional[EnvironmentStep] = Field(default=None, description="Last environment step")

    def step(self, action: Union[ResearchAction, ResearchGlobalAction, Dict[str, Any]]) -> Union[LocalEnvironmentStep, EnvironmentStep]:
        logger.debug(f"ResearchMechanism step: {action}")
        self.current_round += 1
        done = (self.current_round >= self.max_rounds)

        if self.sequential:
            if isinstance(action, dict):
                try:
                    action = ResearchAction.parse_obj(action)
                except Exception as e:
                    logger.error(f"Failed to parse dict into ResearchAction: {e}")
                    raise
            if not isinstance(action, ResearchAction):
                raise TypeError(f"Expected ResearchAction, got {type(action).__name__}")

            # Store in round_summaries
            round_actions = {action.agent_id: action.action.dict()}
            self.round_summaries.append(round_actions)

            # Build local observation
            obs = ResearchObservation(
                own_summary=action.action,
                aggregator_notes=f"Round {self.current_round}, single-agent action."
            )
            local_obs = ResearchLocalObservation(
                agent_id=action.agent_id,
                observation=obs
            )

            local_step = LocalEnvironmentStep(
                observation=local_obs,
                reward=0.0,
                done=done,
                info={
                    "round": self.current_round,
                    "note": "Sequential step",
                    "agent_rewards": {action.agent_id: 1.0}
                }
            )
            self.last_step = local_step
            return local_step

        else:
            # Batched mode: parse a GlobalAction with many local actions
            if isinstance(action, dict):
                try:
                    action = ResearchGlobalAction.parse_obj(action)
                except Exception as e:
                    logger.error(f"Failed to parse dict into ResearchGlobalAction: {e}")
                    raise
            if not isinstance(action, ResearchGlobalAction):
                raise TypeError(f"Expected ResearchGlobalAction, got {type(action).__name__}")

            # Convert each local action to dict for easy storing
            round_actions = {
                agent_id: local_action.action.dict()
                for agent_id, local_action in action.actions.items()
            }
            self.round_summaries.append(round_actions)

            # Build local observations for each agent
            local_observations: Dict[str, ResearchLocalObservation] = {}
            agent_rewards: Dict[str, float] = {}
            
            for agent_id, local_action in action.actions.items():
                local_obs = ResearchLocalObservation(
                    agent_id=agent_id,
                    observation=ResearchObservation(
                        own_summary=local_action.action,
                        aggregator_notes=f"Round {self.current_round}, agent {agent_id}"
                    )
                )
                local_observations[agent_id] = local_obs
                agent_rewards[agent_id] = 1.0

            # If final round, gather all data
            final_data = None
            aggregator_notes = f"End of round {self.current_round}."
            if done:
                final_data = []
                for r_i, entry in enumerate(self.round_summaries, start=1):
                    final_data.append({
                        "round_index": r_i,
                        "summaries": entry,
                    })
                aggregator_notes += " (Final) returning all collected summaries."

            global_obs = ResearchGlobalObservation(
                observations=local_observations,
                all_actions_this_round=round_actions,
                final_all_summaries=final_data,
                aggregator_notes=aggregator_notes
            )
            env_step = EnvironmentStep(
                global_observation=global_obs,
                done=done,
                info={
                    "round": self.current_round,
                    "agent_rewards": agent_rewards
                }
            )
            self.last_step = env_step
            return env_step

    def get_global_state(self) -> Dict[str, Any]:
        """
        Return the mechanism's overall state, e.g. how many rounds have been done,
        or any other state you'd like to expose to agents or orchestrators.
        """
        return {
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "round_summaries_count": len(self.round_summaries),
            "last_step": self.last_step.dict() if self.last_step else None
        }

    def reset(self) -> None:
        """
        Reset mechanism for a new run.
        """
        self.current_round = 0
        self.round_summaries.clear()
        self.last_step = None  # Reset last step
        logger.info("ResearchMechanism reset complete.")


class ResearchEnvironment(MultiAgentEnvironment):
    """
    Multi-agent environment that orchestrates a research session.
    It references a ResearchMechanism that collects agent actions.
    """
    name: str = Field(default="Research Environment", description="Name of the environment")
    action_space: ResearchActionSpace = Field(
        default_factory=lambda: ResearchActionSpace(summary_model=BaseModel),
        description="Defines the Pydantic model for agent's research summaries"
    )
    observation_space: ResearchObservationSpace = Field(
        default_factory=ResearchObservationSpace,
        description="Observation space"
    )
    mechanism: ResearchMechanism = Field(default_factory=ResearchMechanism)

    def __init__(
        self,
        summary_model: Type[BaseModel],
        **data
    ):
        """
        You can inject the user-defined summary_model (e.g. MarketResearch, AssetAnalysis, etc.)
        at construction time. Then the action space references that model for JSON schema.
        """
        action_space = ResearchActionSpace(summary_model=summary_model)
        mechanism = ResearchMechanism()
        super().__init__(
            action_space=action_space,
            mechanism=mechanism,
            **data
        )
        # Initialize our own state tracking
        self._global_state: Dict[str, Any] = {}

    def reset(self) -> GlobalObservation:
        """Override reset to handle our own state management"""
        self.current_step = 0
        self._global_state = {}
        self.history = EnvironmentHistory()
        self.mechanism.reset()
        return GlobalObservation(observations={})

    def get_global_state(self) -> Any:
        """Expose both mechanism state and our internal state."""
        mechanism_state = self.mechanism.get_global_state()
        return {
            **self._global_state,
            **mechanism_state,
            "current_step": self.current_step,
            "max_steps": self.max_steps
        }