# research.py

from datetime import datetime
from importlib import import_module
import json
import random
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict

from market_agents.environments.config import EnvironmentConfig
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

class ResearchEnvironmentConfig(EnvironmentConfig):
    """Configuration for research environment orchestration"""
    name: str = Field(
        default="research",
        description="Name of the research environment"
    )
    api_url: str = Field(
        default="http://localhost:8003",
        description="API endpoint for research environment"
    )
    sub_rounds: int = Field(
        default=2,
        description="Number of sub-rounds within each main round"
    )
    initial_topic: str = Field(
        default="Market Analysis",
        description="Initial research topic"
    )
    group_size: int = Field(
        default=4,
        description="Number of agents in research group"
    )
    schema_model: str = Field(
        default="LiteraryAnalysis",
        description="Name of Pydantic model defining research output schema"
    )
    form_cohorts: bool = Field(
        default=False,
        description="Whether to organize agents into cohorts"
    )

    model_config = SettingsConfigDict(
        extra="allow"
    )

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
    current_topic: str = ""
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
    current_topic: str = ""
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
    """Mechanism that manages research rounds and agent summaries."""
    sequential: bool = Field(
        default=False,
        description="Whether the mechanism is sequential (one agent at a time)."
    )
    current_round: int = Field(
        default=0, 
        description="Current step or round."
    )
    max_rounds: int = Field(
        default=0,
        description="Max steps or rounds"
    )
    current_topic: str = Field(
        default="", 
        description="Current research topic"
    )
    round_summaries: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="History of all round summaries, organized by cohort when form_cohorts=True"
    )
    last_step: Optional[EnvironmentStep] = Field(
        default=None, 
        description="Last environment step"
    )
    summary_model: Type[BaseModel] = Field(
        default=BaseModel, 
        description="Model for validating research summaries"
    )
    form_cohorts: bool = Field(
        default=False,
        description="Whether to organize agents into cohorts"
    )
    group_size: Optional[int] = Field(
        default=None,
        description="Size of research cohorts when form_cohorts is True"
    )
    cohorts: Dict[str, List[Any]] = Field(
        default_factory=dict,
        description="Mapping of cohort IDs to lists of agents"
    )

    def __init__(
        self,
        initial_topic: str = "",
        summary_model: Type[BaseModel] = BaseModel,
        form_cohorts: bool = False,
        group_size: int = 4,
        max_rounds: int = 2,
        **kwargs
    ):
        """Initialize the research mechanism with proper Pydantic model validation."""
        super().__init__(
            current_topic=initial_topic,
            summary_model=summary_model,
            form_cohorts=form_cohorts,
            group_size=group_size,
            max_rounds=max_rounds,
            **kwargs
        )
        logger.info(f"Initialized ResearchMechanism with topic: {initial_topic}, form_cohorts: {form_cohorts}")

    async def form_agent_cohorts(self, agents: List[Any]) -> None:
        """Form research cohorts based on group size from config."""
        if not self.form_cohorts or not self.group_size:
            return

        self.cohorts.clear()
        
        current_cohort = []
        cohort_count = 1

        for agent in agents:
            current_cohort.append(agent)
            if len(current_cohort) >= self.group_size:
                self.cohorts[f"research_cohort_{cohort_count}"] = current_cohort
                current_cohort = []
                cohort_count += 1

        if current_cohort:
            self.cohorts[f"research_cohort_{cohort_count}"] = current_cohort

        logger.info(f"Formed {len(self.cohorts)} research cohorts")
        for cohort_id, cohort_agents in self.cohorts.items():
            logger.info(f"{cohort_id}: {[agent.id for agent in cohort_agents]}")

    def step(
        self, 
        action: Union[LocalAction, GlobalAction],
        cohort_id: Optional[str] = None
    ) -> Union[LocalEnvironmentStep, EnvironmentStep]:
        """Execute one step of the research process."""
        # Check if we're done before incrementing
        done = self.current_round >= self.max_rounds

        # Use provided cohort_id or default
        effective_cohort = cohort_id if cohort_id else "default"
        
        # Initialize cohort's round_summaries if needed
        if effective_cohort not in self.round_summaries:
            self.round_summaries[effective_cohort] = []
            
        # Add new round if needed
        while len(self.round_summaries[effective_cohort]) <= self.current_round:
            self.round_summaries[effective_cohort].append({})

        # Handle single agent action
        if isinstance(action, LocalAction):
            action_dict = action.action
            try:
                validated_content = self.summary_model.model_validate(action_dict)
            except Exception as e:
                logger.error(f"Failed to validate action content against {self.summary_model.__name__}: {e}")
                raise

            # Store the validated content
            self.round_summaries[effective_cohort][self.current_round][action.agent_id] = validated_content.model_dump()

            # Create and return local step
            obs = ResearchObservation(
                current_topic=self.current_topic,
                own_summary=validated_content,
                aggregator_notes=f"Round {self.current_round}"
            )
            local_obs = ResearchLocalObservation(
                agent_id=action.agent_id,
                observation=obs
            )
            local_step = LocalEnvironmentStep(
                observation=local_obs,
                done=done,
                info={
                    "round": self.current_round,
                    "cohort_id": effective_cohort
                }
            )
            
            # Increment round counter after processing
            self.current_round += 1
            return local_step

        # Handle global actions
        else:
            research_actions = {}
            observations = {}

            for agent_id, local_action in action.actions.items():
                action_dict = local_action.action
                try:
                    validated_content = self.summary_model.model_validate(action_dict)
                    research_actions[agent_id] = validated_content.model_dump()

                    # Store the validated content
                    self.round_summaries[effective_cohort][self.current_round][agent_id] = validated_content.model_dump()

                    # Create observation for this agent
                    obs = ResearchObservation(
                        current_topic=self.current_topic,
                        own_summary=validated_content,
                        aggregator_notes=f"Round {self.current_round}"
                    )
                    observations[agent_id] = ResearchLocalObservation(
                        agent_id=agent_id,
                        observation=obs
                    )

                except Exception as e:
                    logger.error(f"Failed to validate action content for agent {agent_id}: {e}")
                    raise

            # Create global step
            global_obs = ResearchGlobalObservation(
                observations=observations,
                all_actions_this_round=research_actions,
                current_topic=self.current_topic,
                aggregator_notes=f"Round {self.current_round} complete"
            )
            
            global_step = EnvironmentStep(
                global_observation=global_obs,
                done=done,
                info={
                    "round": self.current_round,
                    "cohort_id": effective_cohort
                }
            )
            self.last_step = global_step
            
            # Increment round counter after processing
            self.current_round += 1
            return global_step

    def get_global_state(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get global state, optionally filtered for agent's cohort"""
        state = {
            "current_round": self.current_round,
            "current_topic": self.current_topic,
            "form_cohorts": self.form_cohorts,
            "max_rounds": self.max_rounds
        }

        if self.form_cohorts:
            if agent_id:
                # Get agent's specific cohort
                cohort_id = next(
                    (cid for cid, agents in self.cohorts.items() 
                    if any(a.id == agent_id for a in agents)),
                    None
                )
                if cohort_id:
                    state.update({
                        "round_summaries": self.round_summaries.get(cohort_id, []),
                        "round_summaries_count": len(self.round_summaries.get(cohort_id, [])),
                        "cohort_id": cohort_id,
                        "cohort_agents": [a.id for a in self.cohorts[cohort_id]]
                    })
            else:
                # Return all cohorts' data
                state.update({
                    "cohorts": {
                        cid: [a.id for a in agents] 
                        for cid, agents in self.cohorts.items()
                    },
                    "round_summaries": self.round_summaries,
                    "round_summaries_count": {
                        cid: len(summaries) 
                        for cid, summaries in self.round_summaries.items()
                    }
                })
        else:
            # Not using cohorts, return all summaries from default cohort
            state.update({
                "round_summaries": self.round_summaries.get("default", []),
                "round_summaries_count": len(self.round_summaries.get("default", []))
            })

        if self.last_step:
            state["last_step"] = self.last_step.dict()

        return state

    def reset(self) -> None:
        """Reset mechanism for a new run."""
        self.current_round = 0
        self.round_summaries.clear()
        self.last_step = None
        self.cohorts.clear()
        logger.info("ResearchMechanism reset complete.")

class ResearchEnvironment(MultiAgentEnvironment):
    """
    Multi-agent environment that orchestrates a research session.
    It references a ResearchMechanism that collects agent actions.
    """
    name: str = Field(default="research", description="Name of the environment")
    action_space: ResearchActionSpace = Field(
        default_factory=lambda: ResearchActionSpace(summary_model=BaseModel),
        description="Defines the Pydantic model for agent's research summaries"
    )
    observation_space: ResearchObservationSpace = Field(
        default_factory=ResearchObservationSpace,
        description="Observation space"
    )
    mechanism: ResearchMechanism = Field(default_factory=ResearchMechanism)
    initial_topic: Optional[str] = Field(default=None, description="Initial research topic")

    def __init__(self, **config):
        """Initialize environment with config parameters."""
        try:
            # Parse and validate config
            env_config = ResearchEnvironmentConfig(**config)
            
            # Get the schema model
            summary_model = self._get_schema_model(env_config.schema_model)
            
            # Initialize action space with the schema model
            action_space = ResearchActionSpace(summary_model=summary_model)
            
            # Initialize mechanism with relevant config
            mechanism = ResearchMechanism(
                initial_topic=env_config.initial_topic,
                summary_model=summary_model,
                form_cohorts=env_config.form_cohorts,
                group_size=env_config.group_size,
                max_rounds=env_config.sub_rounds
            )

            # Initialize parent class with processed config
            super().__init__(
                name=env_config.name,
                action_space=action_space,
                observation_space=ResearchObservationSpace(),
                mechanism=mechanism,
                initial_topic=env_config.initial_topic
            )
            self._global_state: Dict[str, Any] = {}
            
            # Form cohorts during initialization if enabled
            if env_config.form_cohorts and hasattr(self, 'agents'):
                self.mechanism.form_agent_cohorts(self.agents)
                
        except Exception as e:
            raise ValueError(f"Failed to initialize ResearchEnvironment: {e}")

    def _get_schema_model(self, schema_name: str) -> Type[BaseModel]:
        """Dynamically import and return the schema model class."""
        try:
            schemas_module = import_module('market_agents.orchestrators.research_schemas')
            
            if not hasattr(schemas_module, schema_name):
                raise ValueError(f"Schema model '{schema_name}' not found in research_schemas")
                
            model_class = getattr(schemas_module, schema_name)
            
            if not issubclass(model_class, BaseModel):
                raise ValueError(f"Schema model {schema_name} must be a Pydantic model")
                
            return model_class
        except ImportError as e:
            raise ValueError(f"Could not import research_schemas module: {e}")
        except Exception as e:
            raise ValueError(f"Could not load schema model '{schema_name}': {e}")

    def get_global_state(self, agent_id: str = None) -> Dict[str, Any]:
        """Return the environment's global state with filtered mechanism state."""
        # Get the mechanism's state with agent_id
        mechanism_state = self.mechanism.get_global_state(agent_id) if agent_id else self.mechanism.get_global_state()
        
        # Filter to include only the last round's summaries
        if "round_summaries" in mechanism_state and mechanism_state["round_summaries"]:
            mechanism_state["round_summaries"] = mechanism_state["round_summaries"][-1:]

        return {
            **mechanism_state,
            "current_step": self.mechanism.current_round,
            "max_steps": self.mechanism.max_rounds
        }

    def reset(self) -> GlobalObservation:
        """Override reset to handle our own state management"""
        self.current_step = 0
        self._global_state = {}
        self.history = EnvironmentHistory()
        self.mechanism.reset()
        
        if self.initial_topic:
            self.mechanism.current_topic = self.initial_topic
            
        return GlobalObservation(observations={})