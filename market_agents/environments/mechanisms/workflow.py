from datetime import datetime
from typing import Dict, Any, List, Union, Optional, Type
from enum import Enum
from pydantic import BaseModel, Field

from market_agents.environments.environment import (
    Mechanism,
    LocalAction,
    GlobalAction,
    LocalObservation,
    GlobalObservation,
    EnvironmentStep,
    LocalEnvironmentStep,
    ActionSpace,
    ObservationSpace
)

from market_agents.agents.cognitive_steps import (
    CognitiveStep,
    ActionStep,
    CognitiveEpisode
)

from minference.lite.models import (
    CallableTool,
    StructuredTool,
    ResponseFormat
)

import logging
logger = logging.getLogger(__name__)

class WorkflowStepType(str, Enum):
    """Types of workflow steps."""
    SEQUENTIAL = "sequential"  # Steps executed in order
    PARALLEL = "parallel"    # Steps executed simultaneously
    CONDITIONAL = "conditional"  # Steps with branching logic
    ITERATIVE = "iterative"  # Steps that may repeat

class WorkflowStatus(str, Enum):
    """Status of workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

class WorkflowStepResult(BaseModel):
    """Result from a workflow step execution."""
    step_id: str
    status: str
    result: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class WorkflowAction(LocalAction):
    """Action for workflow step execution."""
    step_id: str
    action: Any
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def sample(cls, agent_id: str) -> 'WorkflowAction':
        return cls(
            agent_id=agent_id,
            step_id="sample_step",
            action={},
            metadata={}
        )

class WorkflowObservation(BaseModel):
    """Observation from workflow execution."""
    current_step: str
    step_result: Optional[WorkflowStepResult] = None
    workflow_state: Dict[str, Any] = Field(default_factory=dict)
    global_context: Dict[str, Any] = Field(default_factory=dict)

class WorkflowLocalObservation(LocalObservation):
    """Local observation for a specific agent."""
    observation: WorkflowObservation

class WorkflowGlobalObservation(GlobalObservation):
    """Global observation containing all agent observations."""
    observations: Dict[str, WorkflowLocalObservation]
    workflow_state: Dict[str, Any] = Field(default_factory=dict)
    step_results: Dict[str, WorkflowStepResult] = Field(default_factory=dict)

class WorkflowState(BaseModel):
    """State maintained across workflow execution."""
    workflow_id: str
    current_step: str
    step_results: Dict[str, WorkflowStepResult] = Field(default_factory=dict)
    global_context: Dict[str, Any] = Field(default_factory=dict)
    error_history: List[Dict[str, Any]] = Field(default_factory=list)
    retry_counts: Dict[str, int] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    start_time: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    def update_state(self, **kwargs):
        """Update workflow state with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_updated = datetime.utcnow()

class WorkflowMechanism(Mechanism):
    """
    Mechanism that manages workflow execution.
    
    Features:
    - Support for sequential and parallel execution
    - State management and history tracking
    - Error handling and retry logic
    - Cohort support for multi-agent workflows
    - Tool validation and execution
    - Event streaming
    """
    sequential: bool = Field(
        default=True,
        description="Whether steps are executed sequentially"
    )
    current_round: int = Field(
        default=0,
        description="Current execution round"
    )
    max_rounds: int = Field(
        default=1,
        description="Maximum number of rounds"
    )
    workflow_type: WorkflowStepType = Field(
        default=WorkflowStepType.SEQUENTIAL,
        description="Type of workflow execution"
    )
    form_cohorts: bool = Field(
        default=False,
        description="Whether to organize agents into cohorts"
    )
    group_size: Optional[int] = Field(
        default=None,
        description="Size of workflow cohorts when form_cohorts is True"
    )
    cohorts: Dict[str, List[Any]] = Field(
        default_factory=dict,
        description="Mapping of cohort IDs to lists of agents"
    )
    state: WorkflowState = Field(
        ...,
        description="Current workflow state"
    )
    available_tools: Dict[str, Union[CallableTool, StructuredTool]] = Field(
        default_factory=dict,
        description="Available tools for workflow steps"
    )

    class Config:
        arbitrary_types_allowed = True

    async def form_agent_cohorts(self, agents: List[Any]) -> None:
        """Form workflow cohorts based on group size."""
        if not self.form_cohorts or not self.group_size:
            return

        self.cohorts.clear()
        current_cohort = []
        cohort_count = 1

        for agent in agents:
            current_cohort.append(agent)
            if len(current_cohort) >= self.group_size:
                cohort_id = f"workflow_cohort_{cohort_count}"
                self.cohorts[cohort_id] = current_cohort
                current_cohort = []
                cohort_count += 1

        if current_cohort:
            cohort_id = f"workflow_cohort_{cohort_count}"
            self.cohorts[cohort_id] = current_cohort

        logger.info(f"Formed {len(self.cohorts)} workflow cohorts")
        for cohort_id, cohort_agents in self.cohorts.items():
            logger.info(f"{cohort_id}: {[agent.id for agent in cohort_agents]}")

    def step(
        self,
        action: Union[LocalAction, GlobalAction],
        cohort_id: Optional[str] = None
    ) -> Union[LocalEnvironmentStep, EnvironmentStep]:
        """Execute a workflow step."""
        # Check if we're done
        done = self.current_round >= self.max_rounds

        # Use provided cohort_id or default
        effective_cohort = cohort_id if cohort_id else "default"

        # Handle single agent action
        if isinstance(action, LocalAction):
            # Process the action and update state
            step_result = self._process_action(action, effective_cohort)
            
            # Create observation
            obs = WorkflowObservation(
                current_step=self.state.current_step,
                step_result=step_result,
                workflow_state=self.state.model_dump(),
                global_context=self.state.global_context
            )
            
            local_obs = WorkflowLocalObservation(
                agent_id=action.agent_id,
                observation=obs
            )

            # Create and return local step
            return LocalEnvironmentStep(
                observation=local_obs,
                done=done,
                info={
                    "round": self.current_round,
                    "cohort_id": effective_cohort,
                    "workflow_state": self.state.model_dump()
                }
            )

        # Handle global actions
        else:
            observations = {}
            step_results = {}

            # Process each agent's action
            for agent_id, local_action in action.actions.items():
                step_result = self._process_action(local_action, effective_cohort)
                step_results[agent_id] = step_result

                # Create observation for this agent
                obs = WorkflowObservation(
                    current_step=self.state.current_step,
                    step_result=step_result,
                    workflow_state=self.state.model_dump(),
                    global_context=self.state.global_context
                )
                observations[agent_id] = WorkflowLocalObservation(
                    agent_id=agent_id,
                    observation=obs
                )

            # Create global observation
            global_obs = WorkflowGlobalObservation(
                observations=observations,
                workflow_state=self.state.model_dump(),
                step_results=step_results
            )

            # Create and return global step
            return EnvironmentStep(
                global_observation=global_obs,
                done=done,
                info={
                    "round": self.current_round,
                    "cohort_id": effective_cohort,
                    "workflow_state": self.state.model_dump()
                }
            )

    def _process_action(
        self,
        action: LocalAction,
        cohort_id: str
    ) -> WorkflowStepResult:
        """Process a single action and update workflow state."""
        try:
            # Extract action content
            if isinstance(action, WorkflowAction):
                step_id = action.step_id
                content = action.action
                metadata = action.metadata
            else:
                step_id = self.state.current_step
                content = action.action
                metadata = {}

            # Create step result
            step_result = WorkflowStepResult(
                step_id=step_id,
                status="completed",
                result=content,
                metadata={
                    "agent_id": action.agent_id,
                    "cohort_id": cohort_id,
                    "round": self.current_round,
                    **metadata
                }
            )

            # Update workflow state
            self.state.step_results[step_id] = step_result
            self.state.update_state(
                current_step=step_id,
                last_updated=datetime.utcnow()
            )

            return step_result

        except Exception as e:
            logger.error(f"Error processing action: {e}")
            return WorkflowStepResult(
                step_id=self.state.current_step,
                status="failed",
                result=None,
                error=str(e)
            )

    def get_global_state(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get global state, filtered by agent's cohort if specified."""
        state = {
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "workflow_type": self.workflow_type,
            "workflow_state": self.state.model_dump()
        }

        if self.form_cohorts:
            if agent_id:
                # Find agent's cohort
                cohort_id = next(
                    (cid for cid, agents in self.cohorts.items() 
                    if any(a.id == agent_id for a in agents)),
                    None
                )
                if cohort_id:
                    state.update({
                        "cohort_id": cohort_id,
                        "cohort_agents": [a.id for a in self.cohorts[cohort_id]]
                    })
            else:
                # Return all cohorts' information
                state.update({
                    "cohorts": {
                        cid: [a.id for a in agents] 
                        for cid, agents in self.cohorts.items()
                    }
                })

        return state

    def reset(self) -> None:
        """Reset the workflow state."""
        self.current_round = 0
        self.state = WorkflowState(
            workflow_id=f"workflow_{datetime.utcnow().timestamp()}",
            current_step=""
        )
        self.cohorts.clear()