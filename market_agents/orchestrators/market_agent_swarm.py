import math
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from market_agents.agents.base_agent.agent import Agent
from market_agents.orchestrators.meta_orchestrator import MetaOrchestrator
from market_agents.orchestrators.config import OrchestratorConfig

class MarketAgentSwarm(BaseModel):
    """A swarm of market agents for high-scale, multi-cohort orchestration and workflow execution."""
    name: str = Field(
        ...,
        description="Name of the swarm"
    )
    agents: List[Agent] = Field(
        ...,
        description="Flat list of all agents in the swarm"
    )
    form_cohorts: bool = Field(
        True,
        description="Whether to split agents into cohorts"
    )
    cohort_size: Optional[int] = Field(
        None,
        description="Number of agents per cohort; used if use_cohorts=True"
    )
    num_cohorts: Optional[int] = Field(
        None,
        description="Number of cohorts; computed if cohort_size provided"
    )
    environments: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Additional environment configurations provided by user"
    )
    max_rounds: int = Field(
        1,
        description="Maximum number of rounds per cohort"
    )
    tool_mode: bool = Field(
        True, 
        description="Whether to enable structured tool mode in orchestrator"
    )
    meta_orchestrator: Optional[MetaOrchestrator] = None

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow"
    }

    def __init__(self, **data: Any):
        super().__init__(**data)

        # Calculate cohort organization
        self.cohorts = self._form_cohorts()

        # Update environment configs with cohort information
        for env in self.environments:
            env["form_cohorts"] = self.form_cohorts
            if self.form_cohorts:
                env["group_size"] = self.cohort_size

        # Initialize MetaOrchestrator with cohort information
        orchestrator_config = OrchestratorConfig(
            name=self.name,
            num_agents=len(self.agents),
            max_rounds=self.max_rounds,
            environment_order=[env["name"] for env in self.environments],
            environment_configs={env["name"]: env for env in self.environments},
            tool_mode=self.tool_mode,
            form_cohorts=self.form_cohorts,
            cohort_size=self.cohort_size if self.form_cohorts else None
        )
        self.meta_orchestrator = MetaOrchestrator(
            config=orchestrator_config,
            agents=self.agents,
            cohorts=self.cohorts if self.form_cohorts else None
        )

    def _form_cohorts(self) -> List[List[Agent]]:
        """Form cohorts based on swarm configuration."""
        if self.form_cohorts:
            if self.cohort_size:
                self.num_cohorts = math.ceil(len(self.agents) / self.cohort_size)
            elif self.num_cohorts:
                self.cohort_size = math.ceil(len(self.agents) / self.num_cohorts)
            else:
                self.num_cohorts = 1
                self.cohort_size = len(self.agents)

            # Build cohorts
            return [
                self.agents[i * self.cohort_size : min((i + 1) * self.cohort_size, len(self.agents))]
                for i in range(self.num_cohorts)
            ]
        else:
            self.num_cohorts = 1
            self.cohort_size = len(self.agents)
            return [self.agents]
        
    async def get_agent_actions(self) -> List[Dict[str, Any]]:
        """Get the final results from all agents in JSONL format."""
        results = []
        for agent in self.agents:
            if hasattr(agent, 'last_action') and agent.last_action:
                jsonl_entry = {
                    "agent_id": str(agent.id),
                    "last_action": agent.last_action if isinstance(agent.last_action, dict) else json.loads(agent.last_action)
                }
                results.append(jsonl_entry)
        return results

    async def deploy_swarm(self, task: str) -> Dict[str, Any]:
        """Run orchestration for the entire swarm across cohorts."""
        # Assign the task to every agent
        for agent in self.agents:
            agent.task = task

        # Execute the orchestration
        await self.meta_orchestrator.run_orchestration()

        # Get agent results
        results = await self.get_agent_actions()
        
        return results