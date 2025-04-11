from typing import Dict, List, Any, Type, Optional, Union, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import json
from market_agents.agents.market_agent import MarketAgent
from market_agents.agents.cognitive_steps import (
    ActionStep,
    CognitiveEpisode,
    PerceptionStep,
    ReflectionStep
)
from market_agents.environments.mechanisms.mcp_server import MCPServerActionSpace, MCPServerEnvironment
from minference.lite.models import (
    CallableTool, 
    StructuredTool, 
    CallableMCPTool,
    ResponseFormat,
    Entity
)

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add a handler if none exists
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class WorkflowStepIO(BaseModel):
    """Input/output values for workflow steps."""
    name: str = Field(..., description="Name of the input/output field")
    data: Union[Type[BaseModel], Dict[str, Any]] = Field(
        default_factory=dict,
        description="Schema (as BaseModel class) or actual data (as dict)"
    )

    def format_for_prompt(self) -> str:
        """Format the input data for use in prompts"""
        if isinstance(self.data, type) and issubclass(self.data, BaseModel):
            return f"{self.name}:\n{self.data.model_json_schema()}"
        return f"{self.name}:\n{json.dumps(self.data, indent=2)}"

    def get_formatted_data(self) -> Dict[str, Any]:
        """Get the data in a format suitable for workflow processing"""
        if isinstance(self.data, type) and issubclass(self.data, BaseModel):
            return {}
        return self.data

    model_config = {
        "arbitrary_types_allowed": True
    }

class CognitiveStepResult(BaseModel):
    """Base class for any cognitive step result"""
    step_type: str = Field(..., description="Type of cognitive step (perception, action, reflection, etc)")
    content: Any = Field(..., description="Result content from the step")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Step-specific metadata")

class EpisodeResult(BaseModel):
    """Result from a full cognitive episode with arbitrary steps"""
    steps: List[CognitiveStepResult] = Field(..., description="Results from each cognitive step")
    episode_metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Metadata about the episode execution"
    )

    @property
    def step_types(self) -> List[str]:
        """Get list of step types in this episode"""
        return [step.step_type for step in self.steps]
    
    def get_step_result(self, step_type: str) -> Optional[CognitiveStepResult]:
        """Get result for a specific step type"""
        for step in self.steps:
            if step.step_type == step_type:
                return step
        return None

class WorkflowStepResult(BaseModel):
    """Complete result from a workflow step execution"""
    step_id: str = Field(..., description="Identifier for the workflow step")
    status: Literal["completed", "failed"] = Field(..., description="Execution status")
    result: Union[EpisodeResult, CognitiveStepResult] = Field(
        ..., 
        description="The actual result, either from an episode or single step"
    )
    tool_results: List[Any] = Field(
        default_factory=list,
        description="Results from tool executions"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="General metadata about the step execution"
    )
    error: Optional[str] = Field(
        None,
        description="Error message if status is failed"
    )

    @property
    def is_episode(self) -> bool:
        """Check if this is an episode result"""
        return isinstance(self.result, EpisodeResult)

    @property
    def is_single_step(self) -> bool:
        """Check if this is a single step result"""
        return isinstance(self.result, CognitiveStepResult)
    
class WorkflowStep(Entity):
    """
    A step in a workflow that executes through MarketAgent's cognitive architecture.
    Can run as a single ActionStep or full cognitive episode.
    """
    name: str = Field(
        ...,
        description="Name identifier for this step"
    )
    environment_name: str = Field(
        ...,
        description="Name of the MCP server environment for this step"
    )
    tools: List[Union[CallableTool, StructuredTool, CallableMCPTool]] = Field(
        ..., 
        description="Tools to be executed in this step"
    )
    subtask: str = Field(
        ..., 
        description="Instruction to follow for this workflow step"
    )
    inputs: List[WorkflowStepIO] = Field(
        default_factory=list,
        description="Input schema definitions"
    )
    outputs: List[WorkflowStepIO] = Field(
        default_factory=list,
        description="Output schema definitions"
    )
    run_full_episode: bool = Field(
        default=False,
        description="Whether to run full cognitive episode (perception->action->reflection) or just action step"
    )
    sequential_tools: bool = Field(
        default=True,
        description="Whether tools should be executed in sequence through ActionStep's workflow mode"
    )

    async def execute(
        self,
        agent: "MarketAgent",
        inputs: Dict[str, Any],
        mcp_servers: Dict[str, MCPServerEnvironment],
        workflow_task: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute this workflow step using MarketAgent's cognitive architecture."""
        try:
            if self.environment_name not in mcp_servers:
                raise ValueError(f"MCP server environment '{self.environment_name}' not found")
            
            # Debug input state
            logger.debug(f"Executing step '{self.name}' with inputs: {inputs}")
            
            # Format the task with inputs
            formatted_task = workflow_task.format(**inputs) if workflow_task else ""
            logger.debug(f"Formatted workflow task: {formatted_task}")
            
            input_context = []
            for io in self.inputs:
                if io.name in inputs:
                    if isinstance(io.data, type) and issubclass(io.data, BaseModel):
                        validated_data = io.data.model_validate(inputs[io.name])
                        io.data = validated_data.model_dump()
                    else:
                        io.data = inputs[io.name]
                    input_context.append(io.format_for_prompt())
            
            formatted_context = "\n".join(input_context)
            logger.debug(f"Formatted context: {formatted_context}")
            
            # Format the subtask with inputs
            formatted_subtask = self.subtask.format(**inputs)
            logger.debug(f"Formatted subtask: {formatted_subtask}")
            
            combined_task = f"""
            {formatted_task}

            ## Current Step: {self.name}

            ## Context:
            {formatted_context}

            ## Sub-task 
            {formatted_subtask}
            """
                        
            agent.task = combined_task
            agent._refresh_prompts()

            # Debug environment state
            mcp_server = mcp_servers[self.environment_name]
            
            # Create a new action space with only the selected tools
            selected_action_space = MCPServerActionSpace(
                mechanism=mcp_server.mechanism,
                selected_tools=self.tools,
                workflow=self.sequential_tools and len(self.tools) > 1
            )
            
            # Create a temporary environment with restricted tools
            mcp_server.action_space = selected_action_space
            agent.chat_thread.tools = self.tools
            
            # Add temporary environment to agent
            agent.environments[self.environment_name] = mcp_server
            initial_history_len = len(mcp_server.mechanism.tool_history.get("default", []))

            
            # Create action step
            action_step = ActionStep(
                step_name=self.name,
                agent_id=agent.id,
                environment_name=self.environment_name,
                environment_info=inputs,
                action_space=self.tools[0] if not self.sequential_tools and self.tools else None
            )

            if not self.sequential_tools:
                agent.chat_thread.llm_config.response_format = ResponseFormat.auto_tools
                agent.chat_thread.workflow_step = None

            if self.run_full_episode:
                # Run full cognitive episode
                episode = CognitiveEpisode(
                    steps=[PerceptionStep, ActionStep, ReflectionStep],
                    environment_name=self.environment_name,
                    metadata={
                        "workflow_step": self.name,
                        "tools": [t.name for t in self.tools]
                    }
                )
                results = await agent.run_episode(episode=episode)
                
                print(f"Tool Execution History:\n {mcp_server.mechanism.tool_history.get("default", [])}")
                tool_results = mcp_server.mechanism.tool_history.get("default", [])[initial_history_len:]
                
                episode_result = EpisodeResult(
                    steps=[
                        CognitiveStepResult(
                            step_type="perception",
                            content=results[0]
                        ),
                        CognitiveStepResult(
                            step_type="action",
                            content=results[1]
                        ),
                        CognitiveStepResult(
                            step_type="reflection",
                            content=results[2]
                        )
                    ],
                    episode_metadata={
                        "workflow_step": self.name,
                        "tools": [t.name for t in self.tools]
                    }
                )

                return WorkflowStepResult(
                    step_id=self.name,
                    status="completed",
                    result=episode_result,
                    tool_results=tool_results,
                    metadata={
                        "agent_id": agent.id,
                        "environment": self.environment_name,
                        "tools_used": [t.name for t in self.tools],
                        "timestamp": datetime.utcnow().isoformat(),
                        "execution_type": "episode"
                    }
                )
            else:
                # Run just the action step
                action_result = await agent.run_step(step=action_step)
                tool_results = mcp_server.mechanism.tool_history.get("default", [])[initial_history_len:]

                return WorkflowStepResult(
                    step_id=self.name,
                    status="completed",
                    result=CognitiveStepResult(
                        step_type="action",
                        content=action_result,
                        metadata={
                            "workflow_step": self.name,
                            "tools": [t.name for t in self.tools]
                        }
                    ),
                    tool_results=tool_results,
                    metadata={
                        "agent_id": agent.id,
                        "environment": self.environment_name,
                        "tools_used": [t.name for t in self.tools],
                        "timestamp": datetime.utcnow().isoformat(),
                        "execution_type": "action_step"
                    }
                )

        except Exception as e:
            return WorkflowStepResult(
                step_id=self.name,
                status="failed",
                result=CognitiveStepResult(
                    step_type="action",
                    content=None,
                    metadata={"error": str(e)}
                ),
                error=str(e),
                metadata={
                    "agent_id": agent.id,
                    "environment": self.environment_name,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )

class WorkflowExecutionResult(BaseModel):
    """Result from a complete workflow execution"""
    workflow_id: str = Field(..., description="Identifier for the workflow")
    final_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Final state after all steps executed"
    )
    step_results: List[WorkflowStepResult] = Field(
        default_factory=list,
        description="Results from each workflow step"
    )
    tool_history: Dict[str, Dict[str, List[Dict[str, Any]]]] = Field(
        default_factory=dict,
        description="Tool execution history per environment and cohort"
    )

    @property
    def successful_steps(self) -> List[WorkflowStepResult]:
        """Get list of successfully completed steps"""
        return [step for step in self.step_results if step.status == "completed"]
    
    @property
    def failed_steps(self) -> List[WorkflowStepResult]:
        """Get list of failed steps"""
        return [step for step in self.step_results if step.status == "failed"]
    
    def get_step_result(self, step_id: str) -> Optional[WorkflowStepResult]:
        """Get result for a specific step"""
        for step in self.step_results:
            if step.step_id == step_id:
                return step
        return None

    def get_cohort_tool_history(self, environment: str, cohort: str = "default") -> List[Dict[str, Any]]:
        """Get tool history for a specific environment and cohort"""
        return self.tool_history.get(environment, {}).get(cohort, [])
        
class Workflow(Entity):
    """A workflow that orchestrates execution of steps across multiple environments."""
    name: str = Field(..., description="Name identifier for this workflow")
    task: str = Field(
        ..., 
        description="High-level task prompt for the entire workflow"
    )
    steps: List[WorkflowStep] = Field(..., description="Ordered sequence of workflow steps")
    mcp_servers: Dict[str, MCPServerEnvironment] = Field(
        ...,
        description="MCP environments for tool execution"
    )

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow"
    }
    @classmethod
    def create(
        cls,
        name: str,
        task: str,
        steps: List[WorkflowStep],
        mcp_servers: Dict[str, MCPServerEnvironment],
    ) -> 'Workflow':
        """Create a new workflow instance while preserving the MCP servers."""
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # Add a handler if none exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        # Create instance without validation
        instance = cls.model_construct(
            name=name,
            task=task,
            steps=steps,
            mcp_servers=mcp_servers
        )
        
        # Log the state
        logger.debug(f"Created workflow with environments: {list(instance.mcp_servers.keys())}")
        return instance
        
    async def execute(
        self,
        agent: "MarketAgent",
        initial_inputs: Dict[str, Any]
    ) -> WorkflowExecutionResult:
        """Execute the workflow across multiple environments."""
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        
        state = initial_inputs.copy()
        results: List[WorkflowStepResult] = []
        
        formatted_workflow_task = self.task.format(**initial_inputs) if self.task else None
            
        previous_result = None
        for i, step in enumerate(self.steps):
            logger.debug(f"Executing step '{step.name}' with inputs: {state}")
            
            # Add previous step result to inputs if available
            if i > 0 and previous_result is not None:
                state["previous_step_result"] = previous_result
            
            step_result = await step.execute(
                agent=agent,
                inputs=state,
                mcp_servers=self.mcp_servers,
                workflow_task=formatted_workflow_task
            )
            
            if step_result.status == "completed":
                # Handle both episode and single step results
                if step_result.is_episode:
                    action_step = step_result.result.get_step_result("action")
                    if action_step:
                        previous_result = action_step.content
                else:
                    # For single step results, get content directly
                    previous_result = step_result.result.content
                    
                results.append(step_result)
                logger.debug(f"Updated state after {step.name}: {state}")
            else:
                logger.error(f"Step '{step.name}' failed: {step_result.error}")
                results.append(step_result)

        return WorkflowExecutionResult(
            workflow_id=self.name,
            final_state=state,
            step_results=results,
            tool_history={
                env_name: env.mechanism.tool_history
                for env_name, env in self.mcp_servers.items()
            }
        )