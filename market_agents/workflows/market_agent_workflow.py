from typing import Dict, List, Any, Optional, Union, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import UUID

from market_agents.agents.market_agent import MarketAgent
from market_agents.agents.cognitive_steps import (
    ActionStep,
    CognitiveEpisode,
    PerceptionStep,
    ReflectionStep
)
from market_agents.environments.mechanisms.mcp_server import MCPServerEnvironment
from minference.lite.models import (
    CallableTool, 
    StructuredTool, 
    CallableMCPTool,
    Entity
)

class WorkflowStepIO(BaseModel):
    """Schema for workflow step input/output validation."""
    name: str = Field(..., description="Name of the input/output field")
    json_schema: Dict[str, Any] = Field(..., description="JSON schema for validation")
    description: str = Field(..., description="Description of this field")
    required: bool = Field(default=True, description="Whether this field is required")

class WorkflowStep(Entity):
    """
    A step in a workflow that executes through MarketAgent's cognitive architecture.
    Can run as a single ActionStep or full cognitive episode.
    """
    name: str = Field(..., description="Name identifier for this step")
    description: str = Field(..., description="Description of what this step does")
    
    tools: List[Union[CallableTool, StructuredTool, CallableMCPTool]] = Field(
        ..., 
        description="Tools to be executed in this step"
    )
    
    environment_name: str = Field(
        ...,
        description="Name of the MCP server environment for this step"
    )
    
    inputs: List[WorkflowStepIO] = Field(
        default_factory=list,
        description="Input schema definitions"
    )
    
    outputs: List[WorkflowStepIO] = Field(
        default_factory=list,
        description="Output schema definitions"
    )
    
    instruction_prompt: str = Field(
        ..., 
        description="Instruction to follow for this workflow"
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
        mcp_servers: Dict[str, MCPServerEnvironment]
    ) -> Dict[str, Any]:
        """Execute this workflow step using MarketAgent's cognitive architecture."""
        
        try:
            
            # Get the specific MCP server for this step
            if self.environment_name not in mcp_servers:
                raise ValueError(f"MCP server environment '{self.environment_name}' not found")
            
            mcp_server = mcp_servers[self.environment_name]
            agent.environments[self.environment_name] = mcp_server
            
            # This ensures tools are properly configured in the expected format
            action_space = mcp_server.action_space
            
            # Filter the action space to only include the tools we want
            tool_names = [tool.name for tool in self.tools]
            filtered_tools = [
                tool for tool in action_space.allowed_actions 
                if tool.name in tool_names
            ]

            # Create action step with the proper action space
            action_step = ActionStep(
                step_name=self.name,
                agent_id=agent.id,
                environment_name=self.environment_name,
                environment_info=inputs,
                action_space=filtered_tools if self.sequential_tools else filtered_tools[0]
            )

            if self.run_full_episode:
                # Run full cognitive episode
                episode = CognitiveEpisode(
                    steps=[PerceptionStep, ActionStep, ReflectionStep],
                    environment_name=self.environment_name,
                    metadata={
                        "workflow_step": self.name,
                        "tools": tool_names
                    }
                )
                results = await agent.run_episode(episode=episode)
                
                return {
                    "step_id": self.name,
                    "perception": results[0],
                    "action": results[1],
                    "reflection": results[2],
                    "status": "completed",
                    "metadata": {
                        "agent_id": agent.id,
                        "environment": self.environment_name,
                        "tools_used": tool_names,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
            else:
                # Run just the action step
                result = await agent.run_step(step=action_step)
                
                return {
                    "step_id": self.name,
                    "result": result,
                    "status": "completed", 
                    "metadata": {
                        "agent_id": agent.id,
                        "environment": self.environment_name,
                        "tools_used": tool_names,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }

        except Exception as e:
            return {
                "step_id": self.name,
                "result": None,
                "status": "failed",
                "error": str(e),
                "metadata": {
                    "agent_id": agent.id,
                    "environment": self.environment_name,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        
class Workflow(Entity):
    """
    A workflow that orchestrates execution of steps across multiple environments.
    """
    name: str = Field(..., description="Name identifier for this workflow")
    description: str = Field(..., description="Description of what this workflow does")
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
        description: str,
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

        logger.debug("Starting create_research_workflow")
        
        # Create instance without validation
        instance = cls.model_construct(
            name=name,
            description=description,
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
    ) -> Dict[str, Any]:
        """Execute the workflow across multiple environments."""
        import logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        
        state = initial_inputs.copy()
        results = []
        
        # Debug the mcp_servers at workflow execution
        logger.debug(f"Workflow execute starting")
        logger.debug(f"mcp_servers type: {type(self.mcp_servers)}")
        logger.debug(f"mcp_servers keys: {list(self.mcp_servers.keys())}")
        logger.debug(f"mcp_servers contents: {self.mcp_servers}")
        
        # Execute steps sequentially
        for step in self.steps:
            logger.debug(f"Executing step '{step.name}' with environment '{step.environment_name}'")
            logger.debug(f"Available environments: {list(self.mcp_servers.keys())}")
            
            if step.environment_name not in self.mcp_servers:
                logger.error(f"Environment '{step.environment_name}' not found in available environments: {list(self.mcp_servers.keys())}")
                raise ValueError(f"Environment '{step.environment_name}' not found")
            
            # Verify the environment before passing it
            env = self.mcp_servers[step.environment_name]
            logger.debug(f"Found environment for step: {env.name}")
            
            step_result = await step.execute(
                agent=agent,
                inputs=state,
                mcp_servers=self.mcp_servers
            )
            
            # Update state with step results
            if step_result["status"] == "completed":
                if "result" in step_result:
                    state.update(step_result["result"])
                else:
                    # Handle full episode results
                    state.update({
                        "perception": step_result["perception"],
                        "action": step_result["action"],
                        "reflection": step_result["reflection"]
                    })
            
            results.append(step_result)

        return {
            "workflow_id": self.name,
            "final_state": state,
            "step_results": results
        }