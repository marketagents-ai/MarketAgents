from typing import List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict, Field
import logging

from market_agents.agents.base_agent.agent import Agent
from market_agents.orchestrators.logger_utils import log_action, log_section, log_task_assignment
from market_agents.orchestrators.meta_orchestrator import MetaOrchestrator
from market_agents.orchestrators.config import OrchestratorConfig
from minference.lite.models import ResponseFormat, StructuredTool

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logging.getLogger("EntityRegistry").setLevel(logging.CRITICAL)

class MarketAgentTeam(BaseModel):
    """A team of market agents that work together."""
    
    name: str = Field(..., description="Name of the team")
    agents: List[Agent] = Field(..., description="List of agents in the team")
    manager: Optional[Agent] = Field(None, description="Team manager agent")
    mode: str = Field(
        "hierarchical", 
        description="Team operation mode: 'hierarchical' (manager delegates), 'collaborative' (group discussion), or 'autonomous' (independent work)"
    )
    use_group_chat: bool = Field(
        None,  
        description="Whether to use group chat environment"
    )
    shared_context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Shared context between team members"
    )
    environments: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Environment configurations for the team"
    )
    meta_orchestrator: Optional[MetaOrchestrator] = None

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow"
    }

    def __init__(self, **data):
        super().__init__(**data)
        
        # Set default use_group_chat based on mode if not provided
        if self.use_group_chat is None:
            self.use_group_chat = self.mode in ["hierarchical", "collaborative"]
        
        # Initialize environment configurations
        env_configs = {}
        env_order = []

        # Configure group chat if enabled
        if self.use_group_chat:
            group_chat_config = {
                "name": f"{self.name}_chat",
                "mechanism": "group_chat",
                "initial_topic": "",
                "sequential": False,
                "form_cohorts": False,
                "sub_rounds": 1,
                "group_size": len(self.agents),
                "api_url": "http://localhost:8002"
            }
            env_configs[group_chat_config["name"]] = group_chat_config
            env_order.append(group_chat_config["name"])
            logger.info(f"Group chat enabled for {self.mode} mode")
        else:
            logger.info("Group chat disabled")

        # Add additional environments if provided
        for environment in self.environments:
            if "name" not in environment or "mechanism" not in environment:
                raise ValueError(f"Environment config must include 'name' and 'mechanism' fields: {environment}")
            
            env_name = environment["name"]
            env_configs[env_name] = environment
            env_order.append(env_name)
            logger.info(f"Added environment {env_name} with mechanism {environment['mechanism']}")

        # Initialize meta orchestrator
        config = OrchestratorConfig(
            name=self.name,
            num_agents=len(self.agents),
            max_rounds=1,
            environment_order=env_order,
            environment_configs=env_configs,
            tool_mode=True
        )

        self.meta_orchestrator = MetaOrchestrator(
            config=config,
            agents=[self.manager] + self.agents if self.manager else self.agents
        )

    async def delegate_tasks(self, task: str) -> Dict[str, Any]:
        """Manager delegates subtasks to team members using structured tool."""
        if not self.manager:
            raise ValueError("Hierarchical mode requires a team manager")

        # Log task start
        log_section(logger, f"Starting Task Delegation - {self.name}")

        # Create delegation tool
        delegation_tool = StructuredTool(
            name="delegate_subtask",
            description="Delegate subtasks to team members. Each team member should receive a relevant subtask.",
            json_schema={
                "type": "object",
                "properties": {
                    "delegations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "agent_id": {
                                    "type": "string",
                                    "description": "ID of the agent to assign the subtask",
                                    "enum": [agent.id for agent in self.agents]
                                },
                                "subtask": {
                                    "type": "string",
                                    "description": "The subtask to be assigned"
                                },
                                "expected_output": {
                                    "type": "string",
                                    "description": "Expected output from this subtask"
                                }
                            },
                            "required": ["agent_id", "subtask", "expected_output"]
                        },
                        "minItems": len(self.agents),
                        "maxItems": len(self.agents),
                        "description": "List of subtask delegations to team members"
                    }
                },
                "required": ["delegations"]
            }
        )

        # Add tool to manager's toolkit
        self.manager.tools = [delegation_tool]
        self.manager.chat_thread.forced_output = delegation_tool
        
        # Log available agents
        agent_info = [f"- {agent.id}: {agent.role}" for agent in self.agents]

        prompt = f"""
        Break down and delegate this task to ALL team members based on their specialties:
        {task}

        Available team members:
        {agent_info}

        Requirements:
        1. EVERY team member must receive a relevant subtask
        2. Subtasks should align with each member's expertise
        3. Each delegation must include clear expected outputs

        Use the delegate_subtask function to assign work to ALL team members.
        """

        logger.info("Manager analyzing and delegating tasks...")
        self.manager.chat_thread.new_message = prompt
        
        delegations = await self.manager.execute()
        log_action(logger, "Manager", delegations, "Task Delegation")
        return delegations

    async def run_environment_orchestration(
        self, 
        task: str, 
        mode: str,
        delegations: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run orchestration across all environments based on mode.
        
        Args:
            task: Main task to be executed
            mode: Team operation mode ('hierarchical', 'collaborative', 'autonomous')
            delegations: Optional task delegations for hierarchical mode
        """
        # Update agent tasks based on mode
        for agent in self.agents:
            if mode == "hierarchical" and delegations:
                # Assign delegated subtasks in hierarchical mode
                subtask = next(
                    (d["subtask"] for d in delegations["delegations"] if d.get("agent_id") == agent.id),
                    None
                )
                if subtask:
                    agent.task = f"The manager has assigned you following sub-task. If tools are available use tools as per your role & sub-task\n{subtask}"
                    log_task_assignment(logger, agent.id, subtask, is_subtask=True)
            else:
                # Same task for all agents in collaborative/autonomous modes
                agent.task = task
                log_task_assignment(logger, agent.id, task, is_subtask=False)

        # Update environment configs with task context
        for env_name, env_config in self.meta_orchestrator.config.environment_configs.items():
            if env_name == "group_chat":
                env_config.initial_topic = task
                #env_config.sequential = (mode == "hierarchical")
            else:
                env_config.task_prompt = task
            logger.info(f"Updated {env_name} environment with task context")

        # Run orchestration across all environments
        log_section(logger, f"Starting {mode.capitalize()} Execution - {self.name}")
        await self.meta_orchestrator.run_orchestration()

        # Synthesize results if manager exists
        if self.manager and mode != "autonomous":
            log_section(logger, "Manager Synthesis")
            synthesis_prompt = f"""
            Synthesize the team's work across all environments into final conclusions.
            
            Original Task:
            {task}
            
            Mode: {mode}
            
            Group Chat Discussion:
            {self.meta_orchestrator.environment_orchestrators.get('group_chat').environment.mechanism.round_messages if 'group_chat' in self.meta_orchestrator.environment_orchestrators else 'No group chat'}
            
            Other Environment Results:
            {[
                f"{env_name}: {env.environment.get_global_state()}"
                for env_name, env in self.meta_orchestrator.environment_orchestrators.items()
                if env_name != 'group_chat'
            ]}
            """
            self.manager.chat_thread.new_message = synthesis_prompt
            self.manager.chat_thread.tools = []
            self.manager.chat_thread.forced_output = None
            self.manager.chat_thread.llm_config.response_format = ResponseFormat.text

            final_result = await self.manager.execute()
            log_action(logger, "Manager", final_result, "Final Synthesis")
            return final_result
        else:
            # Return results from all environments
            return {
                "group_chat": self.meta_orchestrator.environment_orchestrators.get('group_chat').environment.mechanism.round_messages if 'group_chat' in self.meta_orchestrator.environment_orchestrators else None,
                "environment_results": {
                    env_name: env.environment.get_global_state()
                    for env_name, env in self.meta_orchestrator.environment_orchestrators.items()
                    if env_name != 'group_chat'
                }
            }

    async def hierarchical_execution(self, task: str) -> Dict[str, Any]:
        """Team manager delegates subtasks and coordinates work."""
        delegations = await self.delegate_tasks(task)
        return await self.run_environment_orchestration(task, "hierarchical", delegations)

    async def collaborative_execution(self, task: str) -> Dict[str, Any]:
        """Team members work together through environments."""
        return await self.run_environment_orchestration(task, "collaborative")

    async def autonomous_execution(self, task: str) -> Dict[str, Any]:
        """Agents work independently through environments."""
        return await self.run_environment_orchestration(task, "autonomous")

    async def execute(self, task: str) -> Dict[str, Any]:
        """Execute team task based on mode."""
        if self.mode == "hierarchical":
            return await self.hierarchical_execution(task)
        elif self.mode == "collaborative":
            return await self.collaborative_execution(task)
        elif self.mode == "autonomous":
            return await self.autonomous_execution(task)
        else:
            raise ValueError(f"Invalid team mode: {self.mode}")