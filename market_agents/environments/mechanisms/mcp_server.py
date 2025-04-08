from datetime import datetime
import sys
import asyncio
import subprocess
import logging
from typing import Dict, Any, List, Union, Optional
from minference.lite.models import CallableMCPTool, CallableTool, StructuredTool
from market_agents.environments.config import EnvironmentConfig
from mcp.client.stdio import stdio_client
from mcp import StdioServerParameters
from mcp import ClientSession

import sys

from pydantic import Field, BaseModel

from market_agents.environments.environment import (
    EnvironmentHistory,
    MultiAgentEnvironment, 
    Mechanism,
    LocalAction,
    LocalObservation,
    GlobalAction,
    GlobalObservation,
    LocalEnvironmentStep,
    EnvironmentStep,
    ActionSpace,
    ObservationSpace
)
from minference.caregistry import CallableRegistry

logger = logging.getLogger(__name__)
CallableRegistry._logger = logger

class MCPServerEnvironmentConfig(EnvironmentConfig):
    """Configuration for MCP server environment"""
    name: str = Field(
        ...,
        description="Domain-specific name for this instance (e.g., mcp_finance)"
    )
    mechanism: str = Field(
        default="mcp_server",
        description="Type of mechanism (always mcp_server for this class)"
    )
    mcp_server_module: str = Field(
        ...,
        description="Module path to the MCP server"
    )
    mcp_server_class: str = Field(
        default="mcp",
        description="Variable name of MCP server instance"
    )
    sub_rounds: int = Field(
        default=1,
        description="Number of sub-rounds per main round"
    )
    form_cohorts: bool = Field(
        default=False,
        description="Whether to organize agents into cohorts"
    )
    group_size: int = Field(
        default=4,
        description="Size of agent groups when using cohorts"
    )
    task_prompt: str = Field(
        default="",
        description="Initial task prompt"
    )

    model_config = {
        "extra": "allow"
    }

class MCPServerResult(BaseModel):
    """Structure for a single MCP server tool result"""
    tool_name: str
    result: Any
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class MCPServerLocalObservation(LocalObservation):
    """Local observation for a specific agent"""
    agent_id: str
    observation: Dict[str, Any]
    status: str = "pending"
    tool_results: Optional[List[MCPServerResult]] = None

    def dict(self, *args, **kwargs):
        """Custom dict method to handle nested observation"""
        d = super().dict(*args, **kwargs)
        if self.observation:
            d['observation'] = self.observation
        return d

class MCPServerGlobalObservation(GlobalObservation):
    """Global observation containing all agent observations"""
    observations: Dict[str, MCPServerLocalObservation]

class MCPToolAction(LocalAction):
    """Action for invoking an MCP server tool"""
    tool_name: str = Field(..., description="Name of the tool to invoke")
    tool_args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")

    @classmethod
    def sample(cls, agent_id: str) -> 'MCPToolAction':
        """Sample a random tool action (not implemented)"""
        return cls(agent_id=agent_id, tool_name="sample_tool", tool_args={})

class MCPServerMechanism(Mechanism):
    """Mechanism that manages MCP server tool interactions"""
    sequential: bool = Field(
        default=False,
        description="Whether the mechanism is sequential (one agent at a time)"
    )
    current_round: int = Field(
        default=0,
        description="Current interaction round"
    )
    max_rounds: int = Field(
        default=0,
        description="Max steps or rounds"
    )
    tool_history: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=lambda: {"default": []},
        description="History of tool invocations, organized by cohort when form_cohorts=True"
    )
    available_tools: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Available tools from the MCP server"
    )
    form_cohorts: bool = Field(
        default=False,
        description="Whether to organize agents into cohorts"
    )
    group_size: Optional[int] = Field(
        default=None,
        description="Size of agent cohorts when form_cohorts is True"
    )
    cohorts: Dict[str, List[Any]] = Field(
        default_factory=dict,
        description="Mapping of cohort IDs to lists of agents"
    )
    task_prompt: Optional[str] = Field(
        default=None,
        description="Initial task prompt"
    )

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow"
    }

    def __init__(
        self,
        server_path: str = None,
        form_cohorts: bool = False,
        group_size: int = 4,
        **data
    ):
        """Initialize with both MCP server and cohort support"""
        super().__init__(**data)
        
        # Initialize cohort-related attributes
        self.form_cohorts = form_cohorts
        self.group_size = group_size if form_cohorts else None
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize session-related attributes
        self._active_session = None
        self._session = None
        self._read_write = None
        self.client_initialized = False
        
        # Handle MCP server initialization
        if "mcp_server" in data:
            self.mcp_server = data["mcp_server"]
            self.server_process = None
            self.is_external_server = True
        elif server_path:
            self.server_path = server_path
            self.server_process = None
            self.is_external_server = False
            self._start_server_process()
            self._initialize_client_connection()
        else:
            raise ValueError("Either mcp_server or server_path must be provided")
        
        self.available_tools = {}

    async def form_agent_cohorts(self, agents: List[Any]) -> None:
        """Form agent cohorts based on group size from config."""
        if not self.form_cohorts or not self.group_size:
            return

        self.cohorts.clear()
        self.tool_history.clear()
        
        current_cohort = []
        cohort_count = 1

        for agent in agents:
            current_cohort.append(agent)
            if len(current_cohort) >= self.group_size:
                cohort_id = f"mcp_cohort_{cohort_count}"
                self.cohorts[cohort_id] = current_cohort
                self.tool_history[cohort_id] = []
                current_cohort = []
                cohort_count += 1

        if current_cohort:
            cohort_id = f"mcp_cohort_{cohort_count}"
            self.cohorts[cohort_id] = current_cohort
            self.tool_history[cohort_id] = []

        self.logger.info(f"Formed {len(self.cohorts)} MCP server cohorts")
        for cohort_id, cohort_agents in self.cohorts.items():
            self.logger.info(f"{cohort_id}: {[agent.id for agent in cohort_agents]}")
    
    def _start_server_process(self):
        """Start the MCP server as a subprocess"""
        try:
            # Check if we should use mcp run or direct python execution
            if self.server_path.endswith('.py'):
                # Use direct Python execution
                self.server_process = subprocess.Popen(
                    [sys.executable, self.server_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                print(f"Started MCP server process with PID: {self.server_process.pid}")
            else:
                # Use mcp run command
                self.server_process = subprocess.Popen(
                    ["mcp", "run", self.server_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                print(f"Started MCP server with mcp run, PID: {self.server_process.pid}")
                
            # Give the server a moment to start up
            import time
            time.sleep(2)
        except Exception as e:
            print(f"Error starting MCP server: {str(e)}")
            raise

    def _initialize_client_connection(self):
        """Initialize client connection to the MCP server"""
        
        try:
            # Create server parameters for the client
            self.server_params = StdioServerParameters(
                command=sys.executable,
                args=[self.server_path],
                env=None
            )
            print(f"Initialized client connection parameters for server: {self.server_path}")
            
            # Set the mcp_server attribute to the server parameters
            self.mcp_server = self.server_params
            print("Client connection parameters initialized")
            
        except Exception as e:
            print(f"Error initializing client connection: {str(e)}")
            raise

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any], chat_thread_id: Optional[str] = None) -> Any:
        """Execute a tool using a fresh MCP client session"""
        max_retries = 3
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                async with asyncio.timeout(30):
                    async with stdio_client(self.server_params) as (read, write):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            result = await session.call_tool(tool_name, arguments=arguments)
                            
                            # Convert result if needed
                            if hasattr(result, 'model_dump'):
                                result = result.model_dump()
                            elif hasattr(result, 'dict'):
                                result = result.dict()
                            elif hasattr(result, '__dict__'):
                                result = result.__dict__

                            logger.info(f"tool name: {tool_name}\ntool result:\n{result}")

                            # Record tool execution history
                            self.record_tool_history(
                                tool_name=tool_name,
                                arguments=arguments,
                                result=result,
                                chat_thread_id=chat_thread_id
                            )
                            
                            return result

            except (asyncio.TimeoutError, asyncio.CancelledError) as e:
                logger.error(f"Tool execution timed out or was cancelled: {str(e)}")
                last_error = e
                if isinstance(e, asyncio.CancelledError):
                    break
            except Exception as e:
                logger.error(f"Error executing tool {tool_name}: {str(e)}")
                if "unhandled errors in a TaskGroup" in str(e):
                    # Still record the history even for TaskGroup errors
                    try:
                        self.record_tool_history(
                            tool_name=tool_name,
                            arguments=arguments,
                            result=result,
                            chat_thread_id=chat_thread_id
                        )
                    except Exception as record_error:
                        logger.error(f"Failed to record tool history: {str(record_error)}")
                    return result
                last_error = e


                retry_count += 1
                if retry_count < max_retries:
                    await asyncio.sleep(1)
        
        # If we've exhausted retries or got cancelled, raise the last error
        if isinstance(last_error, asyncio.CancelledError):
            raise last_error
        raise last_error or Exception(f"Failed to execute tool {tool_name} after {max_retries} retries")

    def record_tool_history(
        self, 
        tool_name: str,
        arguments: Dict[str, Any],
        result: Any,
        chat_thread_id: Optional[str] = None
    ) -> None:
        """Record tool execution history in appropriate cohort."""
        logger.info(f"Recording tool history for {tool_name}")
        logger.info(f"Current tool_history state: {self.tool_history}")
        
        # Default values
        history_key = "default"
        agent_id = chat_thread_id or "system"

        # Only try to find matching agent if we're using cohorts
        if self.form_cohorts and chat_thread_id:
            for cid, agents in self.cohorts.items():
                for agent in agents:
                    if (hasattr(agent, 'chat_thread') and 
                        agent.chat_thread and 
                        str(agent.chat_thread.id) == str(chat_thread_id)):
                        agent_id = agent.id
                        history_key = cid
                        break
                if agent_id != "system":
                    break

        # Initialize history list if needed
        if history_key not in self.tool_history:
            logger.info("Initializing default history list")
            self.tool_history[history_key] = []

        # Create and record the history entry
        history_entry = {
            "agent_id": agent_id,
            "tool_name": tool_name,
            "arguments": arguments,
            "result": result,
            "round": self.current_round,
            "timestamp": datetime.now().isoformat()
        }

        self.tool_history[history_key].append(history_entry)
        logger.info(f"Added entry to history. New size: {len(self.tool_history[history_key])}")

    async def initialize(self):
        """Initialize the mechanism by extracting available tools"""
        try:
            logger.info("Connecting to MCP Server...")
            async with stdio_client(self.mcp_server) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    logger.info("Connected to MCP Server")
                    
                    tools_result = await session.list_tools()
                    
                    if hasattr(tools_result, 'tools'):
                        self.available_tools = {}
                        logger.info("Found tools:")
                        for tool_info in tools_result.tools:
                            self.available_tools[tool_info.name] = {
                                "name": tool_info.name,
                                "description": tool_info.description,
                                "input_schema": tool_info.inputSchema
                            }
                            logger.debug(f"  - {tool_info.name}")
                        logger.info(f"Successfully loaded {len(self.available_tools)} tools")
                    else:
                        logger.warning("No tools attribute found in result")
                        self.available_tools = {}
                        
        except Exception as e:
            logger.error(f"Error initializing mechanism: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.available_tools = {}
            
    def step(
        self,
        action: Union[GlobalAction, LocalAction],
        cohort_id: Optional[str] = None
    ) -> Union[LocalEnvironmentStep, EnvironmentStep]:
        """Process actions with cohort support"""
        # Use provided cohort_id or default
        effective_cohort = cohort_id if cohort_id else "default"
        
        # Initialize cohort's tool history if needed
        if effective_cohort not in self.tool_history:
            self.tool_history[effective_cohort] = []

        self.current_round += 1
        done = (self.current_round >= self.max_rounds)

        if isinstance(action, GlobalAction):
            observations = {}
            
            for agent_id, agent_action in action.actions.items():
                obs_data = {
                    "action": agent_action.model_dump() if hasattr(agent_action, 'model_dump') else str(agent_action),
                    "round": self.current_round,
                    "status": "success"
                }
                
                # Create the local observation
                observations[agent_id] = MCPServerLocalObservation(
                    agent_id=agent_id,
                    observation=obs_data,
                    status=obs_data["status"]
                )
            
            # Create global observation
            global_obs = MCPServerGlobalObservation(observations=observations)
            
            # Return environment step with required info field
            return EnvironmentStep(
                global_action=action,
                global_observation=global_obs,
                done=done,
                info={  # Add the required info field
                    "round": self.current_round,
                    "max_rounds": self.max_rounds,
                    "tool_history": self.tool_history,
                    "available_tools": list(self.available_tools.keys())
                }
            )
        
        elif isinstance(action, LocalAction):
            # Handle single agent action
            agent_id = action.agent_id
            obs_data = {
                "action": action.model_dump() if hasattr(action, 'model_dump') else str(action),
                "round": self.current_round,
                "status": "success"
            }
            
            # Create the local observation
            local_obs = MCPServerLocalObservation(
                agent_id=agent_id,
                observation=obs_data,
                status=obs_data["status"]
            )
            
            # Return local environment step with required info field
            return LocalEnvironmentStep(
                observation=local_obs,
                done=done,
                info={  # Add the required info field
                    "round": self.current_round,
                    "max_rounds": self.max_rounds,
                    "tool_history": self.tool_history,
                    "available_tools": list(self.available_tools.keys())
                }
            )
        
        else:
            # Handle string actions or other types
            return LocalEnvironmentStep(
                observation=MCPServerLocalObservation(
                    agent_id="system",
                    observation={"action": str(action), "round": self.current_round},
                    status="success"
                ),
                done=done,
                info={  # Add the required info field
                    "round": self.current_round,
                    "max_rounds": self.max_rounds,
                    "tool_history": self.tool_history,
                    "available_tools": list(self.available_tools.keys())
                }
            )
    
    def get_global_state(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Return the environment's global state from mechanism"""
        state = {
            "current_step": self.current_round,
            "max_steps": self.max_rounds,
            "task_prompt": self.task_prompt,
            "available_tools": self.available_tools
        }

        if self.form_cohorts:
            # Add cohort-specific information
            if agent_id:
                cohort_id = next(
                    (cid for cid, agents in self.cohorts.items() 
                    if any(a.id == agent_id for a in agents)),
                    None
                )
                if cohort_id:
                    state.update({
                        "tool_history": self.tool_history.get(cohort_id, []),
                        "cohort_id": cohort_id,
                        "cohort_agents": [a.id for a in self.cohorts[cohort_id]]
                    })
            else:
                state.update({
                    "cohorts": {cid: [a.id for a in agents] for cid, agents in self.cohorts.items()},
                    "tool_history": self.tool_history
                })
        else:
            state["tool_history"] = self.tool_history.get("default", [])

        return state

class MCPServerActionSpace(ActionSpace):
    """Action space that handles MCP server tool invocations"""
    mechanism: MCPServerMechanism = Field(
        ..., 
        description="Mechanism that handles MCP server operations"
    )
    selected_tools: Optional[List[Union[CallableTool, StructuredTool, CallableMCPTool]]] = Field(
        default=None,
        description="Specific tools this action space should provide access to"
    )
    workflow: bool = Field(
        default=False,
        description="Whether each tool should be executed sequentially or auto"
    )
    
    def __init__(self, mechanism: MCPServerMechanism, selected_tools: Optional[List] = None, **data):
        logger.info("Initializing MCPServerActionSpace...")
        
        data.update({
            "mechanism": mechanism,
            "selected_tools": selected_tools
        })
        super().__init__(**data)
        
        self.allowed_actions = []

        if selected_tools is not None:
            tool_names = [t.name for t in selected_tools]
            logger.info(f"Validating selected tools: {tool_names}")

            for tool in selected_tools:
                if isinstance(tool, CallableMCPTool):
                    if tool.name in mechanism.available_tools:
                        tool.mcp_mechanism = mechanism
                        self.allowed_actions.append(tool)
                        logger.debug(f"Added MCP tool: {tool.name}")
                    else:
                        logger.warning(f"Skipping MCP tool {tool.name} - not available in this server")
                else:
                    self.allowed_actions.append(tool)
                    logger.debug(f"Added non-MCP tool: {tool.name}")
        else:    
            logger.info(f"Available tools in mechanism: {list(mechanism.available_tools.keys())}")
            self._create_tools_from_server()

        logger.info(f"Total allowed_actions: {len(self.allowed_actions)}")
        
    # Create a tool for each available MCP server tool
    def _create_tools_from_server(self):
            """Create tools from server's complete tool set"""
            for tool_name, tool_info in self.mechanism.available_tools.items():
                try:
                    if "input_schema" in tool_info and tool_info["input_schema"] is not None:
                        mcp_tool = CallableMCPTool.from_callable(
                            name=tool_name,
                            description=tool_info.get("description"),
                            input_schema=tool_info.get("input_schema")
                        )
                        mcp_tool.mcp_mechanism = self.mechanism
                        self.allowed_actions.append(mcp_tool)
                        logger.debug(f"Created tool: {tool_name}")
                except Exception as e:
                    logger.error(f"Error creating tool {tool_name}: {str(e)}")
    
    def get_action_schema(self):
        """Return JSON schema for all available tools"""
        schemas = {}
        for tool in self.allowed_actions:
            schemas[tool.name] = tool.json_schema()
        return schemas

class MCPServerEnvironment(MultiAgentEnvironment):
    """Environment that manages MCP server operations"""
    name: str = Field(
        default="MCP Server Environment",
        description="Name of the environment"
    )
    mechanism: MCPServerMechanism = Field(
        ...,
        description="Mechanism that handles MCP server operations"
    )
    action_space: Optional[MCPServerActionSpace] = Field(
        default=None,
        description="Action space for MCP server tools"
    )
    observation_space: ObservationSpace = Field(
        default_factory=ObservationSpace,
        description="Observation space for MCP server"
    )

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow"
    }

    def __init__(self, **config):
        """Initialize environment with config parameters."""
        try:
            # Parse and validate config
            env_config = MCPServerEnvironmentConfig(**config)
            
            # Import the MCP server module
            import importlib
            try:
                spec = importlib.util.find_spec(env_config.mcp_server_module)
                if spec is None:
                    raise ImportError(f"Could not find module {env_config.mcp_server_module}")
                server_path = spec.origin
                if not server_path:
                    raise ValueError(f"Could not determine file path for module {env_config.mcp_server_module}")
            except Exception as e:
                raise ValueError(f"Error resolving server path: {e}")

            # Initialize mechanism with config parameters
            mechanism = MCPServerMechanism(
                server_path=server_path,
                form_cohorts=env_config.form_cohorts,
                group_size=env_config.group_size,
                server_class=env_config.mcp_server_class,
                task_prompt=env_config.task_prompt
            )

            # Create a base config dict for parent initialization
            base_config = {
                "name": env_config.name,
                "mechanism": mechanism,
                "observation_space": ObservationSpace()            }

            # Initialize parent class
            super().__init__(**base_config)
                
        except Exception as e:
            raise ValueError(f"Failed to initialize MCPServerEnvironment: {e}")

    async def initialize(self):
        """Initialize the environment by setting up mechanism and action space"""
        logger.info("Initializing MCPServerEnvironment...")
        
        # Initialize mechanism first to get available tools
        logger.info("Initializing mechanism...")
        await self.mechanism.initialize()
        logger.info(f"Mechanism initialized with {len(self.mechanism.available_tools)} tools")
        
        # Create action space with initialized mechanism
        logger.info("Creating action space...")
        self.action_space = MCPServerActionSpace(mechanism=self.mechanism)
        logger.info(f"Action space created with {len(self.action_space.allowed_actions)} tools")
        
        # Form cohorts if enabled
        if self.mechanism.form_cohorts and hasattr(self, 'agents'):
            await self.mechanism.form_agent_cohorts(self.agents)
            
        logger.info("MCPServerEnvironment initialization complete")

    def get_global_state(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Return the environment's global state from mechanism"""
        return self.mechanism.get_global_state(agent_id)

    def reset(self) -> GlobalObservation:
        """Reset environment state"""
        self.current_step = 0
        self.history = EnvironmentHistory()
        self.mechanism.reset()
        
        return MCPServerGlobalObservation(observations={})