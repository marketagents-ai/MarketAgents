from typing import Type, List, Union, Dict, Any, Optional
from pydantic import Field, validator
import logging

from market_agents.agents.base_agent.agent import Agent
from market_agents.agents.cognitive_steps import (
    CognitiveEpisode,
    CognitiveStep,
    PerceptionStep,
    ActionStep,
    ReflectionStep,
    TerminalToolInvoked
)
from market_agents.agents.market_agent_prompter import MarketAgentPromptManager
from market_agents.agents.personas.persona import Persona
from market_agents.economics.econ_agent import EconomicAgent
from minference.lite.inference import InferenceOrchestrator
from minference.lite.models import LLMConfig, CallableTool, StructuredTool
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.knowledge_base_agent import KnowledgeBaseAgent
from market_agents.memory.memory import LongTermMemory, MemoryObject, ShortTermMemory
from market_agents.environments.environment import LocalObservation, MultiAgentEnvironment
from market_agents.agents.protocols.protocol import Protocol
from market_agents.verbal_rl.rl_agent import VerbalRLAgent
from market_agents.verbal_rl.rl_models import BaseRewardFunction

logger = logging.getLogger(__name__)

class MarketAgent(Agent):
    """Market agent with cognitive capabilities and memory management."""

    role: str = Field(
        default="AI Assistant",
        description="Professional role in market context (e.g., 'Research Analyst')"
    )   
    short_term_memory: ShortTermMemory = Field(
        default=None,
        description="Short-term memory storage for recent cognitive stps"
    )
    long_term_memory: LongTermMemory = Field(
        default=None,
        description="Long-term memory storage for episodic memories"
    )
    environments: Dict[str, MultiAgentEnvironment] = Field(
        default_factory=dict,
        description="Dictionary of environments the agent can interact with"
    )
    last_perception: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Most recent perception output from cognitive episode"
    )
    last_action: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Most recent action taken by the agent"
    )
    last_observation: Optional[LocalObservation] = Field(
        default_factory=dict,
        description="Most recent observation received from environment"
    )
    episode_steps: List[MemoryObject] = Field(
        default_factory=list,
        description="List of memory objects from current cognitive episode"
    )
    current_episode: Optional[CognitiveEpisode] = Field(
        default=None,
        description="Currently executing cognitive episode"
    )
    protocol: Optional[Type[Protocol]] = Field(
        default=None,
        description="Communication protocol used by the agent"
    )
    address: str = Field(
        default="",
        description="Agent's endpoint for communication, task assignment etc"
    )
    knowledge_agent: Optional[KnowledgeBaseAgent] = Field(
        default=None,
        description="Knowledge base agent for accessing external information"
    )
    economic_agent: Optional[EconomicAgent] = Field(
        default=None,
        description="Economic agent component for market interactions"
    )
    rl_agent: VerbalRLAgent = Field(
        default_factory=VerbalRLAgent,
        description="Verbal RL subsystem for learning & adaption"
    )
    prompt_manager: MarketAgentPromptManager = Field(
        default_factory=MarketAgentPromptManager,
        description="Manager for handling agent prompts and templates"
    )

    @classmethod
    async def create(
        cls,
        name: str,
        persona: Optional[Persona] = None,
        task: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        tools: Optional[List[Union[CallableTool, StructuredTool]]] = None,
        ai_utils: Optional[InferenceOrchestrator] = None,
        storage_utils: Optional[AgentStorageAPIUtils] = None,
        environments: Optional[Dict[str, MultiAgentEnvironment]] = None,
        protocol: Optional[Type[Protocol]] = None,
        econ_agent: Optional[EconomicAgent] = None,
        knowledge_agent: Optional[KnowledgeBaseAgent] = None,
        reward_function: Optional[BaseRewardFunction] = None,
        
    ) -> 'MarketAgent':
        
        storage_utils = storage_utils or AgentStorageAPIUtils()
        
        agent = cls(
            name=name,
            persona=persona,
            task=task,
            llm_orchestrator=ai_utils or InferenceOrchestrator(),
            llm_config=llm_config or LLMConfig(),
            tools=tools or [],
            environments=environments or {},
            protocol=protocol,
            rl_agent=VerbalRLAgent(reward_function=reward_function) if reward_function else VerbalRLAgent()
        )

        agent.short_term_memory = ShortTermMemory(
            agent_id=agent.id,
            agent_storage_utils=storage_utils,
            default_top_k=storage_utils.config.stm_top_k
        )
        await agent.short_term_memory.initialize()
        
        agent.long_term_memory = LongTermMemory(
            agent_id=agent.id,
            agent_storage_utils=storage_utils,
            default_top_k=storage_utils.config.ltm_top_k
        )
        await agent.long_term_memory.initialize()

        agent.role = persona.role if persona else "AI Agent"
        agent.address = f"agent/{str(agent.id)}"
        agent.economic_agent = econ_agent
        agent.knowledge_agent = knowledge_agent

        if agent.economic_agent:
            agent.economic_agent.id = agent.id
        if agent.knowledge_agent:
            agent.knowledge_agent.id = agent.id

        return agent
        
    async def run_step(
        self,
        step: Optional[Union[CognitiveStep, Type[CognitiveStep]]] = None,
        environment_name: Optional[str] = None,
        *,
        max_steps: int | None = 1,
        terminal_tools: Optional[List[str]] = None,
        **kwargs
    ) -> Union[str, Dict[str, Any], List[Union[str, Dict[str, Any]]]]:
        """
        Execute a single cognitive step.
        
        Args:
            step: CognitiveStep instance or class (defaults to ActionStep)
            environment_name: Optional environment context
            max_steps: Optional[int] – run the step this many times (None or 1 ⇒ single execution).
            terminal_tools: Optional[List[str]] – List of tool names that should terminate execution
            **kwargs: Additional parameters for step initialization
        
        Returns:
            For single step (max_steps=1): The step result
            For multiple steps: List of results including any terminal tool result
            
        Raises:
            TerminalToolInvoked: When a terminal tool is used (in single step mode)
            ValueError: If environment not found or other validation errors
        """
        # Handle iterative execution when max_steps > 1
        if max_steps is not None and max_steps > 1:
            outputs: List[Union[str, Dict[str, Any]]] = []
            try:
                for loop_idx in range(max_steps):
                    logger.info(f"[MarketAgent] run_step loop iteration {loop_idx + 1}/{max_steps}")
                    try:
                        out = await self.run_step(
                            step=step,
                            environment_name=environment_name,
                            max_steps=1,
                            terminal_tools=terminal_tools,
                            **kwargs
                        )
                        outputs.append(out)
                    except TerminalToolInvoked as e:
                        outputs.append(e.payload)
                        logger.info(f"[MarketAgent] Terminal tool '{e.tool_name}' invoked - stopping iteration")
                        break
                return outputs
            except Exception as e:
                logger.error(f"[MarketAgent] Error in run_step: {str(e)}")
                raise

        if environment_name and environment_name not in self.environments:
            raise ValueError(f"Environment {environment_name} not found")
        
        if step and not isinstance(step, type) and hasattr(step, 'environment_name') and not environment_name:
            environment_name = step.environment_name
            logger.info(f"Using environment_name from step: {environment_name}")
        
        env_name = environment_name or next(iter(self.environments.keys()))
        environment = self.environments[env_name]
                
        if step is None:
            step = ActionStep(
                agent_id=self.id,
                environment_name=env_name,
                environment_info=environment.get_global_state(agent_id=self.id),
                terminal_tools=terminal_tools,
                **kwargs
            )
        elif isinstance(step, type):
            step = step(
                agent_id=self.id,
                environment_name=env_name,
                environment_info=environment.get_global_state(agent_id=self.id),
                terminal_tools=terminal_tools,
                **kwargs
            )
        else:
            step.environment_name = env_name
            step.environment_info = environment.get_global_state(agent_id=self.id)
            step.agent_id = self.id
            if terminal_tools is not None:
                step.terminal_tools = terminal_tools
        
        logger.info(f"Executing cognitive step: {step.step_name}")
        result = await step.execute(self)
        return result

    async def run_episode(
        self,
        episode: Optional[CognitiveEpisode] = None,
        environment_name: Optional[str] = None,
        *,
        max_steps: int | None = 1,
        terminal_tools: Optional[List[str]] = None,
        **kwargs
    ) -> Union[List[Union[str, Dict[str, Any]]], List[List[Union[str, Dict[str, Any]]]]]:
        """
        Run a complete cognitive episode.
        
        Args:
            episode: CognitiveEpisode instance (defaults to Perception->Action->Reflection)
            environment_name: Optional environment to use
            max_steps: Optional[int] – run the episode this many times (None or 1 ⇒ single execution).
            terminal_tools: Optional[List[str]] – List of tool names that should terminate execution
            **kwargs: Additional parameters passed to each step
        
        Returns:
            For single episode: List of step results
            For multiple episodes: List of episode results
            
        Raises:
            ValueError: If environment not found or other validation errors
        """
        # Handle iterative execution when max_steps > 1
        if max_steps is not None and max_steps > 1:
            episodes_out: List[List[Union[str, Dict[str, Any]]]] = []
            for loop_idx in range(max_steps):
                logger.info(f"[MarketAgent] run_episode loop iteration {loop_idx + 1}/{max_steps}")
                try:
                    ep_out = await self.run_episode(
                        episode=episode,
                        environment_name=environment_name,
                        max_steps=1,
                        terminal_tools=terminal_tools,
                        **kwargs
                    )
                    episodes_out.append(ep_out)
                except TerminalToolInvoked as e:
                    # Add the terminal tool result and stop iterations
                    if isinstance(ep_out, list):
                        ep_out.append(e.payload)
                    episodes_out.append(ep_out)
                    logger.info(f"[MarketAgent] Terminal tool '{e.tool_name}' invoked - stopping episodes")
                    break
            return episodes_out

        self._refresh_prompts()
        
        if episode is None:
            episode = CognitiveEpisode(
                steps=[PerceptionStep, ActionStep, ReflectionStep],
                environment_name=environment_name or next(iter(self.environments.keys()))
            )
        elif environment_name:
            episode.environment_name = environment_name

        results = []
        try:
            for step_class in episode.steps:
                result = await self.run_step(
                    step=step_class,
                    environment_name=episode.environment_name,
                    terminal_tools=terminal_tools,
                    **kwargs
                )
                results.append(result)
        except TerminalToolInvoked as e:
            results.append(e.payload)
            raise
                
        return results