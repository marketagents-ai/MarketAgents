from typing import Type, List, Union, Dict, Any, Optional
from pydantic import Field, validator
import logging

from market_agents.agents.base_agent.agent import Agent
from market_agents.agents.cognitive_steps import (
    CognitiveEpisode,
    CognitiveStep,
    PerceptionStep,
    ActionStep,
    ReflectionStep
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
        storage_utils: AgentStorageAPIUtils,
        name: str,
        persona: Optional[Persona] = None,
        task: Optional[str] = None,
        llm_config: Optional[LLMConfig] = None,
        tools: Optional[List[Union[CallableTool, StructuredTool]]] = None, 
        ai_utils: Optional[InferenceOrchestrator] = None,
        environments: Optional[Dict[str, MultiAgentEnvironment]] = None,
        protocol: Optional[Type[Protocol]] = None,
        econ_agent: Optional[EconomicAgent] = None,
        knowledge_agent: Optional[KnowledgeBaseAgent] = None,
        reward_function: Optional[BaseRewardFunction] = None,
    ) -> 'MarketAgent':
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
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """
        Execute a single cognitive step.
        
        Args:
            step: CognitiveStep instance or class (defaults to ActionStep)
            environment_name: Optional environment context
            **kwargs: Additional parameters for step initialization
        """
        if environment_name and environment_name not in self.environments:
            raise ValueError(f"Environment {environment_name} not found")
        
        env_name = environment_name or next(iter(self.environments.keys()))
        environment = self.environments[env_name]
        
        if step is None:
            step = ActionStep(
                agent_id=self.id,
                environment_name=env_name,
                environment_info=environment.get_global_state(agent_id=self.id),
                **kwargs
            )
        elif isinstance(step, type):
            step = step(
                agent_id=self.id,
                environment_name=env_name,
                environment_info=environment.get_global_state(agent_id=self.id),
                **kwargs
            )
        else:
            step.environment_name = env_name
            step.environment_info = environment.get_global_state(agent_id=self.id)
            step.agent_id = self.id
        
        logger.info(f"Executing cognitive step: {step.step_name}")
        
        result = await step.execute(self)

        return result

    async def run_episode(
        self,
        episode: Optional[CognitiveEpisode] = None,
        environment_name: Optional[str] = None,
        **kwargs
    ) -> List[Union[str, Dict[str, Any]]]:
        """
        Run a complete cognitive episode.
        
        Args:
            episode: CognitiveEpisode instance (defaults to Perception->Action[Observation]->Reflection)
            environment_name: Optional environment to use
            **kwargs: Additional parameters passed to each step
        """
        self._refresh_prompts()
        
        if episode is None:
            episode = CognitiveEpisode(
                steps=[PerceptionStep, ActionStep, ReflectionStep],
                environment_name=environment_name or next(iter(self.environments.keys()))
            )
        elif environment_name:
            episode.environment_name = environment_name

        results = []
        for step_class in episode.steps:
            result = await self.run_step(
                step=step_class,
                environment_name=episode.environment_name,
                **kwargs
            )
            results.append(result)
            
        return results