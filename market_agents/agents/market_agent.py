import json
import asyncio

from datetime import datetime, timezone
from typing import Dict, Any, Optional, Type, Union, List

from pydantic import Field

from market_agents.agents.base_agent.agent import Agent as LLMAgent
from market_agents.agents.market_agent_prompter import MarketAgentPromptManager, AgentPromptVariables
from market_agents.agents.market_schemas import PerceptionSchema, ReflectionSchema
from market_agents.agents.personas.persona import Persona
from market_agents.agents.protocols.protocol import Protocol
from market_agents.economics.econ_agent import EconomicAgent
from market_agents.environments.environment import MultiAgentEnvironment, LocalObservation
from market_agents.inference.message_models import LLMConfig, LLMPromptContext
from market_agents.memory.config import MarketMemoryConfig
from market_agents.memory.knowledge_base_agent import KnowledgeBaseAgent
from market_agents.memory.memory import MemoryObject, ShortTermMemory, LongTermMemory

class MarketAgent(LLMAgent):
    short_term_memory: ShortTermMemory = None
    long_term_memory: LongTermMemory = None
    environments: Dict[str, MultiAgentEnvironment] = Field(default_factory=dict)
    last_perception: Optional[Dict[str, Any]] = None
    last_action: Optional[Dict[str, Any]] = None
    last_observation: Optional[LocalObservation] = Field(default_factory=dict)
    episode_steps: List[MemoryObject] = Field(default_factory=list)
    protocol: Optional[Type[Protocol]] = None
    address: str = Field(default="", description="Agent's address")
    prompt_manager: MarketAgentPromptManager = Field(default_factory=lambda: MarketAgentPromptManager())
    economic_agent: Optional[EconomicAgent] = None
    knowledge_agent: Optional[KnowledgeBaseAgent] = None

    @classmethod
    def create(
        cls,
        memory_config: MarketMemoryConfig,
        db_conn,
        agent_id: str,
        use_llm: bool,
        llm_config: Optional[LLMConfig] = None,
        environments: Optional[Dict[str, MultiAgentEnvironment]] = None,
        protocol: Optional[Type[Protocol]] = None,
        persona: Optional[Persona] = None,
        econ_agent: Optional[EconomicAgent] = None,
        knowledge_agent: Optional[KnowledgeBaseAgent] = None
    ) -> 'MarketAgent':
        agent = cls(
            id=agent_id,
            short_term_memory=ShortTermMemory(memory_config, db_conn, agent_id),
            long_term_memory=LongTermMemory(memory_config, db_conn, agent_id),
            role=persona.role if persona else "agent",
            persona=persona.persona if persona else None,
            objectives=persona.objectives if persona else None,
            llm_config=llm_config or LLMConfig(),
            environments=environments or {},
            protocol=protocol,
            address=f"agent_{agent_id}_address",
            use_llm=use_llm,
            economic_agent=econ_agent,
            knowledge_agent=knowledge_agent
        )
        return agent

    async def perceive(
        self,
        environment_name: str,
        return_prompt: bool = False,
        structured_tool: bool = False
    ) -> Union[str, LLMPromptContext]:
        if environment_name not in self.environments:
            raise ValueError(f"Environment {environment_name} not found")

        environment_info = self.environments[environment_name].get_global_state()
        stm_cognitive = await self.short_term_memory.retrieve_recent_memories(limit=5)
        short_term_memories = []
        for mem in stm_cognitive:
            short_term_memories.append({
                "cognitive_step": mem.cognitive_step,
                "content": mem.content
            })

        print("\nCognitive Memory Results:")
        memory_strings = [f"Memory {i+1}:\n{mem}" for i, mem in enumerate(short_term_memories)]
        print("\033[94m" + "\n\n".join(memory_strings) + "\033[0m")

        task_str = f"Task: {self.task}" if self.task else ""
        env_state_str = f"Environment state: {str(environment_info)}" if environment_info else ""
        query_str = (task_str + "\n" + env_state_str).strip()

        ltm_episodes = await self.long_term_memory.retrieve_episodic_memories(
             agent_id=self.id,
             query=query_str,
             top_k=2
        )
        if self.knowledge_agent:
            retrieved_documents = self.knowledge_agent.retrieve(self.task, environment_name)

        print("\nEpisodic Memory Results:")
        memory_strings = [f"Memory {i+1}:\n{mem.model_dump()}" for i, mem in enumerate(ltm_episodes)]
        print("\033[94m" + "\n\n".join(memory_strings) + "\033[0m")

        variables = AgentPromptVariables(
            environment_name=environment_name,
            environment_info=environment_info,
            short_term_memory=short_term_memories,
            long_term_memory=[episode.model_dump() for episode in ltm_episodes],
            documents=[doc.model_dump() for doc in retrieved_documents]
        )

        prompt = self.prompt_manager.get_perception_prompt(variables.model_dump())
        response = await self.execute(
            prompt,
            output_format=PerceptionSchema.model_json_schema(),
            json_tool=structured_tool,
            return_prompt=return_prompt,
        )

        if not return_prompt:
            perception_mem = MemoryObject(
                agent_id=self.id,
                cognitive_step="perception",
                metadata={
                    "environment_name": environment_name,
                    "environment_info": environment_info
                },
                content=json.dumps(response),
                created_at=datetime.now(timezone.utc),
            )

            self.episode_steps.append(perception_mem)
            task = asyncio.create_task(self.short_term_memory.store_memory(perception_mem))

            self.last_perception = response
        return response

    async def generate_action(
        self,
        environment_name: str,
        perception: Optional[str] = None,
        return_prompt: bool = False,
        structured_tool: bool = False,
        action_schema: Dict = None
    ) -> Union[Dict[str, Any], LLMPromptContext]:
        if environment_name not in self.environments:
            raise ValueError(f"Environment {environment_name} not found")

        environment = self.environments[environment_name]
        #if perception is None and not return_prompt:
        #    perception = await self.perceive(environment_name)
        environment_info = environment.get_global_state()

        action_space = environment.action_space
        serialized_action_space = {
            "allowed_actions": [action_type.__name__ for action_type in action_space.allowed_actions]
        }

        variables = AgentPromptVariables(
            environment_name=environment_name,
            environment_info=environment_info,
            perception=perception,
            action_space=serialized_action_space,
            last_action=self.last_action,
            observation=self.last_observation
        )

        prompt = self.prompt_manager.get_action_prompt(variables.model_dump())

        if not action_schema:
            action_schema = action_space.get_action_schema()

        response = await self.execute(
            prompt,
            output_format=action_schema,
            json_tool=structured_tool,
            return_prompt=return_prompt
        )

        if not return_prompt:
            action = {"sender": self.id, "content": response}
            self.last_action = response

            action_mem = MemoryObject(
                agent_id=self.id,
                cognitive_step="action",
                metadata={
                    "action_space": serialized_action_space,
                    "last_action": self.last_action,
                    "observation": self.last_observation,
                    "perception": perception,
                    "environment_name": environment_name,
                    "environment_info": environment_info
                },
                content=json.dumps(response),
                created_at=datetime.now(timezone.utc),
            )

            self.episode_steps.append(action_mem)
            task = asyncio.create_task(self.short_term_memory.store_memory(action_mem))

            return action
        else:
            return response

    async def reflect(
        self,
        environment_name: str,
        environment_reward_weight: float = 0.5,
        self_reward_weight: float = 0.5,
        return_prompt: bool = False,
        structured_tool: bool = False,
    ) -> Union[str, LLMPromptContext]:
        if environment_name not in self.environments:
            raise ValueError(f"Environment {environment_name} not found")

        total_weight = environment_reward_weight + self_reward_weight
        if total_weight == 0:
            raise ValueError("Sum of weights must not be zero.")

        environment_reward_weight /= total_weight
        self_reward_weight /= total_weight

        environment = self.environments[environment_name]
        last_step = environment.history.steps[-1][1] if environment.history.steps else None

        if last_step:
            reward = last_step.info.get('agent_rewards', {}).get(self.id, 0.0) or 0.0
            local_observation = last_step.global_observation.observations.get(self.id)
            observation = local_observation.observation if local_observation else {}
        else:
            observation = {}
            reward = 0.0

        environment_info = environment.get_global_state()

        if observation:
            # Pre-serialize observation data
            if hasattr(observation, 'serialize_json'):
                observation_content = json.loads(observation.serialize_json())
            elif hasattr(observation, 'model_dump'):
                observation_content = observation.model_dump()
            else:
                observation_content = str(observation)

            observation_mem = MemoryObject(
                agent_id=self.id,
                cognitive_step="observation",
                metadata={
                    "environment_name": environment_name,
                    "environment_info": environment_info,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                content=json.dumps(observation_content),
                created_at=datetime.now(timezone.utc)
            )

            self.episode_steps.append(observation_mem)
            task = asyncio.create_task(self.short_term_memory.store_memory(observation_mem))

        previous_strategy = "No previous strategy available"
        previous_reflection = await self.short_term_memory.retrieve_recent_memories(cognitive_step='reflection', limit=1)
        if previous_reflection:
            last_reflection_obj = previous_reflection[0]
            previous_strategy = last_reflection_obj.metadata.get("strategy_update", "")
            if isinstance(previous_strategy, list):
                previous_strategy = " ".join(previous_strategy)

        variables = AgentPromptVariables(
            environment_name=environment_name,
            environment_info=environment_info,
            observation=observation,
            last_action=self.last_action,
            reward=reward,
            previous_strategy=previous_strategy
        )

        prompt = self.prompt_manager.get_reflection_prompt(variables.model_dump())

        response = await self.execute(
            prompt,
            output_format=ReflectionSchema.model_json_schema(),
            json_tool=structured_tool,
            return_prompt=return_prompt
        )

        if not return_prompt and isinstance(response, dict):
            self_reward = response.get("self_reward", 0.0)
            environment_reward = reward
            normalized_environment_reward = max(0.0, min(environment_reward / (1 + environment_reward), 1.0))

            total_reward_val = (
                normalized_environment_reward * environment_reward_weight +
                self_reward * self_reward_weight
            )

            reflection_mem = MemoryObject(
                agent_id=self.id,
                cognitive_step="reflection",
                metadata={
                    "total_reward": round(total_reward_val, 4),
                    "self_reward": round(self_reward, 4),
                    "observation": observation,
                    "strategy_update": response.get("strategy_update", ""),
                    "environment_reward": round(environment_reward, 4),
                    "environment_name": environment_name,
                    "environment_info": environment_info
                },
                content=json.dumps(response.get("reflection", "")),
                created_at=datetime.now(timezone.utc),
            )

            self.episode_steps.append(reflection_mem)
            task = await self.short_term_memory.store_memory(reflection_mem)

            task_str = f"Task: {self.task}" if self.task else ""
            env_state_str = f"Environment state: {str(environment_info)}" if environment_info else ""
            query_str = (task_str + "\n" + env_state_str).strip()
            await self.long_term_memory.store_episodic_memory(
                agent_id=self.id,
                task_query=query_str,
                steps=self.episode_steps,
                total_reward=round(total_reward_val),
                strategy_update=response.get("strategy_update", ""),
                metadata=None
            )
            self.episode_steps.clear()
        return response