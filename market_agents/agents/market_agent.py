import json
from datetime import datetime
from typing import Dict, Any, Optional, Type, Union

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
from market_agents.memory.memory import ShortTermMemory, MemoryObject
from market_agents.memory.vector_search import LongTermMemory


class MarketAgent(LLMAgent):
    short_term_memory: ShortTermMemory = None
    long_term_memory: LongTermMemory = None
    last_perception: Optional[Dict[str, Any]] = None
    last_action: Optional[Dict[str, Any]] = None
    last_observation: Optional[LocalObservation] = Field(default_factory=dict)
    environments: Dict[str, MultiAgentEnvironment] = Field(default_factory=dict)
    protocol: Optional[Type[Protocol]] = None
    address: str = Field(default="", description="Agent's address")
    prompt_manager: MarketAgentPromptManager = Field(default_factory=lambda: MarketAgentPromptManager())
    economic_agent: Optional[EconomicAgent] = None

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
    ) -> 'MarketAgent':

        agent = cls(
            id=agent_id,
            long_term_memory=LongTermMemory(memory_config, db_conn),
            short_term_memory=ShortTermMemory(memory_config, db_conn),
            role=persona.role if persona else "agent",
            persona=persona.persona if persona else None,
            objectives=persona.objectives if persona else None,
            llm_config=llm_config or LLMConfig(),
            environments=environments or {},
            protocol=protocol,
            address=f"agent_{agent_id}_address",
            use_llm=use_llm,
            economic_agent=econ_agent
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

        variables = AgentPromptVariables(
            environment_name=environment_name,
            environment_info=environment_info,
            short_term_memory=self.short_term_memory,
            long_term_memory=self.long_term_memory
        )

        prompt = self.prompt_manager.get_perception_prompt(variables.model_dump())

        return await self.execute(prompt, output_format=PerceptionSchema.model_json_schema(), json_tool=structured_tool,
                                  return_prompt=return_prompt)

    async def generate_action(
            self,
            environment_name: str,
            perception: Optional[str] = None,
            return_prompt: bool = False,
            structured_tool: bool = False,
    ) -> Union[Dict[str, Any], LLMPromptContext]:
        if environment_name not in self.environments:
            raise ValueError(f"Environment {environment_name} not found")

        environment = self.environments[environment_name]
        if perception is None and not return_prompt:
            perception = await self.perceive(environment_name)
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

        action_schema = action_space.get_action_schema()

        response = await self.execute(prompt, output_format=action_schema, json_tool=structured_tool,
                                      return_prompt=return_prompt)

        if not return_prompt:
            action = {
                "sender": self.id,
                "content": response,
            }
            self.last_action = response
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
            reward = last_step.info.get('agent_rewards', {}).get(self.id, 0.0)
            if reward is None:
                reward = 0.0
            local_observation = last_step.global_observation.observations.get(self.id)
            if local_observation:
                observation = local_observation.observation
            else:
                observation = {}
        else:
            observation = {}
            reward = 0.0

        environment_info = environment.get_global_state()
        previous_strategy = None
        if self.short_term_memory.memories:
            for memory_item in reversed(self.short_term_memory.memories):
                if 'strategy_update' in memory_item.metadata:
                    previous_strategy = memory_item.metadata['strategy_update']
                    break

        if previous_strategy is None:
            previous_strategy = "No previous strategy available"
        elif isinstance(previous_strategy, list):
            previous_strategy = " ".join(previous_strategy)

        variables = AgentPromptVariables(
            environment_name=environment_name,
            environment_info=environment_info,
            observation=observation if isinstance(observation, dict) else observation.model_dump(),
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
            environment_reward = reward if reward is not None else 0.0

            normalized_environment_reward = environment_reward / (1 + environment_reward)
            normalized_environment_reward = max(0.0, min(normalized_environment_reward, 1.0))

            total_reward = (
                    normalized_environment_reward * environment_reward_weight +
                    self_reward * self_reward_weight
            )
            self.short_term_memory.store_memory(MemoryObject(
                agent_id=self.id,
                cognitive_step="reflection",
                metadata={"total_reward": round(total_reward, 4),
                          "self_reward": round(self_reward, 4),
                          "observation": observation if isinstance(observation, dict) else observation.model_dump(),
                          "strategy_update": response.get("strategy_update", "")},
                content=json.dumps(response.get("reflection", "")),
                created_at=datetime.now(),
            ))
            self.long_term_memory.store_memory(MemoryObject(
                agent_id=self.id,
                cognitive_step="reflection",
                metadata={"total_reward": round(total_reward, 4),
                          "self_reward": round(self_reward, 4),
                          "observation": observation if isinstance(observation, dict) else observation.model_dump(),
                          "strategy_update": response.get("strategy_update", "")},
                content=json.dumps(response.get("reflection", "")),
                created_at=datetime.now(),
            ))
            return response.get("reflection", "")
        else:
            return response
