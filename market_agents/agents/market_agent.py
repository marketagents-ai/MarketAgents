from typing import Dict, Any, List, Optional, Type, Union
from datetime import datetime
from market_agents.agents.market_schemas import PerceptionSchema, ReflectionSchema
from pydantic import Field
from market_agents.agents.base_agent.agent import Agent as LLMAgent
from market_agents.inference.message_models import LLMConfig, LLMPromptContext
from market_agents.economics.econ_agent import EconomicAgent, create_economic_agent
from market_agents.environments.environment import MultiAgentEnvironment, LocalObservation
from market_agents.agents.protocols.protocol import Protocol
from market_agents.agents.market_agent_prompter import MarketAgentPromptManager, AgentPromptVariables
from market_agents.agents.personas.persona import Persona

class MarketAgent(LLMAgent, EconomicAgent):
    memory: List[Dict[str, Any]] = Field(default_factory=list)
    last_action: Optional[Dict[str, Any]] = None
    last_observation: Optional[LocalObservation] = Field(default_factory=dict)
    environments: Dict[str, MultiAgentEnvironment] = Field(default_factory=dict)
    protocol: Type[Protocol] = Field(..., description="Communication protocol class")
    address: str = Field(default="", description="Agent's address")
    prompt_manager: MarketAgentPromptManager = Field(default_factory=lambda: MarketAgentPromptManager())

    @classmethod
    def create(cls, agent_id: int, is_buyer: bool, num_units: int, base_value: float, use_llm: bool,
        initial_cash: float, initial_goods: int, good_name: str, noise_factor: float = 0.1,
        max_relative_spread: float = 0.2, llm_config: Optional[LLMConfig] = None,
        environments: Dict[str, MultiAgentEnvironment] = None, protocol: Type[Protocol] = None,
        persona: Persona = None) -> 'MarketAgent':
    
        econ_agent = create_economic_agent(
            agent_id=str(agent_id),
            goods=[good_name],
            buy_goods=[good_name] if is_buyer else [],
            sell_goods=[] if is_buyer else [good_name],
            base_values={good_name: base_value},
            initial_cash=initial_cash,
            initial_goods={good_name: initial_goods},
            num_units=num_units,
            noise_factor=noise_factor,
            max_relative_spread=max_relative_spread
        )
    
        role = "buyer" if is_buyer else "seller"
        llm_agent = LLMAgent(
            id=str(agent_id),
            role=role,
            persona=persona.persona if persona else None,
            objectives=persona.objectives if persona else None,
            llm_config=llm_config or LLMConfig()
        )

        return cls(
            id=str(agent_id),
            preference_schedule=econ_agent.value_schedules[good_name] if is_buyer else econ_agent.cost_schedules[good_name],
            endowment=econ_agent.endowment,
            utility_function=econ_agent.calculate_utility,
            max_relative_spread=econ_agent.max_relative_spread,
            use_llm=use_llm, 
            address=f"agent_{agent_id}_address",
            environments=environments or {},
            protocol=protocol,  # This should be the ACLMessage class, not an instance
            role=llm_agent.role,
            persona=llm_agent.persona,
            objectives=llm_agent.objectives,
            llm_config=llm_agent.llm_config,
        )

    async def perceive(self, environment_name: str, return_prompt: bool = False) -> Union[str, LLMPromptContext]:
        if environment_name not in self.environments:
            raise ValueError(f"Environment {environment_name} not found")

        environment_info = self.environments[environment_name].get_global_state()
        recent_memories = self.memory[-5:] if self.memory else []
        
        variables = AgentPromptVariables(
            environment_name=environment_name,
            environment_info=environment_info,
            recent_memories=recent_memories if recent_memories else []
        )
        
        prompt = self.prompt_manager.get_perception_prompt(variables.model_dump())
        
        return await self.execute(prompt, output_format=PerceptionSchema.model_json_schema(), return_prompt=return_prompt)

    async def generate_action(self, environment_name: str, perception: Optional[str] = None, return_prompt: bool = False) -> Union[Dict[str, Any], LLMPromptContext]:
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
        
        response = await self.execute(prompt, output_format=action_schema, return_prompt=return_prompt)
        
        if not return_prompt:
            action = {
                "sender": self.id,
                "content": response,
            }
            self.last_action = response
            return action
        else:
            return response

    async def reflect(self, environment_name: str, return_prompt: bool = False) -> Union[str, LLMPromptContext]:
        if environment_name not in self.environments:
            raise ValueError(f"Environment {environment_name} not found")
        
        environment = self.environments[environment_name]
        last_step = environment.history.steps[-1][1] if environment.history.steps else None
        
        if last_step:
            local_step = last_step.get_local_step(self.id)
            observation = local_step.observation.observation
            reward = local_step.info.get('reward', 0)
        else:
            observation = {}
            reward = 0

        environment_info = environment.get_global_state()
        previous_strategy = self.memory[-1].get('strategy_update', 'No previous strategy') if self.memory else 'No previous strategy'
        
        variables = AgentPromptVariables(
            environment_name=environment_name,
            environment_info=environment_info,
            observation=observation,
            last_action=self.last_action,
            reward=reward,
            previous_strategy=previous_strategy
        )
        
        prompt = self.prompt_manager.get_reflection_prompt(variables.model_dump())
        
        response = await self.execute(prompt, output_format=ReflectionSchema.model_json_schema(), return_prompt=return_prompt)
        
        if not return_prompt:
            self.memory.append({
                "type": "reflection",
                "content": response["reflection"],
                "strategy_update": response["strategy_update"],
                "observation": observation,
                "reward": reward,
                "timestamp": datetime.now().isoformat()
            })
            return response["reflection"]
        else:
            return response
