from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from market_agent.market_schemas import ReflectionSchema
from pydantic import Field
from base_agent.agent import Agent as LLMAgent
from base_agent.aiutilities import LLMConfig
from econ_agents.econ_agent import EconomicAgent, create_economic_agent
from environments.environment import Environment
from protocols.protocol import Protocol
from market_agent.market_agent_prompter import MarketAgentPromptManager

class MarketAgent(LLMAgent, EconomicAgent):
    memory: List[Dict[str, Any]] = Field(default_factory=list)
    last_action: Optional[Dict[str, Any]] = None
    environments: Dict[str, Environment] = Field(default_factory=dict)
    protocol: Protocol = Field(..., description="Communication protocol eg. ACL")
    address: str = Field(default="", description="Agent's address")
    prompt_manager: MarketAgentPromptManager = Field(default_factory=lambda: MarketAgentPromptManager())

    @classmethod
    def create(cls, agent_id: int, is_buyer: bool, num_units: int, base_value: float, use_llm: bool,
            initial_cash: float, initial_goods: int, noise_factor: float = 0.1,
            max_relative_spread: float = 0.2, llm_config: Optional[LLMConfig] = None,
            environments: Dict[str, Environment] = None, protocol: Protocol = None) -> 'MarketAgent':
        econ_agent = create_economic_agent(
            agent_id=agent_id,
            is_buyer=is_buyer,
            num_units=num_units,
            base_value=base_value,
            initial_cash=initial_cash,
            initial_goods=initial_goods,
            utility_function_type="step",
            noise_factor=noise_factor,
            max_relative_spread=max_relative_spread
        )
        role = "buyer" if is_buyer else "seller"
        llm_agent = LLMAgent(id=str(agent_id), role=role, llm_config=llm_config or LLMConfig())

        return cls(
            id=str(agent_id),
            is_buyer=is_buyer,
            preference_schedule=econ_agent.preference_schedule,
            endowment=econ_agent.endowment,
            utility_function=econ_agent.utility_function,
            max_relative_spread=econ_agent.max_relative_spread,
            use_llm=use_llm, 
            address=f"agent_{agent_id}_address",
            environments=environments or {},
            protocol=protocol,
            role=llm_agent.role,
            llm_config=llm_agent.llm_config,
            # Add any other necessary fields from llm_agent here
        )

    def perceive(self, environment_name: str) -> str:
        if environment_name not in self.environments:
            raise ValueError(f"Environment {environment_name} not found")

        environment_info = self.environments[environment_name].get_global_state()
        recent_memories = self.memory[-5:] if self.memory else []
        
        variables = {
            "environment_name": environment_name,
            "environment_info": environment_info,
            "recent_memories": recent_memories if recent_memories else 'No recent memories'
        }
        
        prompt = self.prompt_manager.get_perception_prompt(variables)
        
        return self.execute(prompt, output_format="text")

    def generate_action(self, environment_name: str) -> Dict[str, Any]:
        if environment_name not in self.environments:
            raise ValueError(f"Environment {environment_name} not found")

        environment = self.environments[environment_name]
        perception = self.perceive(environment_name)
        environment_info = environment.get_global_state()
        action_space = environment.get_action_space()
        recent_memories = self.memory[-5:] if self.memory else []
        
        variables = {
            "environment_name": environment_name,
            "perception": perception,
            "environment_info": environment_info,
            "recent_memories": recent_memories if recent_memories else 'No recent memories',
            "action_space": action_space
        }
        
        prompt = self.prompt_manager.get_action_prompt(variables)
        
        action_schema = environment.get_action_schema()
        
        response = self.execute(prompt, output_format=action_schema)
        
        action = {
            "sender": self.id,
            "content": response,
        }
        self.last_action = response
        return action

    def reflect(self, environment_name: str) -> None:
        if environment_name not in self.environments:
            raise ValueError(f"Environment {environment_name} not found")
        
        observation = self.environments[environment_name].get_observation(self.id)
        environment_info = self.environments[environment_name].get_global_state()
        reward = observation.content.get('reward', 0)
        previous_strategy = self.memory[-1].get('strategy_update', 'No previous strategy') if self.memory else 'No previous strategy'
        
        variables = {
            "environment_name": environment_name,
            "observation": observation,
            "environment_info": environment_info,
            "last_action": self.last_action,
            "reward": reward,
            "previous_strategy": previous_strategy
        }
        
        prompt = self.prompt_manager.get_reflection_prompt(variables)
        
        response = self.execute(prompt, output_format=ReflectionSchema.schema())
        
        self.memory.append({
            "type": "reflection",
            "content": response["reflection"],
            "strategy_update": response["strategy_update"],
            "observation": observation,
            "reward": reward,
            "timestamp": datetime.now().isoformat()
        })