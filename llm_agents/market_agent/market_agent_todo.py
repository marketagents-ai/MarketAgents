from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pydantic import Field
from base_agent.agent import Agent as LLMAgent
from base_agent.aiutilities import LLMConfig
from econ_agents.econ_agent import EconomicAgent, create_economic_agent
from environments.environment import Environment
from protocols.protocol import Protocol

class MarketAgent(LLMAgent, EconomicAgent):
    memory: List[Dict[str, Any]] = Field(default_factory=list)
    last_action: Optional[Dict[str, Any]] = None
    environments: Dict[str, Environment] = Field(default_factory=dict)
    protocol: Protocol = Field(..., description="Communication protocol eg. ACL")
    address: str = Field(default="", description="Agent's address")

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
        prompt = f"""Perceive the current state of the {environment_name} environment:

        Environment State: {environment_info}
        Recent Memories: {recent_memories if recent_memories else 'No recent memories'}

        Generate a brief monologue about your current perception of this environment."""
        
        return self.execute(prompt)

    def generate_action(self, environment_name: str) -> Dict[str, Any]:
        if environment_name not in self.environments:
            raise ValueError(f"Environment {environment_name} not found")

        perception = self.perceive(environment_name)
        environment_info = self.environments[environment_name].get_global_state()
        action_space = self.environments[environment_name].get_action_space()
        recent_memories = self.memory[-5:] if self.memory else []
        
        prompt = f"""Generate an action for the {environment_name} environment based on the following:

        Perception: {perception}
        Environment State: {environment_info}
        Recent Memories: {recent_memories if recent_memories else 'No recent memories'}
        Available Actions: {action_space}

        Choose an appropriate action for this environment. Respond with a JSON object containing 'type' (either 'bid' or 'ask'), 'price', and 'quantity'."""
        
        response = self.execute(prompt)
        
        action = {
            "sender": self.address,
            "content": response,
        }
        self.last_action = response
        return action

    def reflect(self, environment_name: str) -> None:
        if environment_name not in self.environments:
            raise ValueError(f"Environment {environment_name} not found")
        
        observation = self.environments[environment_name].get_observation(self.id)
        print(observation)
        environment_info = self.environments[environment_name].get_global_state()
        reward = observation.content.get('reward', 0)
        previous_strategy = self.memory[-1].get('strategy_update', 'No previous strategy') if self.memory else 'No previous strategy'
        
        prompt = f"""Reflect on this observation from the {environment_name} environment:

        Observation: {observation}
        Environment State: {environment_info}
        Last Action: {self.last_action}
        Reward: {reward}

        Actions:
        1. Reflect on the observation and surplus based on your last action
        2. Update strategy based on this reflection, the surplus, and your previous strategy

        Previous strategy: {previous_strategy}"""
        
        response = self.execute(prompt)
        
        # Split the response into reflection and strategy update
        reflection, strategy_update = response.split('\n\n', 1) if '\n\n' in response else (response, '')
        
        self.memory.append({
            "type": "reflection",
            "content": reflection,
            "strategy_update": strategy_update,
            "observation": observation,
            "reward": reward,
            "timestamp": datetime.now().isoformat()
        })