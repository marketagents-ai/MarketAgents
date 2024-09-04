from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import Field
from base_agent.agent import Agent as LLMAgent
from base_agent.aiutilities import LLMConfig
from econ_agents.econ_agent import EconomicAgent, create_economic_agent
from market_agent.market_schemas import MarketActionSchema

class MarketAgent(LLMAgent, EconomicAgent):
    memory: List[Dict[str, Any]] = Field(default_factory=list)
    use_llm: bool = Field(default=False, description="Whether to use LLM for decision making")
    address: str = Field(default="", description="Agent's address")

    @classmethod
    def create(cls, agent_id: int, is_buyer: bool, num_units: int, base_value: float, use_llm: bool,
               initial_cash: float, initial_goods: int, noise_factor: float = 0.1,
               max_relative_spread: float = 0.2, llm_config: Optional[LLMConfig] = None) -> 'MarketAgent':
        econ_agent = create_economic_agent(agent_id, is_buyer, num_units, base_value, initial_cash, initial_goods,
                                           "step", noise_factor, max_relative_spread)
        role = econ_agent.get_role()
        llm_agent = LLMAgent(role=role, llm_config=llm_config or LLMConfig(), 
                             output_format=MarketActionSchema.model_json_schema())

        return cls(is_buyer=is_buyer, preference_schedule=econ_agent.preference_schedule,
                   endowment=econ_agent.endowment, utility_function=econ_agent.utility_function,
                   max_relative_spread=econ_agent.max_relative_spread, use_llm=use_llm, 
                   address=f"agent_{agent_id}_address", **llm_agent.dict())

    def generate_action(self, market_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        monologue = self.percieve()
        
        prompt = f"Generate a market action based on the following:\n\nMonologue: {monologue}\n\nMarket History: {market_info}"
        
        response = self.execute(prompt)
        return response

    def percieve(self) -> str:

        prompt = f"Based on these recent memories, generate a brief monologue about your current market perception:\n\n{self.memory}"
        
        return self.execute(prompt)

    def reflect(self, observation: Dict[str, Any]) -> None:
        prompt = f"Reflect on this market observation and update beliefs:\n\n{observation}"
        reflection = self.execute(prompt)
        
        self.memory.append({
            "type": "reflection",
            "content": reflection,
            "observation": observation,
            "timestamp": datetime.now().isoformat()
        })