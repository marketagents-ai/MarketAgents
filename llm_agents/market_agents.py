import json
import random
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from ziagents import ZIAgent, Order, Trade
from agents import Agent as LLMAgent
from schema import MarketActionSchema, BidSchema

class MarketAgent(BaseModel):
    zi_agent: ZIAgent
    llm_agent: LLMAgent
    use_llm: bool = Field(default=False, description="Whether to use LLM for decision making")

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def create(cls, agent_id: int, is_buyer: bool, num_units: int, base_value: float, use_llm: bool,
               max_relative_spread: float = 0.2, llm_config: Dict[str, Any] = None):
        zi_agent = ZIAgent.generate(agent_id, is_buyer, num_units, base_value, max_relative_spread)
        
        role = "buyer" if is_buyer else "seller"
        #system_prompt = f"You are a {role} agent in a double auction market. Your goal is to maximize your profit."
        llm_agent = LLMAgent(role=role, llm_config=llm_config or {}, output_format="MarketActionSchema")

        return cls(zi_agent=zi_agent, llm_agent=llm_agent, use_llm=use_llm)

    def generate_bid(self, market_info: dict) -> Optional[Order]:
        if self.use_llm:
            return self._generate_llm_bid(market_info)
        else:
            return self.zi_agent.generate_bid()

    def _generate_llm_bid(self, market_info: dict) -> Optional[Order]:
        market_info_str = self._get_market_info(market_info)
        llm_response = self.llm_agent.execute(f"Generate a market action based on the following market information: {market_info_str}")
        print(f"\n---LLM JSON RESPONSE---")
        print(json.dumps(json.loads(llm_response), indent=2))
        try:
            market_action = MarketActionSchema.parse_raw(llm_response)
            if market_action.action == "hold":
                return None
            
            bid = market_action.bid
            return Order(
                agent_id=self.zi_agent.id,
                is_buy=self.zi_agent.preference_schedule.is_buyer,
                quantity=bid.quantity,
                price=bid.price,
                base_value=self.zi_agent.preference_schedule.get_value(self.zi_agent.allocation.goods + bid.quantity) if self.zi_agent.preference_schedule.is_buyer else None,
                base_cost=self.zi_agent.preference_schedule.get_value(self.zi_agent.allocation.initial_goods - self.zi_agent.allocation.goods + bid.quantity) if not self.zi_agent.preference_schedule.is_buyer else None
            )
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return None

    def _get_market_info(self, market_info: dict) -> str:
        return f"""

        Current Cash: {self.zi_agent.allocation.cash}
        Current Goods: {self.zi_agent.allocation.goods}
        Last Trade Price: {market_info['last_trade_price'] if market_info['last_trade_price'] is not None else 'N/A'}
        Average Market Price: {market_info['average_price'] if market_info['average_price'] is not None else 'N/A'}
        Total Trades: {market_info['total_trades']}
        Current Round: {market_info['current_round']}
        Base Value/Cost: {self.zi_agent.preference_schedule.get_value(self.zi_agent.allocation.goods + 1)}
        """

    def finalize_trade(self, trade: Trade):
        self.zi_agent.finalize_trade(trade)

    def respond_to_order(self, order: Order, accepted: bool):
        self.zi_agent.respond_to_order(order, accepted)

    def calculate_trade_surplus(self, trade: Trade) -> float:
        return self.zi_agent.calculate_trade_surplus(trade)

    def calculate_individual_surplus(self) -> float:
        return self.zi_agent.calculate_individual_surplus()

    def plot_order_history(self):
        self.zi_agent.plot_order_history()