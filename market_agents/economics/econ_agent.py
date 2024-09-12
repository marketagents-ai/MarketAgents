from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, computed_field
import random
from functools import cached_property

from market_agents.economics.econ_models import MarketAction, Bid, Ask, Trade, Endowment, Basket, Good, BuyerPreferenceSchedule, SellerPreferenceSchedule

class EconomicAgent(BaseModel):
    id: str
    is_buyer: bool
    endowment: Endowment
    value_schedules: Dict[str, BuyerPreferenceSchedule]
    cost_schedules: Dict[str, SellerPreferenceSchedule]
    max_relative_spread: float = Field(default=0.2)

    @computed_field
    @cached_property
    def marginal_value(self) -> Dict[str, float]:
        if self.is_buyer:
            return {good_name: schedule.get_value(int(self.endowment.current_basket.get_good_quantity(good_name) + 1))
                    for good_name, schedule in self.value_schedules.items()}
        else:
            return {good_name: schedule.get_value(int(self.endowment.current_basket.get_good_quantity(good_name) + 1))
                    for good_name, schedule in self.cost_schedules.items()}

    def get_role(self) -> str:
        return "buyer" if self.is_buyer else "seller"

    def generate_bid(self, good_name: str, market_info: Optional[Dict] = None) -> Optional[Bid]:
        if not self._can_generate_bid(good_name):
            return None
        price = self._calculate_bid_price(good_name)
        return Bid(price=price, quantity=1)

    def generate_ask(self, good_name: str, market_info: Optional[Dict] = None) -> Optional[Ask]:
        if not self._can_generate_ask(good_name):
            return None
        price = self._calculate_ask_price(good_name)
        return Ask(price=price, quantity=1)

    def finalize_trade(self, trade: Trade):
        self.endowment.add_trade(trade)

    def calculate_utility(self) -> float:
        if self.is_buyer:
            return sum(schedule.get_value(int(self.endowment.current_basket.get_good_quantity(good_name)))
                       for good_name, schedule in self.value_schedules.items())
        else:
            return self.endowment.current_basket.cash

    def calculate_individual_surplus(self) -> float:
        return self._calculate_buyer_surplus() if self.is_buyer else self._calculate_seller_surplus()

    def _can_generate_bid(self, good_name: str) -> bool:
        return (self.is_buyer and 
                self.marginal_value[good_name] > 0 and 
                self.endowment.current_basket.cash >= self.marginal_value[good_name])

    def _can_generate_ask(self, good_name: str) -> bool:
        return not self.is_buyer and self.marginal_value[good_name] > 0

    def _calculate_bid_price(self, good_name: str) -> float:
        price = random.uniform(
            self.marginal_value[good_name] * (1 - self.max_relative_spread),
            self.marginal_value[good_name]
        )
        return min(price, self.endowment.current_basket.cash, self.marginal_value[good_name])

    def _calculate_ask_price(self, good_name: str) -> float:
        price = random.uniform(
            self.marginal_value[good_name],
            self.marginal_value[good_name] * (1 + self.max_relative_spread)
        )
        return max(price, self.marginal_value[good_name])

    def _calculate_buyer_surplus(self) -> float:
        return self.calculate_utility() - (self.endowment.initial_basket.cash - self.endowment.current_basket.cash)

    def _calculate_seller_surplus(self) -> float:
        production_cost = sum(schedule.get_value(int(self.endowment.current_basket.get_good_quantity(good_name)))
                              for good_name, schedule in self.cost_schedules.items())
        return self.endowment.current_basket.cash - self.endowment.initial_basket.cash - production_cost

def create_economic_agent(
    agent_id: str,
    is_buyer: bool,
    goods: List[str],
    base_values: Dict[str, float],
    initial_cash: float,
    initial_goods: Dict[str, float],
    num_units: int = 10,
    noise_factor: float = 0.1,
    max_relative_spread: float = 0.2,
) -> EconomicAgent:
    initial_goods_list = [Good(name=name, quantity=quantity) for name, quantity in initial_goods.items()]
    initial_basket = Basket(
        cash=initial_cash,
        goods=initial_goods_list
    )
    endowment = Endowment(
        initial_basket=initial_basket,
        agent_id=agent_id
    )

    if is_buyer:
        value_schedules = {
            good: BuyerPreferenceSchedule(
                num_units=num_units,
                base_value=base_values[good],
                noise_factor=noise_factor
            ) for good in goods
        }
        cost_schedules = {}
    else:
        value_schedules = {}
        cost_schedules = {
            good: SellerPreferenceSchedule(
                num_units=num_units,
                base_value=base_values[good],
                noise_factor=noise_factor
            ) for good in goods
        }

    return EconomicAgent(
        id=agent_id,
        is_buyer=is_buyer,
        endowment=endowment,
        value_schedules=value_schedules,
        cost_schedules=cost_schedules,
        max_relative_spread=max_relative_spread
    )

# Example usage
if __name__ == "__main__":
    # Define parameters for creating an economic agent
    agent_id = "agent1"
    is_buyer = True
    goods = ["apple", "banana"]
    base_values = {"apple": 10.0, "banana": 8.0}
    initial_cash = 100.0
    initial_goods = {"apple": 2.0, "banana": 3.0}

    # Create an economic agent
    agent = create_economic_agent(
        agent_id=agent_id,
        is_buyer=is_buyer,
        goods=goods,
        base_values=base_values,
        initial_cash=initial_cash,
        initial_goods=initial_goods,
    )

    # Print agent information
    print(f"Agent ID: {agent.id}")
    print(f"Is Buyer: {agent.is_buyer}")
    print(f"Initial Endowment:")
    print(f"  Cash: {agent.endowment.initial_basket.cash}")
    print(f"  Goods: {agent.endowment.initial_basket.goods_dict}")

    # Generate a bid for an apple
    bid = agent.generate_bid("apple")
    if bid:
        print(f"\nGenerated bid for apple: Price = {bid.price}, Quantity = {bid.quantity}")
    else:
        print("\nUnable to generate bid for apple")

    # Calculate and print utility
    utility = agent.calculate_utility()
    print(f"\nCurrent Utility: {utility}")