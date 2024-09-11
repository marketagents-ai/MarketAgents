from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field, computed_field
import random
from functools import cached_property

from market_agents.economics.utility import (
    UtilityFunction, CostFunction,
    create_utility_function, create_cost_function
)

from market_agents.economics.econ_models import MarketAction, Bid, Ask, Trade, Endowment, Basket, Good

class EconomicAgent(BaseModel):
    id: str
    is_buyer: bool
    endowment: Endowment
    utility_function: UtilityFunction
    cost_function: Optional[CostFunction] = None
    max_relative_spread: float = Field(default=0.2)

    @computed_field
    @cached_property
    def marginal_value(self) -> Dict[str, float]:
        if self.is_buyer:
            return {good.name: self.utility_function.marginal_utility(self.endowment.current_basket, good.name) 
                    for good in self.endowment.current_basket.goods}
        else:
            if self.cost_function is None:
                raise ValueError("Seller agent must have a cost function")
            return {good.name: self.cost_function.marginal_cost(self.endowment.current_basket, good.name) 
                    for good in self.endowment.current_basket.goods}

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
        return self.utility_function.evaluate(self.endowment.current_basket)

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
        if self.cost_function is None:
            raise ValueError("Seller agent must have a cost function")
        production_cost = self.cost_function.evaluate(self.endowment.current_basket)
        return self.endowment.current_basket.cash - self.endowment.initial_basket.cash - production_cost

def create_economic_agent(
    agent_id: str,
    is_buyer: bool,
    goods: List[str],
    base_values: Dict[str, float],
    initial_cash: float,
    initial_goods: Dict[str, float],
    utility_function_type: Literal["stepwise", "cobb-douglas"],
    cost_function_type: Literal["stepwise", "quadratic"],
    num_units: int = 10,
    noise_factor: float = 0.1,
    cash_weight: float = 1.0,
    max_relative_spread: float = 0.2,
    cobb_douglas_scale: float = 100  # Add this parameter
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

    utility_function = create_utility_function(
        utility_function_type, goods, base_values, num_units, noise_factor, cash_weight, cobb_douglas_scale
    )
    
    cost_function = None
    if not is_buyer:
        cost_function = create_cost_function(
            cost_function_type, goods, base_values, num_units, noise_factor
        )

    return EconomicAgent(
        id=agent_id,
        is_buyer=is_buyer,
        endowment=endowment,
        utility_function=utility_function,
        cost_function=cost_function,
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
    utility_function_type = "cobb-douglas"
    cost_function_type = "quadratic"
    cobb_douglas_scale = 10  # Add this line

    # Create an economic agent
    agent = create_economic_agent(
        agent_id=agent_id,
        is_buyer=is_buyer,
        goods=goods,
        base_values=base_values,
        initial_cash=initial_cash,
        initial_goods=initial_goods,
        utility_function_type=utility_function_type,
        cost_function_type=cost_function_type,
        cobb_douglas_scale=cobb_douglas_scale  # Add this line
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