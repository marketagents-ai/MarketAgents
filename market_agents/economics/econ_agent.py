from typing import List, Dict, Any, Optional, Union, Mapping
from pydantic import BaseModel, Field, computed_field, model_validator
import random
from functools import cached_property
import logging
import math

from market_agents.economics.econ_models import MarketAction, Bid, Ask, Trade, Endowment, Basket, Good, BuyerPreferenceSchedule, SellerPreferenceSchedule

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class EconomicAgent(BaseModel):
    id: str
    endowment: Endowment
    value_schedules: Dict[str, BuyerPreferenceSchedule] = Field(default_factory=dict)
    cost_schedules: Dict[str, SellerPreferenceSchedule] = Field(default_factory=dict)
    max_relative_spread: float = Field(default=0.2)

    @model_validator(mode='after')
    def validate_schedules(self):
        for good in set(self.value_schedules.keys()) & set(self.cost_schedules.keys()):
            raise ValueError(f"Agent cannot be both buyer and seller of the same good: {good}")
        return self

    @computed_field
    @cached_property
    def marginal_value(self) -> Dict[str, float]:
        values = {}
        for good_name, schedule in self.value_schedules.items():
            quantity = int(self.endowment.current_basket.get_good_quantity(good_name))
            values[good_name] = schedule.get_value(quantity + 1)
        for good_name, schedule in self.cost_schedules.items():
            quantity = int(self.endowment.current_basket.get_good_quantity(good_name))
            values[good_name] = schedule.get_value(quantity) if quantity > 0 else schedule.get_value(1)
        logger.debug(f"Agent {self.id} marginal values: {values}")
        return values

    def is_buyer(self, good_name: str) -> bool:
        return good_name in self.value_schedules

    def is_seller(self, good_name: str) -> bool:
        return good_name in self.cost_schedules

    def generate_bid(self, good_name: str, market_info: Optional[Dict] = None) -> Optional[Bid]:
        logger.debug(f"Agent {self.id} attempting to generate bid for {good_name}")
        if not self._can_generate_bid(good_name):
            logger.debug(f"Agent {self.id} cannot generate bid for {good_name}")
            return None
        price = self._calculate_bid_price(good_name)
        logger.debug(f"Agent {self.id} generated bid for {good_name}: price={price}")
        return Bid(price=price, quantity=1)

    def generate_ask(self, good_name: str, market_info: Optional[Dict] = None) -> Optional[Ask]:
        logger.debug(f"Agent {self.id} attempting to generate ask for {good_name}")
        if not self._can_generate_ask(good_name):
            logger.debug(f"Agent {self.id} cannot generate ask for {good_name}")
            return None
        price = self._calculate_ask_price(good_name)
        logger.debug(f"Agent {self.id} generated ask for {good_name}: price={price}")
        return Ask(price=price, quantity=1)

    def process_trade(self, trade: Trade):
        # Calculate current utility
        current_utility = self.calculate_utility(self.endowment.current_basket.goods_dict, self.endowment.current_basket.cash)
        
        # Simulate the trade
        new_basket = self.endowment.simulate_trade(trade)
        
        # Calculate new utility
        new_utility = self.calculate_utility(new_basket.goods_dict, new_basket.cash)
        
        if new_utility < current_utility:
            raise ValueError(f"Trade would reduce utility for Agent {self.id}")
        
        # Apply the trade
        self.endowment.add_trade(trade)
        logger.debug(f"Agent {self.id} processed trade. New utility: {new_utility:.2f}")

    def calculate_utility(self, allocation: Mapping[str, int], cash: float) -> float:
        utility = cash
        logger.debug(f"Agent {self.id} calculating utility. Initial cash: {utility}")
        
        for good, quantity in allocation.items():
            if good in self.value_schedules:
                schedule = self.value_schedules[good]
                value_sum = sum(schedule.get_value(q) for q in range(1, quantity + 1))
                utility += value_sum
                logger.debug(f"Buyer utility for {good}: +{value_sum:.2f} (value)")
            elif good in self.cost_schedules:
                schedule = self.cost_schedules[good]
                inventory_value = sum(schedule.get_value(q) for q in range(1, quantity + 1))
                utility += inventory_value
                logger.debug(f"Seller utility for {good}: +{inventory_value:.2f} (inventory value)")

        logger.debug(f"Agent {self.id} final utility: {utility:.2f}")
        return utility

    def calculate_individual_surplus(self) -> float:
        current_utility = self.calculate_utility(self.endowment.current_basket.goods_dict, self.endowment.current_basket.cash)
        initial_utility = self.calculate_utility(self.endowment.initial_basket.goods_dict, self.endowment.initial_basket.cash)
        print(f"current_utility: {current_utility}, initial_utility: {initial_utility}")
        surplus = current_utility - initial_utility
        logger.debug(f"Agent {self.id} calculated surplus: {surplus}")
        return surplus

    def _can_generate_bid(self, good_name: str) -> bool:
        can_bid = (self.is_buyer(good_name) and 
                   self.endowment.current_basket.cash > 0)
        logger.debug(f"Agent {self.id} can generate bid for {good_name}: {can_bid}")
        return can_bid

    def _can_generate_ask(self, good_name: str) -> bool:
        can_ask = (self.is_seller(good_name) and 
                   self.marginal_value[good_name] > 0 and
                   self.endowment.current_basket.get_good_quantity(good_name) > 0)
        logger.debug(f"Agent {self.id} can generate ask for {good_name}: {can_ask}")
        return can_ask

    def _calculate_bid_price(self, good_name: str) -> float:
        max_bid = min(self.endowment.current_basket.cash, self.marginal_value[good_name])
        price = random.uniform(
            max_bid * (1 - self.max_relative_spread),
            max_bid
        )
        logger.debug(f"Agent {self.id} calculated bid price for {good_name}: {price}")
        return price

    def _calculate_ask_price(self, good_name: str) -> float:
        min_price = self.marginal_value[good_name]
        price = random.uniform(
            min_price,
            min_price * (1 + self.max_relative_spread)
        )
        logger.debug(f"Agent {self.id} calculated ask price for {good_name}: {price}")
        return price

    def print_status(self):
        print(f"\nAgent ID: {self.id}")
        print(f"Current Endowment:")
        print(f"  Cash: {self.endowment.current_basket.cash:.2f}")
        print(f"  Goods: {self.endowment.current_basket.goods_dict}")
        for good in self.value_schedules.keys() | self.cost_schedules.keys():
            if self.is_buyer(good):
                print(f"Buyer Values for {good}:")
                schedule = self.value_schedules[good]
                for quantity, value in schedule.values.items():
                    print(f"  Quantity: {quantity}, Value: {value:.2f}")
            elif self.is_seller(good):
                print(f"Seller Costs for {good}:")
                schedule = self.cost_schedules[good]
                for quantity, cost in schedule.values.items():
                    print(f"  Quantity: {quantity}, Cost: {cost:.2f}")
        current_utility = self.calculate_utility(self.endowment.current_basket.goods_dict, self.endowment.current_basket.cash)
        print(f"Current Utility: {current_utility:.2f}")
        print(f"Current Surplus: {self.calculate_individual_surplus():.2f}")

def create_economic_agent(
    agent_id: str,
    goods: List[str],
    buy_goods: List[str],
    sell_goods: List[str],
    base_values: Dict[str, float],
    initial_cash: float,
    initial_goods: Dict[str, int],
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

    value_schedules = {
        good: BuyerPreferenceSchedule(
            num_units=num_units,
            base_value=base_values[good],
            noise_factor=noise_factor
        ) for good in buy_goods
    }
    cost_schedules = {
        good: SellerPreferenceSchedule(
            num_units=num_units,
            base_value=base_values[good] * 0.8,  # Set seller base value lower than buyer base value
            noise_factor=noise_factor
        ) for good in sell_goods
    }

    return EconomicAgent(
        id=agent_id,
        endowment=endowment,
        value_schedules=value_schedules,
        cost_schedules=cost_schedules,
        max_relative_spread=max_relative_spread
    )

# Example usage
if __name__ == "__main__":
    # Define parameters for creating economic agents
    goods = ["apple", "banana"]
    base_values = {"apple": 10.0, "banana": 8.0}
    initial_cash = 100.0
    initial_goods = {"apple": 0, "banana": 0}

    # Create agents
    buyer = create_economic_agent(
        agent_id="agent_1",
        goods=goods,
        buy_goods=["apple", "banana"],
        sell_goods=[],
        base_values=base_values,
        initial_cash=initial_cash,
        initial_goods=initial_goods,
    )

    seller = create_economic_agent(
        agent_id="agent_2",
        goods=goods,
        buy_goods=[],
        sell_goods=["apple", "banana"],
        base_values={good: value * 0.8 for good, value in base_values.items()},  # Lower base values for seller
        initial_cash=0,
        initial_goods={"apple": 10, "banana": 10},
    )

    # Print initial status
    print("Initial Status:")
    buyer.print_status()
    seller.print_status()

    # Generate bids and asks until a match is found or max attempts reached
    trade_id = 1
    max_attempts = 10
    for good in goods:
        print(f"\nTrading {good}:")
        for attempt in range(max_attempts):
            bid = buyer.generate_bid(good)
            ask = seller.generate_ask(good)
            
            if bid and ask:
                print(f"  Attempt {attempt + 1}: Bid = {bid.price:.2f}, Ask = {ask.price:.2f}")
                
                if bid.price >= ask.price:
                    trade_price = (bid.price + ask.price) / 2
                    trade = Trade(
                        trade_id=trade_id,
                        buyer_id=buyer.id,
                        seller_id=seller.id,
                        price=trade_price,
                        quantity=1,
                        good_name=good
                    )
                    print(f"  Trade executed: Price = {trade_price:.2f}, Quantity = 1")
                    
                    try:
                        buyer.process_trade(trade)
                        seller.process_trade(trade)
                        trade_id += 1
                        break
                    except ValueError as e:
                        print(f"  Trade failed: {str(e)}")
            else:
                print(f"  Attempt {attempt + 1}: No valid bid or ask generated")
        
        if attempt == max_attempts - 1:
            print(f"  No successful trade for {good} after {max_attempts} attempts")

    # Print final status
    print("\nFinal Status:")
    buyer.print_status()
    seller.print_status()