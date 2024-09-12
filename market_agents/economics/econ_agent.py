from typing import List, Dict, Optional
from pydantic import BaseModel, Field, model_validator
import random
import logging

from market_agents.economics.econ_models import (
    MarketAction,
    Bid,
    Ask,
    Trade,
    Endowment,
    Basket,
    Good,
    BuyerPreferenceSchedule,
    SellerPreferenceSchedule,
)

# Set up logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EconomicAgent(BaseModel):
    id: str
    endowment: Endowment
    value_schedules: Dict[str, BuyerPreferenceSchedule] = Field(default_factory=dict)
    cost_schedules: Dict[str, SellerPreferenceSchedule] = Field(default_factory=dict)
    max_relative_spread: float = Field(default=0.2)

    @model_validator(mode='after')
    def validate_schedules(self):
        overlapping_goods = set(self.value_schedules.keys()) & set(self.cost_schedules.keys())
        if overlapping_goods:
            raise ValueError(f"Agent cannot be both buyer and seller of the same good(s): {overlapping_goods}")
        return self

    def is_buyer(self, good_name: str) -> bool:
        return good_name in self.value_schedules

    def is_seller(self, good_name: str) -> bool:
        return good_name in self.cost_schedules

    def generate_bid(self, good_name: str, market_info: Optional[Dict] = None) -> Optional[Bid]:
        if not self._can_generate_bid(good_name):
            return None
        price = self._calculate_bid_price(good_name)
        return Bid(price=price, quantity=1) if price is not None else None

    def generate_ask(self, good_name: str, market_info: Optional[Dict] = None) -> Optional[Ask]:
        if not self._can_generate_ask(good_name):
            return None
        price = self._calculate_ask_price(good_name)
        return Ask(price=price, quantity=1) if price is not None else None

    def process_trade(self, trade: Trade) -> bool:
        current_utility = self.calculate_utility(self.endowment.current_basket)
        new_basket = self.endowment.simulate_trade(trade)
        new_utility = self.calculate_utility(new_basket)
        
        if new_utility < current_utility:
            logger.warning(f"Trade would reduce utility for Agent {self.id}. Skipping this trade.")
            return False
        
        self.endowment.add_trade(trade)
        logger.info(f"Agent {self.id} processed trade. New utility: {new_utility:.2f}")
        return True

    def calculate_utility(self, basket: Basket) -> float:
        utility = basket.cash
        
        for good, quantity in basket.goods_dict.items():
            if self.is_buyer(good):
                schedule = self.value_schedules[good]
                value_sum = sum(schedule.get_value(q) for q in range(1, int(quantity) + 1))
                utility += value_sum
            elif self.is_seller(good):
                schedule = self.cost_schedules[good]
                initial_quantity = self.endowment.initial_basket.get_good_quantity(good)
                units_sold = initial_quantity - quantity
                # Correctly sum the costs of unsold units
                value_sum = sum(schedule.get_value(q) for q in range(int(units_sold + 1), int(initial_quantity + 1)))
                utility += value_sum
        
        return utility

    def calculate_individual_surplus(self) -> float:
        current_utility = self.calculate_utility(self.endowment.current_basket)
        initial_utility = self.calculate_utility(self.endowment.initial_basket)
        surplus = current_utility - initial_utility
        logger.info(f"Agent {self.id} calculated surplus: {surplus:.2f}")
        return surplus

    def _can_generate_bid(self, good_name: str) -> bool:
        return (
            self.is_buyer(good_name)
            and self.endowment.current_basket.cash > 0
            and self.endowment.current_basket.get_good_quantity(good_name) < self.value_schedules[good_name].num_units
        )

    def _can_generate_ask(self, good_name: str) -> bool:
        return (
            self.is_seller(good_name)
            and self.endowment.current_basket.get_good_quantity(good_name) > 0
        )

    def _calculate_bid_price(self, good_name: str) -> Optional[float]:
        schedule = self.value_schedules[good_name]
        current_quantity = self.endowment.current_basket.get_good_quantity(good_name)
        next_unit = current_quantity + 1  # The next unit to buy
        if next_unit > schedule.num_units:
            return None  # Already bought maximum desired units
        max_bid = min(self.endowment.current_basket.cash, schedule.get_value(int(next_unit)))
        # Bias towards max_bid
        price = random.uniform(max_bid * (1 - self.max_relative_spread / 2), max_bid)
        return price

    def _calculate_ask_price(self, good_name: str) -> Optional[float]:
        schedule = self.cost_schedules[good_name]
        initial_quantity = self.endowment.initial_basket.get_good_quantity(good_name)
        current_quantity = self.endowment.current_basket.get_good_quantity(good_name)
        units_sold = initial_quantity - current_quantity
        next_unit = units_sold + 1  # The next unit to be sold
        if next_unit > schedule.num_units:
            return None  # No more units to sell
        min_ask = schedule.get_value(int(next_unit))
        # Bias towards min_ask
        price = random.uniform(min_ask, min_ask * (1 + self.max_relative_spread / 2))
        return price

    def print_status(self):
        print(f"\nAgent ID: {self.id}")
        print(f"Current Endowment:")
        print(f"  Cash: {self.endowment.current_basket.cash:.2f}")
        print(f"  Goods: {self.endowment.current_basket.goods_dict}")
        current_utility = self.calculate_utility(self.endowment.current_basket)
        print(f"Current Utility: {current_utility:.2f}")
        self.calculate_individual_surplus()

def create_economic_agent(
    agent_id: str,
    goods: List[str],
    buy_goods: List[str],
    sell_goods: List[str],
    base_values: Dict[str, float],
    initial_cash: float,
    initial_goods: Dict[str, int],
    num_units: int = 20,
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
            base_value=base_values[good] * 0.7,  # Set seller base value lower than buyer base value
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

if __name__ == "__main__":
    # Set up logging for the main script
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Set random seed for reproducibility
    random.seed(42)
    
    # Define parameters for creating economic agents
    goods = ["apple", "banana"]
    base_values = {"apple": 20.0, "banana": 16.0}  # Increased base values for buyers
    initial_cash = 10000.0
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
        num_units=20,
        noise_factor=0.05,
        max_relative_spread=0.2
    )

    seller = create_economic_agent(
        agent_id="agent_2",
        goods=goods,
        buy_goods=[],
        sell_goods=["apple", "banana"],
        base_values={"apple": 15.0, "banana": 12.0},  # Lower base values for sellers
        initial_cash=0,
        initial_goods={"apple": 20, "banana": 20},
        num_units=20,
        noise_factor=0.05,
        max_relative_spread=0.2
    )

    # Print initial status
    print("Initial Status:")
    buyer.print_status()
    seller.print_status()

    # Generate bids and asks until a match is found or max attempts reached
    trade_id = 1
    max_attempts = 1000
    for good in goods:
        print(f"\nTrading {good}:")
        for attempt in range(max_attempts):
            bid = buyer.generate_bid(good)
            ask = seller.generate_ask(good)
            
            if bid and ask:
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
                    
                    buyer.process_trade(trade)
                    seller.process_trade(trade)
                    trade_id += 1
        else:
            print(f"  No successful trade for {good} after {max_attempts} attempts")

    # Print final status
    print("\nFinal Status:")
    buyer.print_status()
    seller.print_status()