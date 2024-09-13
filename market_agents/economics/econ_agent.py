from typing import List, Dict, Optional
from pydantic import BaseModel, Field, model_validator, computed_field
import random
import logging
from functools import cached_property
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
    pending_orders: Dict[str, List[MarketAction]] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_schedules(self):
        overlapping_goods = set(self.value_schedules.keys()) & set(self.cost_schedules.keys())
        if overlapping_goods:
            raise ValueError(f"Agent cannot be both buyer and seller of the same good(s): {overlapping_goods}")
        return self
    
    @computed_field
    @property
    def initial_utility(self) -> float:
        utility = self.endowment.initial_basket.cash
        for good, quantity in self.endowment.initial_basket.goods_dict.items():
            if self.is_buyer(good):
                # Buyers start with cash and no goods
                pass
            elif self.is_seller(good):
                schedule = self.cost_schedules[good]
                initial_quantity = int(quantity)
                initial_cost = sum(schedule.get_value(q) for q in range(1, initial_quantity + 1))
                utility += initial_cost  # Add total cost of initial inventory
        return utility

    @computed_field
    @property
    def current_utility(self) -> float:
        return self.calculate_utility(self.endowment.current_basket)

    @computed_field
    @property
    def current_cash(self) -> float:
        return self.endowment.current_basket.cash
    
    @computed_field
    @property
    def pending_cash(self) -> float:
        return sum(order.price * order.quantity for orders in self.pending_orders.values() for order in orders if isinstance(order, Bid))
    
    @computed_field
    @property
    def available_cash(self) -> float:
        return self.current_cash - self.pending_cash
    
    
    def get_pending_bid_quantity(self, good_name: str) -> int:
        """ returns the quantity of the good that the agent has pending orders for"""
        return sum(order.quantity for order in self.pending_orders.get(good_name, []) if isinstance(order, Bid))
    
    def get_pending_ask_quantity(self, good_name: str) -> int:
        """ returns the quantity of the good that the agent has pending orders for"""
        return sum(order.quantity for order in self.pending_orders.get(good_name, []) if isinstance(order, Ask))
    
    def get_current_value(self, good_name: str) -> Optional[float]:
        """ returns the current value of the next unit of the good  for buyers only
        it takes in consideration both the current inventory and the pending orders"""
        if not self.is_buyer(good_name):
            return None
        current_quantity = self.endowment.current_basket.get_good_quantity(good_name)
        pending_quantity = self.get_pending_bid_quantity(good_name)
        total_quantity = int(current_quantity + pending_quantity)
        if total_quantity > self.value_schedules[good_name].num_units:
            return None
        return self.value_schedules[good_name].get_value(total_quantity+1)
    
    def get_current_cost(self, good_name: str) -> Optional[float]:
        """ returns the current cost of the good for sellers only
        it takes in consideration both the current inventory and the pending orders"""
        if not self.is_seller(good_name):
            return None
        starting_quantity = self.endowment.initial_basket.get_good_quantity(good_name)
        current_quantity = self.endowment.current_basket.get_good_quantity(good_name)
        pending_quantity = self.get_pending_ask_quantity(good_name)
        total_quantity = int(starting_quantity - current_quantity + pending_quantity)+1

        return self.cost_schedules[good_name].get_value(total_quantity)

    def is_buyer(self, good_name: str) -> bool:
        return good_name in self.value_schedules

    def is_seller(self, good_name: str) -> bool:
        return good_name in self.cost_schedules

    def generate_bid(self, good_name: str) -> Optional[Bid]:
        if not self._can_generate_bid(good_name):
            return None
        price = self._calculate_bid_price(good_name)
        if price is not None:
            bid = Bid(price=price, quantity=1)
            # Update pending orders
            self.pending_orders.setdefault(good_name, []).append(bid)
            return bid
        else:
            return None

    def generate_ask(self, good_name: str) -> Optional[Ask]:
        if not self._can_generate_ask(good_name):
            return None
        price = self._calculate_ask_price(good_name)
        if price is not None:
            ask = Ask(price=price, quantity=1)
            # Update pending orders
            self.pending_orders.setdefault(good_name, []).append(ask)
            return ask
        else:
            return None
    
    def would_accept_trade(self, trade: Trade) -> bool:
        if self.is_buyer(trade.good_name) and trade.buyer_id == self.id:
            marginal_value = self.get_current_value(trade.good_name)
            if marginal_value is None:
                print("trade rejected because marginal_value is None")
                return False
            print(f"marginal_value: {marginal_value}, trade.price: {trade.price}")
            return marginal_value >= trade.price
        elif self.is_seller(trade.good_name) and trade.seller_id == self.id:
            marginal_cost = self.get_current_cost(trade.good_name)
            if marginal_cost is None:
                print("trade rejected because marginal_cost is None")
                return False
            print(f"marginal_cost: {marginal_cost}, trade.price: {trade.price}")
            return trade.price >= marginal_cost
        else:
            self_type = "buyer" if self.is_buyer(trade.good_name) else "seller"
            print(f"trade rejected because it's not for the agent {self.id} vs trade.buyer_id: {trade.buyer_id}, trade.seller_id: {trade.seller_id} with type {self_type}")
            return False


    def process_trade(self, trade: Trade):
        if self.is_buyer(trade.good_name) and trade.buyer_id == self.id:
            # Find the exactly matching bid from pending_orders
            bids = self.pending_orders.get(trade.good_name, [])
            matching_bid = next((bid for bid in bids if bid.quantity == trade.quantity and bid.price == trade.bid_price), None)
            if matching_bid:
                bids.remove(matching_bid)
                if not bids:
                    del self.pending_orders[trade.good_name]
            else:
                raise ValueError(f"Trade {trade.trade_id} processed but matching bid not found for agent {self.id}")
        elif self.is_seller(trade.good_name) and trade.seller_id == self.id:
            # Find the exactly matching ask from pending_orders
            asks = self.pending_orders.get(trade.good_name, [])
            matching_ask = next((ask for ask in asks if ask.quantity == trade.quantity and ask.price == trade.ask_price), None)
            if matching_ask:
                asks.remove(matching_ask)
                if not asks:
                    del self.pending_orders[trade.good_name]
            else:
                raise ValueError(f"Trade {trade.trade_id} processed but matching ask not found for agent {self.id}")
        
        # Only update the endowment after passing the value error checks
        self.endowment.add_trade(trade)
        new_utility = self.calculate_utility(self.endowment.current_basket)
        logger.info(f"Agent {self.id} processed trade. New utility: {new_utility:.2f}")

    def calculate_utility(self, basket: Basket) -> float:
        utility = basket.cash
        
        for good, quantity in basket.goods_dict.items():
            if self.is_buyer(good):
                schedule = self.value_schedules[good]
                value_sum = sum(schedule.get_value(q) for q in range(1, int(quantity) + 1))
                utility += value_sum
            elif self.is_seller(good):
                schedule = self.cost_schedules[good]
                starting_quantity = self.endowment.initial_basket.get_good_quantity(good)
                unsold_units = int(basket.get_good_quantity(good))
                sold_units = starting_quantity - unsold_units
                # Unsold inventory should be valued at its cost, not higher
                unsold_cost = sum(schedule.get_value(q) for q in range(sold_units+1, starting_quantity + 1))

                utility += unsold_cost  # Add the cost of unsold units
        return utility



    def calculate_individual_surplus(self) -> float:
        current_utility = self.calculate_utility(self.endowment.current_basket)
        surplus = current_utility - self.initial_utility
        print(f"current_utility: {current_utility}, initial_utility: {self.initial_utility}, surplus: {surplus}")
        return surplus



    def _can_generate_bid(self, good_name: str) -> bool:
        if not self.is_buyer(good_name):
            return False

        available_cash = self.available_cash 
        if available_cash <= 0:
            return False
        current_quantity = self.endowment.current_basket.get_good_quantity(good_name)
        pending_quantity = sum(order.quantity for order in self.pending_orders.get(good_name, []) if isinstance(order, Bid))
        total_quantity = current_quantity + pending_quantity
        return total_quantity < self.value_schedules[good_name].num_units

    def _can_generate_ask(self, good_name: str) -> bool:
        if not self.is_seller(good_name):
            return False
        current_quantity = self.endowment.current_basket.get_good_quantity(good_name)
        pending_quantity = sum(order.quantity for order in self.pending_orders.get(good_name, []) if isinstance(order, Ask))
        total_quantity = current_quantity - pending_quantity
        return total_quantity > 0

    def _calculate_bid_price(self, good_name: str) -> Optional[float]:

        current_value = self.get_current_value(good_name)
        if current_value is None:
            return None
        max_bid = min(self.endowment.current_basket.cash, current_value*0.99)
        price = random.uniform(max_bid * (1 - self.max_relative_spread), max_bid)
        return price

    def _calculate_ask_price(self, good_name: str) -> Optional[float]:
        current_cost = self.get_current_cost(good_name)
        if current_cost is None:
            return None
        min_ask = current_cost * 1.01
        price = random.uniform(min_ask, min_ask * (1 + self.max_relative_spread))
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
    trade_ids = {good:0 for good in goods}
    max_attempts = 1000
    for good in goods:
        print(f"\nTrading {good}:")
        for attempt in range(max_attempts):
            buyer_value = buyer.get_current_value(good)
            seller_cost = seller.get_current_cost(good)
            if buyer_value is not None and seller_cost is not None:
                if buyer_value >= seller_cost:
                    bid = buyer.generate_bid(good)
                    ask = seller.generate_ask(good)
                    if bid and ask:
                        if bid.price >= ask.price:
                            trade_price = (bid.price + ask.price) / 2
                            trade = Trade(
                                trade_id=trade_ids[good],
                                buyer_id=buyer.id,
                                seller_id=seller.id,
                                price=trade_price,
                                quantity=1,
                                good_name=good,
                                bid_price=bid.price,
                                ask_price=ask.price
                            )
                            print(f"  Trade executed: Price good:{good}= {trade_price:.2f}, Quantity = 1")
                            
                            buyer.process_trade(trade)
                            seller.process_trade(trade)
                            trade_ids[good] += 1
                else:
                    print(f"  No match found for {good} at attempt {attempt} because buyer_value {buyer_value} < seller_cost {seller_cost}")
                    break


    # Print final status
    print("\nFinal Status:")
    buyer.print_status()
    seller.print_status()
    print("trade_ids: ", trade_ids)
    print("buyer surplus: ", buyer.calculate_individual_surplus())
    print("seller surplus: ", seller.calculate_individual_surplus())