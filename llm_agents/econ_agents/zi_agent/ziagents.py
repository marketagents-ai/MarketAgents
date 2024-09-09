import random
import logging
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, computed_field, model_validator
import matplotlib.pyplot as plt
from functools import cached_property
from typing_extensions import Self

# Set up logger
logger = logging.getLogger(__name__)

class PreferenceSchedule(BaseModel):
    """Base class for preference schedules."""
    num_units: int = Field(..., description="Number of units")
    base_value: float = Field(..., description="Base value for the first unit")
    noise_factor: float = Field(default=0.1, description="Noise factor for value generation")
    is_buyer: bool = Field(default=True, description="Whether the agent is a buyer")

    @computed_field
    @cached_property
    def values(self) -> Dict[int, float]:
        """Generate and cache the values for each unit."""
        raise NotImplementedError("Subclasses must implement this method")

    @computed_field
    @cached_property
    def initial_endowment(self) -> float:
        """Calculate and cache the initial endowment."""
        raise NotImplementedError("Subclasses must implement this method")

    def get_value(self, quantity: int) -> float:
        """Get the value for a specific quantity."""
        return self.values.get(quantity, 0.0)

    def plot_schedule(self, block: bool = False) -> None:
        """Plot the preference schedule."""
        quantities = list(self.values.keys())
        values = list(self.values.values())
        
        plt.figure(figsize=(10, 6))
        plt.plot(quantities, values, marker='o')
        plt.title(f"{'Demand' if self.is_buyer else 'Supply'} Schedule")
        plt.xlabel("Quantity")
        plt.ylabel("Value/Cost")
        plt.grid(True)
        plt.show(block=block)

class BuyerPreferenceSchedule(PreferenceSchedule):
    """Preference schedule for buyers."""
    endowment_factor: float = Field(default=1.2, description="Factor to calculate initial endowment")
    is_buyer: bool = Field(default=True, description="Whether the agent is a buyer")

    @computed_field
    @cached_property
    def values(self) -> Dict[int, float]:
        """Generate and cache the values for each unit."""
        values = {}
        current_value = self.base_value
        for i in range(1, self.num_units + 1):
            noise = random.uniform(-self.noise_factor, self.noise_factor) * current_value
            new_value = max(1, current_value + noise)  # Ensure no zero values
            if i > 1:
                new_value = min(new_value, values[i-1])  # Ensure monotonicity
            values[i] = new_value
            current_value *= random.uniform(0.95, 1.0)
        return values

    @computed_field
    @cached_property
    def initial_endowment(self) -> float:
        """Calculate and cache the initial endowment."""
        return sum(self.values.values()) * self.endowment_factor

class SellerPreferenceSchedule(PreferenceSchedule):
    """Preference schedule for sellers."""
    is_buyer: bool = Field(default=False, description="Whether the agent is a buyer")

    @computed_field
    @cached_property
    def values(self) -> Dict[int, float]:
        """Generate and cache the values for each unit."""
        values = {}
        current_value = self.base_value
        for i in range(1, self.num_units + 1):
            noise = random.uniform(-self.noise_factor, self.noise_factor) * current_value
            new_value = max(1, current_value + noise)  # Ensure no zero values
            if i > 1:
                new_value = max(new_value, values[i-1])  # Ensure monotonicity
            values[i] = new_value
            current_value *= random.uniform(1.0, 1.05)
        return values

    @computed_field
    @cached_property
    def initial_endowment(self) -> float:
        """Calculate and cache the initial endowment."""
        return sum(self.values.values())
    
class MarketAction(BaseModel):
    """Represents a market action (bid or ask)."""
    price: float = Field(..., description="Price of the order")
    quantity: int = Field(default=1, ge=1, le=1, description="Quantity of the order, constrained to 1")

class Order(BaseModel):
    """Base class for orders."""
    agent_id: int = Field(..., description="Unique identifier of the agent placing the order")
    market_action: MarketAction = Field(..., description="Market action of the order")

    @computed_field
    @cached_property
    def is_buy(self) -> bool:
        """Determine if the order is a buy order."""
        return isinstance(self, Bid)

class Bid(Order):
    """Represents a bid order."""
    base_value: float = Field(..., description="Base value of the item for the buyer")
    
    @model_validator(mode='after')
    def check_bid_validity(self) -> Self:
        """Validate that the bid price is not higher than the base value."""
        if self.market_action.price > self.base_value:
            raise ValueError("Bid price is higher than base value")
        return self

class Ask(Order):
    """Represents an ask order."""
    base_cost: float = Field(..., description="Base cost of the item for the seller")

    @model_validator(mode='after')
    def check_ask_validity(self) -> Self:
        """Validate that the ask price is not lower than the base cost."""
        if self.market_action.price < self.base_cost:
            raise ValueError("Ask price is lower than base cost")
        return self
    
class IllegalOrder(Order):
    """Represents an illegal order."""
    base_value: Optional[float] = Field(None, description="Base value of the item for the buyer")
    base_cost: Optional[float] = Field(None, description="Base cost of the item for the seller")

class Trade(BaseModel):
    """Represents a completed trade."""
    trade_id: int = Field(..., description="Unique identifier for the trade")
    bid: Bid = Field(..., description="The bid involved in the trade")
    ask: Ask = Field(..., description="The ask involved in the trade")
    price: float = Field(..., description="The price at which the trade was executed")
    round: int = Field(..., description="The round in which the trade occurred")
    quantity: int = 1

    @computed_field
    @property
    def buyer_surplus(self) -> float:
        """Calculate the buyer's surplus from the trade."""
        return self.bid.base_value - self.price

    @computed_field
    @property
    def seller_surplus(self) -> float:
        """Calculate the seller's surplus from the trade."""
        return self.price - self.ask.base_cost

    @computed_field
    @property
    def total_surplus(self) -> float:
        """Calculate the total surplus from the trade."""
        return self.buyer_surplus + self.seller_surplus

    @model_validator(mode='after')
    def check_trade_validity(self) -> Self:
        """Validate that the bid price is not lower than the ask price."""
        if self.bid.market_action.price < self.ask.market_action.price:
            raise ValueError("Bid price is lower than Ask price")
        return self

class MarketInfo(BaseModel):
    """Represents current market information."""
    last_trade_price: Optional[float] = Field(None, description="Price of the last executed trade")
    average_price: Optional[float] = Field(None, description="Average price of all executed trades")
    total_trades: int = Field(0, description="Total number of executed trades")
    current_round: int = Field(1, description="Current round number")

class Allocation(BaseModel):
    """Represents an agent's current allocation."""
    goods: int = Field(default=0, description="Quantity of goods")
    cash: float = Field(default=0.0, description="Amount of cash")
    initial_goods: int = Field(default=0, description="Initial quantity of goods")
    initial_cash: float = Field(default=0.0, description="Initial amount of cash")

class AgentHistory(BaseModel):
    """Represents an agent's trading history."""
    active_orders: List[Union[Bid, Ask]] = Field(default_factory=list, description="List of active orders")
    illegal_orders: List[IllegalOrder] = Field(default_factory=list, description="List of illegal orders")
    accepted_trades: List[Trade] = Field(default_factory=list, description="List of accepted trades")
    expired_orders: List[Union[Bid, Ask]] = Field(default_factory=list, description="List of expired orders")

class ZIAgent(BaseModel):
    """Represents a Zero Intelligence agent."""
    id: int = Field(..., description="Unique identifier for the agent")
    preference_schedule: PreferenceSchedule = Field(..., description="Preference schedule of the agent")
    allocation: Allocation = Field(..., description="Current allocation of goods and cash")
    max_relative_spread: float = Field(default=0.2, description="Maximum relative price spread")
    history: AgentHistory = Field(default_factory=AgentHistory, description="History of agent's actions")

    @computed_field
    @cached_property
    def is_buyer(self) -> bool:
        """Determine if the agent is a buyer."""
        return self.preference_schedule.is_buyer

    @computed_field
    @property
    def current_quantity(self) -> int:
        """Get the current quantity of goods held by the agent."""
        return self.allocation.goods if self.is_buyer else self.allocation.initial_goods - self.allocation.goods

    @computed_field
    @property
    def base_value(self) -> float:
        """Get the base value for the next unit."""
        return self.preference_schedule.get_value(self.current_quantity + 1)
    
    def generate_bid_market_action(self, market_info: Optional[MarketInfo] = None) -> Optional[MarketAction]:
        """Generate a bid market action."""
        return self._generate_bid_market_action_zi(market_info)
    
    def generate_ask_market_action(self, market_info: Optional[MarketInfo] = None) -> Optional[MarketAction]:
        """Generate an ask market action."""
        return self._generate_ask_market_action_zi(market_info)
    
    def _generate_bid_market_action_zi(self, market_info: Optional[MarketInfo] = None) -> Optional[MarketAction]:
        """Generate a Zero Intelligence bid market action."""
        if not self.is_buyer or self.base_value <= 0 or self.allocation.cash < self.base_value:
            return None
        price = random.uniform(self.base_value * (1 - self.max_relative_spread), self.base_value)
        price = min(price, self.allocation.cash, self.base_value)
        return MarketAction(price=price, quantity=1)

    def _generate_ask_market_action_zi(self, market_info: Optional[MarketInfo] = None) -> Optional[MarketAction]:
        """Generate a Zero Intelligence ask market action."""
        if self.is_buyer or self.base_value <= 0:
            return None
        price = random.uniform(self.base_value, self.base_value * (1 + self.max_relative_spread))
        price = max(price, self.base_value)
        return MarketAction(price=price, quantity=1)

    def generate_bid(self, market_info: Optional[MarketInfo] = None) -> Optional[Bid]:
        """Generate a bid order."""
        if not self.is_buyer or self.base_value <= 0 or self.allocation.cash < self.base_value:
            return None
        market_action = self.generate_bid_market_action(market_info)
        if market_action is None:
            return None
        try:
            return Bid(agent_id=self.id, market_action=market_action, base_value=self.base_value)
        except ValueError as e:
            logger.warning(f"Agent {self.id} generated an invalid bid: {e}")
            illegal_order = IllegalOrder(agent_id=self.id, market_action=market_action, base_value=self.base_value, base_cost=None)
            self.add_illegal_order(illegal_order)
            return None

    def generate_ask(self, market_info: Optional[MarketInfo] = None) -> Optional[Ask]:
        """Generate an ask order."""
        if self.is_buyer or self.base_value <= 0:
            return None
        market_action = self.generate_ask_market_action(market_info)
        if market_action is None:
            return None
        try:
            return Ask(agent_id=self.id, market_action=market_action, base_cost=self.base_value)
        except ValueError as e:
            logger.warning(f"Agent {self.id} generated an invalid ask: {e}")
            illegal_order = IllegalOrder(agent_id=self.id, market_action=market_action, base_value=None, base_cost=self.base_value)
            self.add_illegal_order(illegal_order)
            return None

    def finalize_bid(self, trade: Trade) -> None:
        """Finalize a bid trade."""
        if trade.bid.agent_id != self.id:
            raise ValueError(f"Agent {self.id} is not the buyer in this trade")
        self.allocation.cash -= trade.price
        self.allocation.goods += 1
        self.history.accepted_trades.append(trade)

    def finalize_ask(self, trade: Trade) -> None:
        """Finalize an ask trade."""
        if trade.ask.agent_id != self.id:
            raise ValueError(f"Agent {self.id} is not the seller in this trade")
        self.allocation.cash += trade.price
        self.allocation.goods -= 1
        self.history.accepted_trades.append(trade)

    def add_active_order(self, order: Union[Bid, Ask]) -> None:
        """Add an active order to the agent's history."""
        self.history.active_orders.append(order)

    def add_illegal_order(self, order: IllegalOrder) -> None:
        """Add an illegal order to the agent's history."""
        self.history.illegal_orders.append(order)

    def expire_order(self, order: Union[Bid, Ask]) -> None:
        """Expire an order and move it to the expired orders list."""
        self.history.active_orders.remove(order)
        self.history.expired_orders.append(order)

    @computed_field
    @property
    def individual_surplus(self) -> float:
        """Calculate the individual surplus for the agent."""
        if self.is_buyer:
            goods_utility = sum(self.preference_schedule.get_value(q) for q in range(1, self.allocation.goods + 1))
            return goods_utility - (self.allocation.initial_cash - self.allocation.cash)
        else:
            goods_cost = sum(self.preference_schedule.get_value(q) for q in range(1, self.allocation.initial_goods - self.allocation.goods + 1))
            return self.allocation.cash - self.allocation.initial_cash - goods_cost

def create_preference_schedule(
    is_buyer: bool,
    num_units: int = 10,
    base_value: float = 100,
    noise_factor: float = 0.1,
    endowment_factor: float = 1.2
) -> PreferenceSchedule:
    """Create a preference schedule for an agent."""
    if is_buyer:
        return BuyerPreferenceSchedule(
            num_units=num_units,
            base_value=base_value,
            noise_factor=noise_factor,
            endowment_factor=endowment_factor
        )
    else:
        return SellerPreferenceSchedule(num_units=num_units, base_value=base_value, noise_factor=noise_factor)

def create_zi_agent(
    agent_id: int,
    is_buyer: bool,
    num_units: int,
    base_value: float,
    initial_cash: float,
    initial_goods: int,
    noise_factor: float = 0.1,
    max_relative_spread: float = 0.2
) -> ZIAgent:
    preference_schedule = create_preference_schedule(
        is_buyer=is_buyer,
        num_units=num_units,
        base_value=base_value,
        noise_factor=noise_factor
    )
    allocation = Allocation(
        cash=initial_cash,
        goods=initial_goods,
        initial_cash=initial_cash,
        initial_goods=initial_goods
    )
    return ZIAgent(
        id=agent_id,
        preference_schedule=preference_schedule,
        allocation=allocation,
        max_relative_spread=max_relative_spread
    )

# Example usage
if __name__ == "__main__":
    # Create a buyer and a seller
    buyer = create_zi_agent(agent_id=1, is_buyer=True, num_units=10, base_value=100, initial_cash=1000, initial_goods=0)
    seller = create_zi_agent(agent_id=2, is_buyer=False, num_units=10, base_value=80, initial_cash=0, initial_goods=10)

    trade_history = []

    for round in range(1, 11):  # 10 rounds of trading
        print(f"\nRound {round}")
        
        # Generate orders
        bid = buyer.generate_bid()
        ask = seller.generate_ask()

        if bid and ask:
            print(f"Bid: {bid.market_action.price:.2f}, Ask: {ask.market_action.price:.2f}")

            if bid.market_action.price >= ask.market_action.price:
                # Create a trade
                trade_price = (bid.market_action.price + ask.market_action.price) / 2
                trade = Trade(trade_id=len(trade_history) + 1, bid=bid, ask=ask, price=trade_price, round=round)

                # Finalize the trade
                buyer.finalize_bid(trade)
                seller.finalize_ask(trade)

                trade_history.append(trade)

                print(f"Trade executed at price: {trade_price:.2f}")
                print(f"Buyer surplus: {trade.buyer_surplus:.2f}")
                print(f"Seller surplus: {trade.seller_surplus:.2f}")
                print(f"Total surplus: {trade.total_surplus:.2f}")
            else:
                print("No trade: Bid price is lower than Ask price")
        else:
            print("No trade: Either Bid or Ask is None")

        # Print current states
        print(f"Buyer state: Cash={buyer.allocation.cash:.2f}, Goods={buyer.allocation.goods}")
        print(f"Seller state: Cash={seller.allocation.cash:.2f}, Goods={seller.allocation.goods}")

    # Print final states and surpluses
    print("\nFinal Results:")
    print(f"Buyer final state: Cash={buyer.allocation.cash:.2f}, Goods={buyer.allocation.goods}")
    print(f"Seller final state: Cash={seller.allocation.cash:.2f}, Goods={seller.allocation.goods}")
    print(f"Buyer individual surplus: {buyer.individual_surplus:.2f}")
    print(f"Seller individual surplus: {seller.individual_surplus:.2f}")

    # Plot trade prices
    plt.figure(figsize=(10, 6))
    plt.plot([t.round for t in trade_history], [t.price for t in trade_history], marker='o')
    plt.title("Trade Prices Over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show(block=False)

    # Plot preference schedules
    buyer.preference_schedule.plot_schedule(block=False)
    seller.preference_schedule.plot_schedule(block=False)

    # Keep the program running until plots are closed
    plt.show()