from pydantic import BaseModel, Field, computed_field, model_validator
from functools import cached_property
from typing import List, Dict
import random
from copy import deepcopy
from datetime import datetime
import uuid

class MarketAction(BaseModel):
    price: float = Field(..., description="Price of the order")
    quantity: int = Field(default=1, ge=1,le=1, description="Quantity of the order")

class Bid(MarketAction):
    @computed_field
    @property
    def is_buyer(self) -> bool:
        return True

class Ask(MarketAction):
    @computed_field
    @property
    def is_buyer(self) -> bool:
        return False

class Trade(BaseModel):
    trade_id: int = Field(..., description="Unique identifier for the trade")
    buyer_id: str = Field(..., description="ID of the buyer")
    seller_id: str = Field(..., description="ID of the seller")
    price: float = Field(..., description="The price at which the trade was executed")
    ask_price: float = Field(ge=0, description="The price at which the ask was executed")
    bid_price: float = Field(ge=0, description="The price at which the bid was executed")
    quantity: int = Field(default=1, description="The quantity traded")
    good_name: str = Field(default="consumption_good", description="The name of the good traded")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the trade")

    @model_validator(mode='after')
    def rational_trade(self):
        if self.ask_price > self.bid_price:
            raise ValueError(f"Ask price {self.ask_price} is more than bid price {self.bid_price}")
        return self
    
class Good(BaseModel):
    name: str
    quantity: float

class Basket(BaseModel):
    cash: float
    goods: List[Good]

    @computed_field
    @cached_property
    def goods_dict(self) -> Dict[str, int]:
        return {good.name: int(good.quantity) for good in self.goods}

    def update_good(self, name: str, quantity: float):
        for good in self.goods:
            if good.name == name:
                good.quantity = quantity
                return
        self.goods.append(Good(name=name, quantity=quantity))

    def get_good_quantity(self, name: str) -> int:
        return int(next((good.quantity for good in self.goods if good.name == name), 0))

class Endowment(BaseModel):
    initial_basket: Basket
    trades: List[Trade] = Field(default_factory=list)
    agent_id: str

    @computed_field
    @property
    def current_basket(self) -> Basket:
        temp_basket = deepcopy(self.initial_basket)

        for trade in self.trades:
            if trade.buyer_id == self.agent_id:
                temp_basket.cash -= trade.price * trade.quantity
                temp_basket.update_good(trade.good_name, temp_basket.get_good_quantity(trade.good_name) + trade.quantity)
            elif trade.seller_id == self.agent_id:
                temp_basket.cash += trade.price * trade.quantity
                temp_basket.update_good(trade.good_name, temp_basket.get_good_quantity(trade.good_name) - trade.quantity)

        # Create a new Basket instance with the calculated values
        return Basket(
            cash=temp_basket.cash,
            goods=[Good(name=good.name, quantity=good.quantity) for good in temp_basket.goods]
        )

    def add_trade(self, trade: Trade):
        self.trades.append(trade)
        # Clear the cached property to ensure it's recalculated
        if 'current_basket' in self.__dict__:
            del self.__dict__['current_basket']

    def simulate_trade(self, trade: Trade) -> Basket:
        temp_basket = deepcopy(self.current_basket)

        if trade.buyer_id == self.agent_id:
            temp_basket.cash -= trade.price * trade.quantity
            temp_basket.update_good(trade.good_name, temp_basket.get_good_quantity(trade.good_name) + trade.quantity)
        elif trade.seller_id == self.agent_id:
            temp_basket.cash += trade.price * trade.quantity
            temp_basket.update_good(trade.good_name, temp_basket.get_good_quantity(trade.good_name) - trade.quantity)

        return temp_basket

class PreferenceSchedule(BaseModel):
    num_units: int = Field(..., description="Number of units")
    base_value: float = Field(..., description="Base value for the first unit")
    noise_factor: float = Field(default=0.1, description="Noise factor for value generation")
    is_buyer: bool = Field(default=True, description="Whether the agent is a buyer")

    @computed_field
    @cached_property
    def values(self) -> Dict[int, float]:
        raise NotImplementedError("Subclasses must implement this method")

    @computed_field
    @cached_property
    def initial_endowment(self) -> float:
        raise NotImplementedError("Subclasses must implement this method")

    def get_value(self, quantity: int) -> float:
        return self.values.get(quantity, 0.0)

    def plot_schedule(self, block=False):
        quantities = list(self.values.keys())
        values = list(self.values.values())
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(quantities, values, marker='o')
        plt.title(f"{'Demand' if self.is_buyer else 'Supply'} Schedule")
        plt.xlabel("Quantity")
        plt.ylabel("Value/Cost")
        plt.grid(True)
        plt.show(block=block)

class BuyerPreferenceSchedule(PreferenceSchedule):
    endowment_factor: float = Field(default=1.2, description="Factor to calculate initial endowment")
    is_buyer: bool = Field(default=True, description="Whether the agent is a buyer")

    @computed_field
    @cached_property
    def values(self) -> Dict[int, float]:
        values = {}
        current_value = self.base_value
        for i in range(1, self.num_units + 1):
            # Decrease current_value by 2% to 5% plus noise
            decrement = current_value * random.uniform(0.02, self.noise_factor)
            
            new_value = current_value-decrement
            values[i] = new_value
            current_value = new_value
        return values

    @computed_field
    @cached_property
    def initial_endowment(self) -> float:
        return sum(self.values.values()) * self.endowment_factor

class SellerPreferenceSchedule(PreferenceSchedule):
    is_buyer: bool = Field(default=False, description="Whether the agent is a buyer")
    @computed_field
    @cached_property
    def values(self) -> Dict[int, float]:
        values = {}
        current_value = self.base_value
        for i in range(1, self.num_units + 1):
            # Increase current_value by 2% to 5% plus noise
            increment = current_value * random.uniform(0.02, self.noise_factor)
            new_value = current_value+increment
            values[i] = new_value
            current_value = new_value
        return values

    @computed_field
    @cached_property
    def initial_endowment(self) -> float:
        return sum(self.values.values())