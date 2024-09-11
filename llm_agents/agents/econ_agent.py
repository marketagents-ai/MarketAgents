from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, computed_field
import random
from functools import cached_property
from abc import ABC, abstractmethod

class UtilityFunction(BaseModel, ABC):
    @abstractmethod
    def calculate(self, quantity: int) -> float:
        pass

class StepUtilityFunction(UtilityFunction):
    max_goods: int

    def calculate(self, quantity: int) -> float:
        return 1 if quantity <= self.max_goods else 0

class CobbDouglasUtilityFunction(UtilityFunction):
    alpha: float

    def calculate(self, quantity: int) -> float:
        return quantity ** self.alpha

class Endowment(BaseModel):
    cash: float
    goods: int
    initial_cash: float
    initial_goods: int

class PreferenceSchedule(BaseModel):
    num_units: int
    base_value: float
    noise_factor: float = Field(default=0.1)
    is_buyer: bool

    @computed_field
    @cached_property
    def values(self) -> Dict[int, float]:
        return self._generate_values()

    def _generate_values(self) -> Dict[int, float]:
        values = {}
        current_value = self.base_value
        for i in range(1, self.num_units + 1):
            noise = random.uniform(-self.noise_factor, self.noise_factor) * current_value
            new_value = max(1, current_value + noise)
            if i > 1:
                new_value = min(new_value, values[i-1]) if self.is_buyer else max(new_value, values[i-1])
            values[i] = new_value
            current_value *= random.uniform(0.95, 1.0) if self.is_buyer else random.uniform(1.0, 1.05)
        return values

    def get_value(self, quantity: int) -> float:
        return self.values.get(quantity, 0.0)

class EconomicAgent(BaseModel):
    id: int
    is_buyer: bool
    preference_schedule: PreferenceSchedule
    endowment: Endowment
    utility_function: UtilityFunction
    max_relative_spread: float = Field(default=0.2)

    @computed_field
    @property
    def current_quantity(self) -> int:
        return self.endowment.goods if self.is_buyer else self.endowment.initial_goods - self.endowment.goods

    @computed_field
    @property
    def base_value(self) -> float:
        return self.preference_schedule.get_value(self.current_quantity + 1)

    def get_role(self) -> str:
        return "buyer" if self.is_buyer else "seller"

    def generate_bid(self, market_info: Optional[Dict] = None) -> Optional[Dict]:
        if not self._can_generate_bid():
            return None
        price = self._calculate_bid_price()
        return {"price": price, "quantity": 1}

    def generate_ask(self, market_info: Optional[Dict] = None) -> Optional[Dict]:
        if not self._can_generate_ask():
            return None
        price = self._calculate_ask_price()
        return {"price": price, "quantity": 1}

    def finalize_trade(self, trade: Dict):
        if trade['buyer_id'] == self.id:
            self._update_buyer_endowment(trade)
        elif trade['seller_id'] == self.id:
            self._update_seller_endowment(trade)

    def calculate_utility(self) -> float:
        return self.utility_function.calculate(self.endowment.goods)

    def calculate_individual_surplus(self) -> float:
        return self._calculate_buyer_surplus() if self.is_buyer else self._calculate_seller_surplus()

    def _can_generate_bid(self) -> bool:
        return self.is_buyer and self.base_value > 0 and self.endowment.cash >= self.base_value

    def _can_generate_ask(self) -> bool:
        return not self.is_buyer and self.base_value > 0

    def _calculate_bid_price(self) -> float:
        price = random.uniform(self.base_value * (1 - self.max_relative_spread), self.base_value)
        return min(price, self.endowment.cash, self.base_value)

    def _calculate_ask_price(self) -> float:
        price = random.uniform(self.base_value, self.base_value * (1 + self.max_relative_spread))
        return max(price, self.base_value)

    def _update_buyer_endowment(self, trade: Dict):
        self.endowment.cash -= trade['price']
        self.endowment.goods += trade['quantity']

    def _update_seller_endowment(self, trade: Dict):
        self.endowment.cash += trade['price']
        self.endowment.goods -= trade['quantity']

    def _calculate_buyer_surplus(self) -> float:
        goods_utility = sum(self.preference_schedule.get_value(q) for q in range(1, self.endowment.goods + 1))
        return goods_utility - (self.endowment.initial_cash - self.endowment.cash)

    def _calculate_seller_surplus(self) -> float:
        goods_cost = sum(self.preference_schedule.get_value(q) for q in range(1, self.endowment.initial_goods - self.endowment.goods + 1))
        return self.endowment.cash - self.endowment.initial_cash - goods_cost
    
    def update_state(self, observation: Dict[str, Any]):
        if 'cash' in observation:
            self.endowment.cash = observation['cash']
        if 'goods' in observation:
            self.endowment.goods = observation['goods']

def create_economic_agent(
    agent_id: int,
    is_buyer: bool,
    num_units: int,
    base_value: float,
    initial_cash: float,
    initial_goods: int,
    utility_function_type: Literal["step", "cobb-douglas"],
    noise_factor: float = 0.1,
    max_relative_spread: float = 0.2
) -> EconomicAgent:
    preference_schedule = PreferenceSchedule(
        num_units=num_units,
        base_value=base_value,
        noise_factor=noise_factor,
        is_buyer=is_buyer
    )
    endowment = Endowment(
        cash=initial_cash,
        goods=initial_goods,
        initial_cash=initial_cash,
        initial_goods=initial_goods
    )
    utility_function: UtilityFunction
    if utility_function_type == "step":
        utility_function = StepUtilityFunction(max_goods=num_units)
    else:
        utility_function = CobbDouglasUtilityFunction(alpha=0.5)
    
    return EconomicAgent(
        id=agent_id,
        is_buyer=is_buyer,
        preference_schedule=preference_schedule,
        endowment=endowment,
        utility_function=utility_function,
        max_relative_spread=max_relative_spread
    )
