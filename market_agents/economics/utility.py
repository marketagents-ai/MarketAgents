# utility.py

from typing import Dict, List, Union, Literal
from pydantic import BaseModel, Field, computed_field
from functools import cached_property
from abc import ABC, abstractmethod
import random
from market_agents.economics.econ_models import Basket, Good




class UtilityFunction(BaseModel, ABC):
    @abstractmethod
    def evaluate(self, basket: Basket) -> float:
        pass

    @abstractmethod
    def marginal_utility(self, basket: Basket, good_name: str) -> float:
        pass

class CostFunction(BaseModel, ABC):
    @abstractmethod
    def evaluate(self, basket: Basket) -> float:
        pass

    @abstractmethod
    def marginal_cost(self, basket: Basket, good_name: str) -> float:
        pass

class StepwiseUtility(UtilityFunction):
    values: Dict[str, Dict[int, float]]

    def evaluate(self, basket: Basket) -> float:
        total_utility = 0
        for good in basket.goods:
            good_values = self.values.get(good.name, {})
            total_utility += sum(good_values.get(i, 0) for i in range(1, int(good.quantity) + 1))
        return total_utility

    def marginal_utility(self, basket: Basket, good_name: str) -> float:
        good_quantity = next((g.quantity for g in basket.goods if g.name == good_name), 0)
        return self.values.get(good_name, {}).get(int(good_quantity) + 1, 0)

class CobbDouglasUtility(UtilityFunction):
    alphas: Dict[str, float]
    scale: float = 100

    def evaluate(self, basket: Basket) -> float:
        utility = 1
        for good in basket.goods:
            if good.name in self.alphas:
                utility *= (good.quantity + 1) ** self.alphas[good.name]  # Add 1 to avoid zero utility
        return self.scale * utility * (basket.cash + 1) ** (1 - sum(self.alphas.values()))  # Include cash in utility

    def marginal_utility(self, basket: Basket, good_name: str) -> float:
        if good_name == "cash":
            return self.scale * (1 - sum(self.alphas.values())) * self.evaluate(basket) / (basket.cash + 1)
        if good_name not in self.alphas:
            return 0
        good_quantity = next((g.quantity for g in basket.goods if g.name == good_name), 0)
        return self.scale * self.alphas[good_name] * self.evaluate(basket) / (good_quantity + 1)

class InducedUtility(UtilityFunction):
    goods_utility: UtilityFunction
    cash_weight: float = 1.0

    def evaluate(self, basket: Basket) -> float:
        return self.cash_weight * basket.cash + self.goods_utility.evaluate(basket)

    def marginal_utility(self, basket: Basket, good_name: str) -> float:
        if good_name == "cash":
            return self.cash_weight
        return self.goods_utility.marginal_utility(basket, good_name)

class StepwiseCost(CostFunction):
    costs: Dict[str, Dict[int, float]]
    scale: float = 1.0

    def evaluate(self, basket: Basket) -> float:
        total_cost = 0
        for good in basket.goods:
            good_costs = self.costs.get(good.name, {})
            total_cost += sum(good_costs.get(i, 0) for i in range(1, int(good.quantity) + 1))
        return self.scale * total_cost

    def marginal_cost(self, basket: Basket, good_name: str) -> float:
        good_quantity = next((g.quantity for g in basket.goods if g.name == good_name), 0)
        return self.costs.get(good_name, {}).get(int(good_quantity) + 1, 0)

class QuadraticCost(CostFunction):
    parameters: Dict[str, Dict[str, float]]
    scale: float = 1.0

    def evaluate(self, basket: Basket) -> float:
        total_cost = 0
        for good in basket.goods:
            if good.name in self.parameters:
                a, b, c = self.parameters[good.name].values()
                total_cost += a * good.quantity**2 + b * good.quantity + c
        return self.scale * total_cost

    def marginal_cost(self, basket: Basket, good_name: str) -> float:
        if good_name not in self.parameters:
            return 0
        good_quantity = next((g.quantity for g in basket.goods if g.name == good_name), 0)
        a, b, _ = self.parameters[good_name].values()
        return self.scale * (2 * a * good_quantity + b)

def create_stepwise_values(base_value: float, num_units: int, is_decreasing: bool = True, noise_factor: float = 0.1) -> Dict[int, float]:
    values = {}
    current_value = base_value
    for i in range(1, num_units + 1):
        noise = random.uniform(-noise_factor, noise_factor) * current_value
        new_value = max(1, current_value + noise)
        if i > 1:
            if is_decreasing:
                new_value = min(new_value, values[i-1])
            else:
                new_value = max(new_value, values[i-1])
        values[i] = round(new_value, 2)
        if is_decreasing:
            current_value *= random.uniform(0.95, 1.0)
        else:
            current_value *= random.uniform(1.0, 1.05)
    return values

def create_utility_function(
    function_type: Literal["stepwise", "cobb-douglas"],
    goods: List[str],
    base_values: Dict[str, float],
    num_units: int = 10,
    noise_factor: float = 0.1,
    cash_weight: float = 1.0,
    cobb_douglas_scale: float = 100  # Add this parameter
) -> UtilityFunction:
    if function_type == "stepwise":
        values = {good: create_stepwise_values(base_values[good], num_units, is_decreasing=True, noise_factor=noise_factor) for good in goods}
        goods_utility = StepwiseUtility(values=values)
    elif function_type == "cobb-douglas":
        total_alpha = sum(base_values.values())
        alphas = {good: base_values[good] / total_alpha for good in goods}
        goods_utility = CobbDouglasUtility(alphas=alphas, scale=cobb_douglas_scale)  # Include scale here
    else:
        raise ValueError(f"Unknown utility function type: {function_type}")
    
    return InducedUtility(goods_utility=goods_utility, cash_weight=cash_weight)

def create_cost_function(
    function_type: Literal["stepwise", "quadratic"],
    goods: List[str],
    base_costs: Dict[str, float],
    num_units: int = 10,
    noise_factor: float = 0.1,
    cost_scale: float = 1.0  # Add this parameter
) -> CostFunction:
    if function_type == "stepwise":
        costs = {good: create_stepwise_values(base_costs[good], num_units, is_decreasing=False, noise_factor=noise_factor) for good in goods}
        return StepwiseCost(costs=costs, scale=cost_scale)
    elif function_type == "quadratic":
        parameters = {good: {"a": 0.1, "b": base_costs[good], "c": 10} for good in goods}
        return QuadraticCost(parameters=parameters, scale=cost_scale)
    else:
        raise ValueError(f"Unknown cost function type: {function_type}")

# Example usage
if __name__ == "__main__":
    goods = ["apple", "banana"]
    base_values : Dict[str, float] = {"apple": 10, "banana": 8}
    base_costs : Dict[str, float] = {"apple": 5, "banana": 4}

    utility_function = create_utility_function("stepwise", goods, base_values, cash_weight=1.0)
    cost_function = create_cost_function("quadratic", goods, base_costs)

    basket1 = Basket(cash=100, goods=[Good(name="apple", quantity=3), Good(name="banana", quantity=2)])
    basket2 = Basket(cash=50, goods=[Good(name="apple", quantity=3), Good(name="banana", quantity=2)])

    print(f"Utility of basket1: {utility_function.evaluate(basket1)}")
    print(f"Utility of basket2: {utility_function.evaluate(basket2)}")
    print(f"Marginal Utility of Apple: {utility_function.marginal_utility(basket1, 'apple')}")
    print(f"Marginal Utility of Cash: {utility_function.marginal_utility(basket1, 'cash')}")
    print(f"Cost: {cost_function.evaluate(basket1)}")
    print(f"Marginal Cost of Apple: {cost_function.marginal_cost(basket1, 'apple')}")