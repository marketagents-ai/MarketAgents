import os
import logging
from pydantic import BaseModel, Field, computed_field, model_validator
from typing import List, Tuple, Optional
from functools import cached_property
import matplotlib.pyplot as plt

from ziagents import ZIAgent, create_zi_agent

# Set up logger
logger = logging.getLogger(__name__)

class CurvePoint(BaseModel):
    quantity: float
    price: float

class BaseCurve(BaseModel):
    points: List[CurvePoint]

    def get_x_y_values(self) -> Tuple[List[float], List[float]]:
        x_values = []
        y_values = []
        for point in self.points:
            x_values.extend([point.quantity - 1, point.quantity])
            y_values.extend([point.price, point.price])
        return x_values, y_values

class InitialDemandCurve(BaseCurve):
    pass
    # @model_validator(mode='after')
    # def validate_monotonicity(self):
    #     sorted_points = sorted(self.points, key=lambda p: p.quantity)
    #     for i in range(1, len(sorted_points)):
    #         if sorted_points[i].price > sorted_points[i-1].price:
    #             #plot the curve
    #             plt.plot(sorted_points[i].quantity, sorted_points[i].price, 'ro')
    #             plt.plot(sorted_points[i-1].quantity, sorted_points[i-1].price, 'bo')
    #             plt.show()
    #             raise ValueError("Initial demand curve must be monotonically decreasing")
    #     return self

class InitialSupplyCurve(BaseCurve):
    pass
    # @model_validator(mode='after')
    # def validate_monotonicity(self):
    #     sorted_points = sorted(self.points, key=lambda p: p.quantity)
    #     for i in range(1, len(sorted_points)):
    #         if sorted_points[i].price < sorted_points[i-1].price:
    #             raise ValueError("Initial supply curve must be monotonically increasing")
    #     return self

class Environment(BaseModel):
    agents: List[ZIAgent]

    @cached_property
    def buyers(self) -> List[ZIAgent]:
        return [agent for agent in self.agents if agent.is_buyer]

    @cached_property
    def sellers(self) -> List[ZIAgent]:
        return [agent for agent in self.agents if not agent.is_buyer]

    @computed_field
    @cached_property
    def initial_demand_curve(self) -> InitialDemandCurve:
        return self._generate_initial_demand_curve()

    @computed_field
    @cached_property
    def initial_supply_curve(self) -> InitialSupplyCurve:
        return self._generate_initial_supply_curve()

    @computed_field
    @property
    def current_demand_curve(self) -> BaseCurve:
        return self._generate_current_demand_curve()

    @computed_field
    @property
    def current_supply_curve(self) -> BaseCurve:
        return self._generate_current_supply_curve()

    def _generate_initial_demand_curve(self) -> InitialDemandCurve:
        points = []
        for buyer in self.buyers:
            for quantity, value in buyer.preference_schedule.values.items():
                points.append(CurvePoint(quantity=quantity, price=value))
        points.sort(key=lambda p: (-p.quantity, -p.price))
        return InitialDemandCurve(points=points)

    def _generate_initial_supply_curve(self) -> InitialSupplyCurve:
        points = []
        for seller in self.sellers:
            for quantity, cost in seller.preference_schedule.values.items():
                points.append(CurvePoint(quantity=quantity, price=cost))
        points.sort(key=lambda p: (p.quantity, p.price))
        return InitialSupplyCurve(points=points)

    def _generate_current_demand_curve(self) -> BaseCurve:
        points = []
        for buyer in self.buyers:
            for quantity, value in buyer.preference_schedule.values.items():
                if buyer.allocation.goods < quantity:
                    points.append(CurvePoint(quantity=quantity, price=value))
        points.sort(key=lambda p: p.quantity)
        return BaseCurve(points=points)

    def _generate_current_supply_curve(self) -> BaseCurve:
        points = []
        for seller in self.sellers:
            for quantity, cost in seller.preference_schedule.values.items():
                if seller.allocation.goods >= quantity:
                    points.append(CurvePoint(quantity=quantity, price=cost))
        points.sort(key=lambda p: p.quantity)
        return BaseCurve(points=points)

    @computed_field
    @property
    def remaining_trade_opportunities(self) -> int:
        potential_trades = 0
        for buyer in self.buyers:
            for seller in self.sellers:
                if buyer.allocation.cash > 0 and seller.allocation.goods > 0:
                    buyer_value = buyer.preference_schedule.get_value(buyer.allocation.goods + 1)
                    seller_cost = seller.preference_schedule.get_value(seller.allocation.goods)
                    if buyer_value > seller_cost and buyer.allocation.cash >= seller_cost:
                        potential_trades += 1
        return potential_trades

    @computed_field
    @property
    def remaining_surplus(self) -> float:
        remaining_surplus = 0.0
        for buyer in self.buyers:
            for seller in self.sellers:
                if buyer.allocation.cash > 0 and seller.allocation.goods > 0:
                    buyer_value = buyer.preference_schedule.get_value(buyer.allocation.goods + 1)
                    seller_cost = seller.preference_schedule.get_value(seller.allocation.goods)
                    if buyer_value > seller_cost:
                        remaining_surplus += (buyer_value - seller_cost)
        return remaining_surplus

    @computed_field
    @property
    def total_utility(self) -> float:
        return sum(agent.individual_surplus for agent in self.agents)

    @computed_field
    @property
    def ce_price(self) -> float:
        return self.calculate_equilibrium(initial=False)[0]

    @computed_field
    @property
    def ce_quantity(self) -> float:
        return self.calculate_equilibrium(initial=False)[1]

    @computed_field
    @property
    def ce_buyer_surplus(self) -> float:
        return self.calculate_equilibrium(initial=False)[2]

    @computed_field
    @property
    def ce_seller_surplus(self) -> float:
        return self.calculate_equilibrium(initial=False)[3]

    @computed_field
    @property
    def ce_total_surplus(self) -> float:
        return self.ce_buyer_surplus + self.ce_seller_surplus

    @computed_field
    @property
    def efficiency(self) -> float:
        extracted_surplus = self.total_utility
        theoretical_surplus = self.ce_total_surplus
        if theoretical_surplus <= 0:
            raise ValueError("Theoretical surplus is zero or negative")
        efficiency = extracted_surplus / theoretical_surplus
        if efficiency < 0:
            raise ValueError("Negative efficiency detected")
        return efficiency

    def get_agent(self, agent_id: int) -> Optional[ZIAgent]:
        """Retrieve an agent by their ID."""
        return next((agent for agent in self.agents if agent.id == agent_id), None)

    def print_market_state(self):
        logger.info("Market State:")
        for agent in self.agents:
            role = "Buyer" if agent.is_buyer else "Seller"
            logger.info(f"Agent {agent.id} ({role}):")
            logger.info(f"  Goods: {agent.allocation.goods}")
            logger.info(f"  Cash: {agent.allocation.cash:.2f}")
            logger.info(f"  Utility: {agent.individual_surplus:.2f}")
        logger.info(f"Total Market Utility: {self.total_utility:.2f}")
        logger.info(f"Remaining Trade Opportunities: {self.remaining_trade_opportunities}")
        logger.info(f"Remaining Surplus: {self.remaining_surplus:.2f}")
        logger.info(f"Market Efficiency: {self.efficiency:.2%}")

    def calculate_equilibrium(self, initial: bool = True) -> Tuple[float, float, float, float, float]:
        demand_curve = self.initial_demand_curve if initial else self.current_demand_curve
        supply_curve = self.initial_supply_curve if initial else self.current_supply_curve

        demand_x, demand_y = demand_curve.get_x_y_values()
        supply_x, supply_y = supply_curve.get_x_y_values()

        ce_quantity = 0
        ce_price = 0
        for i in range(0, min(len(demand_x), len(supply_x)), 2):
            if demand_y[i] >= supply_y[i]:
                ce_quantity = i // 2 + 1
                ce_price = (demand_y[i] + supply_y[i]) / 2
            else:
                break

        buyer_surplus = sum(max(demand_y[i] - ce_price, 0) for i in range(0, ce_quantity * 2, 2))
        seller_surplus = sum(max(ce_price - supply_y[i], 0) for i in range(0, ce_quantity * 2, 2))
        total_surplus = buyer_surplus + seller_surplus

        return ce_price, ce_quantity, buyer_surplus, seller_surplus, total_surplus

    def plot_supply_demand_curves(self, initial: bool = True, save_location: Optional[str] = None):
        demand_curve = self.initial_demand_curve if initial else self.current_demand_curve
        supply_curve = self.initial_supply_curve if initial else self.current_supply_curve

        demand_x, demand_y = demand_curve.get_x_y_values()
        supply_x, supply_y = supply_curve.get_x_y_values()

        ce_price, ce_quantity, buyer_surplus, seller_surplus, total_surplus = self.calculate_equilibrium(initial)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.step(demand_x, demand_y, where='post', label='Demand', color='blue', linestyle='--')
        ax.step(supply_x, supply_y, where='post', label='Supply', color='red', linestyle='--')
        
        max_x = max(max(demand_x), max(supply_x))
        min_y = min(min(demand_y), min(supply_y))
        max_y = max(max(demand_y), max(supply_y))
        
        ax.set_xlim(0, max_x)
        ax.set_ylim(min_y * 0.9, max_y * 1.1)

        ax.axvline(x=ce_quantity, color='green', linestyle='--', label=f'CE Quantity: {ce_quantity}')
        ax.axhline(y=ce_price, color='purple', linestyle='--', label=f'CE Price: {ce_price:.2f}')

        ax.set_xlabel('Quantity')
        ax.set_ylabel('Price')
        ax.set_title(f"{'Initial' if initial else 'Current'} Supply and Demand Curves with CE")
        ax.legend()
        ax.grid(True)

        if save_location:
            file_name = "initial_supply_demand.png" if initial else "current_supply_demand.png"
            file_path = os.path.join(save_location, file_name)
            fig.savefig(file_path)
            logger.info(f"Plot saved to {file_path}")
        
        return fig

def generate_market_agents(num_agents: int, num_units: int, buyer_base_value: int, seller_base_value: int, spread: float) -> List[ZIAgent]:
    agents = []
    for i in range(num_agents):
        is_buyer = i < num_agents // 2
        base_value = buyer_base_value if is_buyer else seller_base_value
        
        agent = create_zi_agent(
            agent_id=i,
            is_buyer=is_buyer,
            num_units=num_units,
            base_value=base_value,
            initial_cash=1000 if is_buyer else 0,
            initial_goods=0 if is_buyer else num_units,
            max_relative_spread=spread
        )
        agents.append(agent)
    
    return agents

if __name__ == "__main__":
    # Generate test agents
    num_buyers = 5
    num_sellers = 5
    spread = 0.5

    agents = generate_market_agents(
        num_agents=num_buyers + num_sellers, 
        num_units=5, 
        buyer_base_value=100, 
        seller_base_value=80, 
        spread=spread
    )
    
    # Create the environment
    env = Environment(agents=agents)

    # Print initial market state
    env.print_market_state()

    # Plot initial supply and demand curves
    env.plot_supply_demand_curves(initial=True, save_location=".")

    # Simulate some trades (this is where you'd normally run your auction)
    # For demonstration, let's just modify some agent allocations
    for i in range(3):  # Simulate 3 trades
        buyer = env.buyers[i]
        seller = env.sellers[i]
        trade_price = (buyer.base_value + seller.base_value) / 2
        buyer.allocation.goods += 1
        buyer.allocation.cash -= trade_price
        seller.allocation.goods -= 1
        seller.allocation.cash += trade_price

    # Print final market state
    print("\nAfter simulated trades:")
    env.print_market_state()

    # Plot current supply and demand curves
    env.plot_supply_demand_curves(initial=False, save_location=".")

    # Print equilibrium values
    ce_price, ce_quantity, ce_buyer_surplus, ce_seller_surplus, ce_total_surplus = env.calculate_equilibrium(initial=False)
    print(f"\nCompetitive Equilibrium:")
    print(f"Price: {ce_price:.2f}")
    print(f"Quantity: {ce_quantity}")
    print(f"Buyer Surplus: {ce_buyer_surplus:.2f}")
    print(f"Seller Surplus: {ce_seller_surplus:.2f}")
    print(f"Total Surplus: {ce_total_surplus:.2f}")
    print(f"Market Efficiency: {env.efficiency:.2%}")