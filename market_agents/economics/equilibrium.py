from typing import List, Dict, Tuple
from pydantic import BaseModel
import matplotlib.pyplot as plt
import logging
from market_agents.economics.econ_agent import EconomicAgent, create_economic_agent

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Equilibrium(BaseModel):
    agents: List[EconomicAgent]
    goods: List[str]

    def calculate_equilibrium(self) -> Dict[str, Dict[str, float]]:
        equilibria = {}
        for good in self.goods:
            logger.info(f"Calculating equilibrium for {good}")
            demand_prices, supply_prices = self._aggregate_curves(good)
            equilibrium_price, equilibrium_quantity = self._find_intersection(demand_prices, supply_prices)
            equilibria[good] = {
                "price": equilibrium_price,
                "quantity": equilibrium_quantity,
                "buyer_surplus": self._calculate_surplus(demand_prices, equilibrium_price, equilibrium_quantity, is_buyer=True),
                "seller_surplus": self._calculate_surplus(supply_prices, equilibrium_price, equilibrium_quantity, is_buyer=False),
            }
            equilibria[good]["total_surplus"] = equilibria[good]["buyer_surplus"] + equilibria[good]["seller_surplus"]
            logger.info(f"Equilibrium for {good}: {equilibria[good]}")
        return equilibria

    def _aggregate_curves(self, good: str) -> Tuple[List[float], List[float]]:
        demand_prices = []
        supply_prices = []

        # Aggregate demand
        for agent in self.agents:
            if agent.is_buyer(good):
                schedule = agent.value_schedules[good]
                for quantity in range(1, schedule.num_units + 1):
                    value = schedule.get_value(quantity)
                    demand_prices.append(value)

        # Aggregate supply
        for agent in self.agents:
            if agent.is_seller(good):
                schedule = agent.cost_schedules[good]
                for quantity in range(1, schedule.num_units + 1):
                    cost = schedule.get_value(quantity)
                    supply_prices.append(cost)

        # Sort the marginal values and costs
        demand_prices.sort(reverse=True)
        supply_prices.sort()
        logger.debug(f"Aggregated demand prices for {good}: {demand_prices}")
        logger.debug(f"Aggregated supply prices for {good}: {supply_prices}")
        return demand_prices, supply_prices

    def _find_intersection(self, demand_prices: List[float], supply_prices: List[float]) -> Tuple[float, int]:
        # Find the quantity where demand price >= supply price
        quantity = 0
        max_quantity = min(len(demand_prices), len(supply_prices))
        for i in range(max_quantity):
            demand_price = demand_prices[i]
            supply_price = supply_prices[i]
            logger.debug(f"At quantity {i+1}: demand_price={demand_price}, supply_price={supply_price}")
            if demand_price >= supply_price:
                quantity += 1
            else:
                break
        if quantity == 0:
            logger.info("No equilibrium found")
            return 0, 0
        equilibrium_price = (demand_prices[quantity - 1] + supply_prices[quantity - 1]) / 2
        logger.info(f"Equilibrium found at price {equilibrium_price} with quantity {quantity}")
        return equilibrium_price, quantity

    def _calculate_surplus(self, prices: List[float], price: float, quantity: int, is_buyer: bool) -> float:
        surplus = 0.0
        for i in range(quantity):
            value = prices[i]
            if is_buyer:
                surplus += value - price
            else:
                surplus += price - value
        return surplus

    def plot_supply_demand(self, good: str):
        demand_prices, supply_prices = self._aggregate_curves(good)
        # Build cumulative quantities for plotting
        demand_quantities = [0] + list(range(1, len(demand_prices) + 1))
        demand_prices_plot = [demand_prices[0]] + demand_prices
        supply_quantities = [0] + list(range(1, len(supply_prices) + 1))
        supply_prices_plot = [supply_prices[0]] + supply_prices
        plt.figure(figsize=(10, 6))
        plt.step(demand_quantities, demand_prices_plot, where='post', label='Aggregate Demand', color='blue')
        plt.step(supply_quantities, supply_prices_plot, where='post', label='Aggregate Supply', color='red')
        equilibrium = self.calculate_equilibrium()[good]
        plt.plot([equilibrium['quantity']], [equilibrium['price']], 'go', label='Equilibrium')
        plt.title(f'Aggregate Supply and Demand Curves for {good}')
        plt.xlabel('Quantity')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Set up logging for the main script
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create some test agents
    buyers = [
        create_economic_agent(
            agent_id=f"buyer_{i}",
            goods=["apple"],
            buy_goods=["apple"],
            sell_goods=[],
            base_values={"apple": 100},
            initial_cash=1000,
            initial_goods={"apple": 0},
            num_units=10,
            noise_factor=0.1,
            max_relative_spread=0.2
        ) for i in range(5)
    ]
    
    sellers = [
        create_economic_agent(
            agent_id=f"seller_{i}",
            goods=["apple"],
            buy_goods=[],
            sell_goods=["apple"],
            base_values={"apple": 80},
            initial_cash=0,
            initial_goods={"apple": 10},
            num_units=10,
            noise_factor=0.1,
            max_relative_spread=0.2
        ) for i in range(5)
    ]
    
    # Create the Equilibrium object
    equilibrium = Equilibrium(agents=buyers + sellers, goods=["apple"])
    
    # Calculate and print the equilibrium
    result = equilibrium.calculate_equilibrium()
    print("Equilibrium Results:")
    for good, data in result.items():
        print(f"\nGood: {good}")
        for key, value in data.items():
            print(f"  {key}: {value}")
    
    # Plot the supply and demand curves
    equilibrium.plot_supply_demand("apple")