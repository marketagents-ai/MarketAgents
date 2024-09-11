from typing import List, Dict, Tuple
from market_agents.economics.econ_agent import EconomicAgent
from market_agents.economics.econ_models import Good, Basket
import numpy as np
import matplotlib.pyplot as plt

def compute_equilibrium(agents: List[EconomicAgent], goods: List[str], initial_prices: Dict[str, float], max_iterations: int = 5000, tolerance: float = 0.0001) -> Tuple[Dict[str, float], Dict[str, float]]:
    prices = initial_prices.copy()
    damping_factor = 0.3
    learning_rate = 0.05
    
    for iteration in range(max_iterations):
        excess_demand: Dict[str, float] = {good: 0.0 for good in goods}
        
        # Compute excess demand for each good
        for agent in agents:
            demand = compute_demand(agent, prices, goods)
            supply = compute_supply(agent, prices, goods)
            
            for good in goods:
                excess_demand[good] += demand.get(good, 0.0) - supply.get(good, 0.0)
        
        # Update prices based on excess demand
        max_price_change = 0.0
        for good in goods:
            step_size = learning_rate / (1 + iteration * 0.001)  # Slower decrease in step size
            price_change = step_size * excess_demand[good] / (len(agents) * abs(prices[good]))  # Normalize by current price
            new_price = max(0.01, prices[good] * (1 + damping_factor * price_change))
            max_price_change = max(max_price_change, abs(new_price - prices[good]) / prices[good])
            prices[good] = new_price
        
        # Check if equilibrium is reached
        if max_price_change < tolerance:
            break
    
    # Compute final quantities
    quantities: Dict[str, float] = {good: 0.0 for good in goods}
    for agent in agents:
        demand = compute_demand(agent, prices, goods)
        for good, quantity in demand.items():
            quantities[good] += quantity
    
    return prices, quantities

def compute_demand(agent: EconomicAgent, prices: Dict[str, float], goods: List[str]) -> Dict[str, float]:
    demand: Dict[str, float] = {}
    budget = agent.endowment.current_basket.cash
    current_goods = {good.name: good.quantity for good in agent.endowment.current_basket.goods}
    
    for good in goods:
        quantity = current_goods.get(good, 0.0)
        price = prices[good]
        while budget > 0:
            marginal_utility = agent.utility_function.marginal_utility(
                Basket(cash=budget, goods=[Good(name=g, quantity=current_goods.get(g, 0.0) + (quantity - current_goods.get(g, 0.0) if g == good else 0.0)) for g in goods]),
                good
            )
            if marginal_utility <= price or budget < price * 0.1:
                break
            quantity += 0.1
            budget -= 0.1 * price
        
        demand[good] = max(0, quantity - current_goods.get(good, 0.0))
    
    return demand

def compute_supply(agent: EconomicAgent, prices: Dict[str, float], goods: List[str]) -> Dict[str, float]:
    if agent.is_buyer:
        return {good: 0.0 for good in goods}
    
    supply: Dict[str, float] = {}
    for good in goods:
        quantity = 0.0
        price = prices[good]
        while True:
            if agent.cost_function is None:
                break
            marginal_cost = agent.cost_function.marginal_cost(
                Basket(cash=0, goods=[Good(name=good, quantity=quantity)]),
                good
            )
            if marginal_cost >= price:
                break
            quantity += 0.1
        
        supply[good] = quantity
    
    return supply

def plot_equilibrium(agents: List[EconomicAgent], goods: List[str], equilibrium_prices: Dict[str, float], equilibrium_quantities: Dict[str, float]):
    for good in goods:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Generate a more reasonable price range
        price_range = np.linspace(equilibrium_prices[good] * 0.5, equilibrium_prices[good] * 1.5, 100)
        
        # Compute aggregate demand and supply
        demand = []
        supply = []
        for price in price_range:
            total_demand = sum(compute_demand(agent, {good: price}, [good])[good] for agent in agents)
            total_supply = sum(compute_supply(agent, {good: price}, [good])[good] for agent in agents)
            demand.append(total_demand)
            supply.append(total_supply)
        
        # Plot demand and supply curves
        ax.plot(demand, price_range, label='Demand', color='blue')
        ax.plot(supply, price_range, label='Supply', color='red')
        
        # Plot equilibrium point
        ax.plot(equilibrium_quantities[good], equilibrium_prices[good], 'go', markersize=10, label='Equilibrium')
        
        # Plot equilibrium lines
        ax.axhline(y=equilibrium_prices[good], color='green', linestyle='--', alpha=0.5)
        ax.axvline(x=equilibrium_quantities[good], color='green', linestyle='--', alpha=0.5)
        
        ax.set_title(f'Supply and Demand for {good}')
        ax.set_xlabel('Quantity')
        ax.set_ylabel('Price')
        ax.legend()
        
        # Add text with equilibrium price and quantity
        ax.text(0.05, 0.95, f"Equilibrium Price: {equilibrium_prices[good]:.2f}\nEquilibrium Quantity: {equilibrium_quantities[good]:.2f}", 
                transform=ax.transAxes, verticalalignment='top')
        
        # Set reasonable axis limits
        max_quantity = max(max(demand), max(supply), equilibrium_quantities[good])
        ax.set_xlim(0, max_quantity * 1.1)
        ax.set_ylim(min(price_range), max(price_range))
        
        plt.tight_layout()
        plt.show()

        # Print demand and supply at equilibrium price
        eq_demand = sum(compute_demand(agent, {good: equilibrium_prices[good]}, [good])[good] for agent in agents)
        eq_supply = sum(compute_supply(agent, {good: equilibrium_prices[good]}, [good])[good] for agent in agents)
        print(f"{good} - Equilibrium Price: {equilibrium_prices[good]:.2f}, Equilibrium Quantity: {equilibrium_quantities[good]:.2f}")
        print(f"Demand at equilibrium price: {eq_demand:.2f}")
        print(f"Supply at equilibrium price: {eq_supply:.2f}")
        print("---")

# Example usage
if __name__ == "__main__":
    from market_agents.economics.utility import create_utility_function, create_cost_function
    from market_agents.economics.econ_agent import create_economic_agent
    
    goods = ["apple", "banana"]
    base_values: Dict[str, float] = {"apple": 10.0, "banana": 8.0}
    initial_prices: Dict[str, float] = {"apple": 5.0, "banana": 4.0}
    
    agents = [
        create_economic_agent("buyer1", True, goods, base_values, 100.0, {}, "cobb-douglas", "quadratic", cobb_douglas_scale=10),
        create_economic_agent("buyer2", True, goods, base_values, 120.0, {}, "cobb-douglas", "quadratic", cobb_douglas_scale=10),
        create_economic_agent("seller1", False, goods, base_values, 50.0, {"apple": 10.0, "banana": 8.0}, "cobb-douglas", "quadratic", cobb_douglas_scale=10),
        create_economic_agent("seller2", False, goods, base_values, 60.0, {"apple": 12.0, "banana": 10.0}, "cobb-douglas", "quadratic", cobb_douglas_scale=10)
    ]
    
    print("Initial agent details:")
    for agent in agents:
        print(f"{agent.id}:")
        print(f"  Is buyer: {agent.is_buyer}")
        print(f"  Initial cash: {agent.endowment.current_basket.cash}")
        print(f"  Initial goods: {agent.endowment.current_basket.goods_dict}")
        if agent.is_buyer:
            print(f"  Utility function: {type(agent.utility_function).__name__}")
        else:
            print(f"  Cost function: {type(agent.cost_function).__name__}")
    
    print("\nComputing equilibrium...")
    equilibrium_prices, equilibrium_quantities = compute_equilibrium(agents, goods, initial_prices)
    
    print("\nEquilibrium Prices:", equilibrium_prices)
    print("Equilibrium Quantities:", equilibrium_quantities)

    print("\nAgent Details at Equilibrium:")
    for agent in agents:
        demand = compute_demand(agent, equilibrium_prices, goods)
        supply = compute_supply(agent, equilibrium_prices, goods)
        print(f"{agent.id} - Demand: {demand}, Supply: {supply}")
    
    # Add this line to plot the equilibrium
    plot_equilibrium(agents, goods, equilibrium_prices, equilibrium_quantities)

    # Add this section to check marginal utilities and costs at equilibrium
    print("\nMarginal Utilities and Costs at Equilibrium:")
    for agent in agents:
        for good in goods:
            if agent.is_buyer:
                mu = agent.utility_function.marginal_utility(
                    Basket(cash=agent.endowment.current_basket.cash, goods=[Good(name=good, quantity=demand[good])]),
                    good
                )
                print(f"{agent.id} - {good} Marginal Utility: {mu:.2f}")
            else:
                if agent.cost_function:
                    mc = agent.cost_function.marginal_cost(
                        Basket(cash=0, goods=[Good(name=good, quantity=supply[good])]),
                        good
                    )
                    print(f"{agent.id} - {good} Marginal Cost: {mc:.2f}")
