from typing import List, Dict
from pydantic import BaseModel, Field, computed_field
from functools import cached_property
from market_agents.economics.econ_models import SavableBaseModel
from market_agents.economics.econ_agent import ZiFactory, ZiParams, EconomicAgent
from market_agents.economics.equilibrium import Equilibrium, EquilibriumResults
import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class Scenario(SavableBaseModel):
    name: str = Field(default="scenario")
    goods: List[str]
    factories: List[ZiFactory]
    _current_episode: int = 0

    @property
    def num_episodes(self) -> int:
        return len(self.factories)

    def next_episode(self) -> int:
        current = self._current_episode
        self._current_episode = (self._current_episode + 1) % self.num_episodes
        return current

    @computed_field
    @cached_property
    def equilibriums(self) -> List[Equilibrium]:
        logger.info("Computing equilibriums...")
        return [
            Equilibrium(agents=self._get_agents(episode), goods=self.goods)
            for episode in range(self.num_episodes)
        ]

    def _get_agents(self, episode: int) -> List[EconomicAgent]:
        return self.factories[episode].agents

    @computed_field
    @property
    def agents(self) -> List[EconomicAgent]:
        return self._get_agents(self._current_episode)

    @computed_field
    def ce(self) -> Equilibrium:
        return self.equilibriums[self._current_episode]

    @computed_field
    @cached_property
    def prices(self) -> Dict[str, List[float]]:
        return {
            good: [eq.calculate_equilibrium()[good].price for eq in self.equilibriums]
            for good in self.goods
        }

    @computed_field
    @cached_property
    def quantities(self) -> Dict[str, List[int]]:
        return {
            good: [eq.calculate_equilibrium()[good].quantity for eq in self.equilibriums]
            for good in self.goods
        }

    def run(self) -> List[Dict[str, EquilibriumResults]]:
        return [eq.calculate_equilibrium() for eq in self.equilibriums]
    
    def plot_dynamic_equilibrium(self, good: str, include_supply_demand: bool = False):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        equilibrium_prices = []
        equilibrium_quantities = []
        
        for episode, equilibrium in enumerate(self.equilibriums):
            eq_data = equilibrium.calculate_equilibrium()[good]
            eq_price = eq_data.price
            eq_quantity = eq_data.quantity
            equilibrium_prices.append(eq_price)
            equilibrium_quantities.append(eq_quantity)
            
            if include_supply_demand:
                demand_prices, supply_prices = equilibrium._aggregate_curves(good)
                demand_quantities = list(range(1, len(demand_prices) + 1))
                supply_quantities = list(range(1, len(supply_prices) + 1))
                
                ax.step(demand_quantities, demand_prices, where='post', 
                        alpha=0.3, color='blue', label=f'Demand (episode {episode})' if episode == 0 else "")
                ax.step(supply_quantities, supply_prices, where='post', 
                        alpha=0.3, color='red', label=f'Supply (episode {episode})' if episode == 0 else "")
            
            # Plot equilibrium point
            ax.plot(eq_quantity, eq_price, 'go', markersize=10)
            
            # Add dashed lines to axes
            ax.axhline(y=eq_price, color='g', linestyle='--', alpha=0.5)
            ax.axvline(x=eq_quantity, color='g', linestyle='--', alpha=0.5)
        
        # Plot the dynamic equilibrium curve
        ax.plot(equilibrium_quantities, equilibrium_prices, 'b-', linewidth=2, label='Dynamic Equilibrium')
        
        ax.set_xlabel('Quantity')
        ax.set_ylabel('Price')
        ax.set_title(f'Dynamic Equilibrium for {good}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    buyer_params = ZiParams(
        id="buyer_template",
        initial_cash=10000.0,
        initial_goods={"apple": 0},
        base_values={"apple": 20.0},
        num_units=5,
        noise_factor=0.05,
        max_relative_spread=0.2,
        is_buyer=True
    )

    seller_params = ZiParams(
        id="seller_template",
        initial_cash=0,
        initial_goods={"apple": 20},
        base_values={"apple": 15.0},
        num_units=5,
        noise_factor=0.05,
        max_relative_spread=0.2,
        is_buyer=False
    )

    # Create a list of factories, one for each episode
    factories = [
        ZiFactory(
            id=f"factory_episode_{i}",
            goods=["apple"],
            num_buyers=int(10+(i*3)),  # Increase buyers by 1 each episode
            num_sellers=10+ i,     # Keep sellers constant
            buyer_params=buyer_params,
            seller_params=seller_params
        )
        for i in range(0,500,5)  
    ]

    scenario = Scenario(
        name="Growing Buyers Scenario",
        goods=["apple"],
        factories=factories
    )

    # Plot dynamic equilibrium without supply and demand curves
    fig1 = scenario.plot_dynamic_equilibrium("apple", include_supply_demand=False)
    fig1.savefig("dynamic_equilibrium_simple.png")

    # Plot dynamic equilibrium with supply and demand curves
    fig2 = scenario.plot_dynamic_equilibrium("apple", include_supply_demand=True)
    fig2.savefig("dynamic_equilibrium_with_curves.png")

    plt.show()