import os
import logging
from collections import defaultdict
from functools import cached_property
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from market_agent.market_schemas import MarketActionSchema
from pydantic import BaseModel, Field, computed_field, model_validator

from market_agent.market_agent_todo import MarketAgent
from environments.environment import Environment
from protocols.protocol import Protocol
from environments.auction.auction import DoubleAuction
from protocols.acl_message import ACLMessage, Performative

# Set up logger
logger = logging.getLogger(__name__)


class CurvePoint(BaseModel):
    """Represents a point on a supply or demand curve."""

    quantity: float = Field(..., description="Quantity of goods")
    price: float = Field(..., description="Price of goods")


class BaseCurve(BaseModel):
    """Base class for supply and demand curves."""

    points: List[CurvePoint] = Field(..., description="List of points defining the curve")

    def get_x_y_values(self) -> Tuple[List[float], List[float]]:
        """Extract x and y values from curve points."""
        x_values = []
        y_values = []
        for point in self.points:
            x_values.extend([point.quantity, point.quantity])
            y_values.extend([point.price, point.price])
        return x_values, y_values


class InitialDemandCurve(BaseCurve):
    """Represents the initial demand curve."""

    @model_validator(mode='after')
    def validate_monotonicity(self):
        """Ensure the demand curve is monotonically decreasing."""
        sorted_points = sorted(self.points, key=lambda p: p.quantity)
        for i in range(1, len(sorted_points)):
            if sorted_points[i].price > sorted_points[i-1].price:
                raise ValueError("Initial demand curve must be monotonically decreasing")
        return self


class InitialSupplyCurve(BaseCurve):
    """Represents the initial supply curve."""

    @model_validator(mode='after')
    def validate_monotonicity(self):
        """Ensure the supply curve is monotonically increasing."""
        sorted_points = sorted(self.points, key=lambda p: p.quantity)
        for i in range(1, len(sorted_points)):
            if sorted_points[i].price < sorted_points[i-1].price:
                raise ValueError("Initial supply curve must be monotonically increasing")
        return self


class AuctionEnvironment(Environment):
    """Represents the auction environment with agents and market dynamics."""

    agents: List[MarketAgent] = Field(..., description="List of agents in the environment")
    max_steps: int = Field(..., description="Maximum number of steps in the auction")
    protocol: Protocol = Field(..., description="Base protocol for agent communication")
    auction_type: Literal['double'] = Field(..., description="Type of auction")
    current_step: int = Field(default=0, description="Current step in the auction")
    auction: Any = Field(None, description="The auction mechanism")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.auction = self._create_auction()

    def _create_auction(self):
        if self.auction_type == 'double':
            return DoubleAuction(agents=self.agents, max_rounds=self.max_steps)
        else:
            raise ValueError(f"Unsupported auction type: {self.auction_type}")
        
    @property
    def buyers(self) -> List[MarketAgent]:
        """Get all buyer agents in the environment."""
        return [agent for agent in self.agents if self._is_buyer(agent)]

    @property
    def sellers(self) -> List[MarketAgent]:
        """Get all seller agents in the environment."""
        return [agent for agent in self.agents if not self._is_buyer(agent)]

    @staticmethod
    def _is_buyer(agent: MarketAgent) -> bool:
        """Check if an agent is a buyer."""
        return agent.is_buyer

    def get_agent(self, agent_id: str) -> Optional[MarketAgent]:
        """Get an agent by its ID."""
        for agent in self.agents:
            if agent.id == agent_id:
                return agent
        return None

    @computed_field
    @cached_property
    def initial_demand_curve(self) -> InitialDemandCurve:
        """Generate the initial demand curve."""
        return self._generate_initial_demand_curve()

    @computed_field
    @cached_property
    def initial_supply_curve(self) -> InitialSupplyCurve:
        """Generate the initial supply curve."""
        return self._generate_initial_supply_curve()

    @computed_field
    @property
    def current_demand_curve(self) -> BaseCurve:
        """Generate the current demand curve."""
        return self._generate_current_demand_curve()

    @computed_field
    @property
    def current_supply_curve(self) -> BaseCurve:
        """Generate the current supply curve."""
        return self._generate_current_supply_curve()

    def get_global_state(self) -> Dict[str, Any]:
        """Get the current state of the auction environment."""
        return {
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "current_demand_curve": self.current_demand_curve.dict(),
            "current_supply_curve": self.current_supply_curve.dict(),
            "remaining_trade_opportunities": self.remaining_trade_opportunities,
            "remaining_surplus": self.remaining_surplus,
            "total_utility": self.total_utility,
            "ce_price": self.ce_price,
            "ce_quantity": self.ce_quantity,
            "efficiency": self.efficiency
        }

    def _generate_initial_demand_curve(self) -> InitialDemandCurve:
        """Generate the initial demand curve based on buyer preferences."""
        aggregated_demand = defaultdict(float)
        for buyer in self.buyers:
            preference_schedule = buyer.preference_schedule
            for quantity, value in preference_schedule.values.items():
                aggregated_demand[value] += quantity
        
        points = []
        cumulative_quantity = 0
        for price, quantity in sorted(aggregated_demand.items(), reverse=True):
            cumulative_quantity += quantity
            points.append(CurvePoint(quantity=cumulative_quantity, price=price))
        
        return InitialDemandCurve(points=points)

    def _generate_initial_supply_curve(self) -> InitialSupplyCurve:
        """Generate the initial supply curve based on seller preferences."""
        aggregated_supply = defaultdict(float)
        for seller in self.sellers:
            preference_schedule = seller.preference_schedule
            for quantity, cost in preference_schedule.values.items():
                aggregated_supply[cost] += quantity
        
        points = []
        cumulative_quantity = 0
        for price, quantity in sorted(aggregated_supply.items()):
            cumulative_quantity += quantity
            points.append(CurvePoint(quantity=cumulative_quantity, price=price))
        
        return InitialSupplyCurve(points=points)

    def _generate_current_demand_curve(self) -> BaseCurve:
        """Generate the current demand curve based on buyer preferences and allocations."""
        aggregated_demand = defaultdict(float)
        for buyer in self.buyers:
            preference_schedule = buyer.preference_schedule
            allocation = buyer.endowment
            for quantity, value in preference_schedule.values.items():
                if allocation.goods < quantity:
                    aggregated_demand[value] += (quantity - allocation.goods)
        
        points = []
        cumulative_quantity = 0
        for price, quantity in sorted(aggregated_demand.items(), reverse=True):
            cumulative_quantity += quantity
            points.append(CurvePoint(quantity=cumulative_quantity, price=price))
        
        return BaseCurve(points=points)

    def _generate_current_supply_curve(self) -> BaseCurve:
        """Generate the current supply curve based on seller preferences and allocations."""
        aggregated_supply = defaultdict(float)
        for seller in self.sellers:
            preference_schedule = seller.preference_schedule
            allocation = seller.endowment
            for quantity, cost in preference_schedule.values.items():
                if allocation.goods >= quantity:
                    aggregated_supply[cost] += allocation.goods - quantity + 1
        
        points = []
        cumulative_quantity = 0
        for price, quantity in sorted(aggregated_supply.items()):
            cumulative_quantity += quantity
            points.append(CurvePoint(quantity=cumulative_quantity, price=price))
        
        return BaseCurve(points=points)

    @computed_field
    @property
    def remaining_trade_opportunities(self) -> int:
        """Calculate the number of remaining trade opportunities."""
        potential_trades = 0
        for buyer in self.buyers:
            for seller in self.sellers:
                if buyer.endowment.cash > 0 and seller.endowment.goods > 0:
                    buyer_value = buyer.preference_schedule.get_value(buyer.endowment.goods + 1)
                    seller_cost = seller.preference_schedule.get_value(seller.endowment.goods)
                    if buyer_value > seller_cost and buyer.endowment.cash >= seller_cost:
                        potential_trades += 1
        return potential_trades

    @computed_field
    @property
    def remaining_surplus(self) -> float:
        """Calculate the remaining surplus in the market."""
        remaining_surplus = 0.0
        for buyer in self.buyers:
            for seller in self.sellers:
                if buyer.endowment.cash > 0 and seller.endowment.goods > 0:
                    buyer_value = buyer.preference_schedule.get_value(buyer.endowment.goods + 1)
                    seller_cost = seller.preference_schedule.get_value(seller.endowment.goods)
                    if buyer_value > seller_cost:
                        remaining_surplus += (buyer_value - seller_cost)
        return remaining_surplus

    @computed_field
    @property
    def total_utility(self) -> float:
        """Calculate the total utility of all agents."""
        return sum(agent.calculate_individual_surplus() for agent in self.agents)

    @computed_field
    @property
    def ce_price(self) -> float:
        """Get the competitive equilibrium price."""
        return self.calculate_equilibrium(initial=False)[0]

    @computed_field
    @property
    def ce_quantity(self) -> float:
        """Get the competitive equilibrium quantity."""
        return self.calculate_equilibrium(initial=False)[1]

    @computed_field
    @property
    def ce_buyer_surplus(self) -> float:
        """Get the competitive equilibrium buyer surplus."""
        return self.calculate_equilibrium(initial=False)[2]

    @computed_field
    @property
    def ce_seller_surplus(self) -> float:
        """Get the competitive equilibrium seller surplus."""
        return self.calculate_equilibrium(initial=False)[3]

    @computed_field
    @property
    def ce_total_surplus(self) -> float:
        """Get the total competitive equilibrium surplus."""
        return self.ce_buyer_surplus + self.ce_seller_surplus

    @computed_field
    @property
    def efficiency(self) -> float:
        """Calculate the market efficiency."""
        extracted_surplus = self.total_utility
        theoretical_surplus = self.ce_total_surplus
        if theoretical_surplus <= 0:
            raise ValueError("Theoretical surplus is zero or negative")
        efficiency = extracted_surplus / theoretical_surplus
        if efficiency < 0:
            raise ValueError("Negative efficiency detected")
        return efficiency

    def print_market_state(self):
        """Print the current market state."""
        logger.info("Market State:")
        for agent in self.agents:
            role = "Buyer" if agent.is_buyer else "Seller"
            logger.info(f"Agent {agent.id} ({role}):")
            logger.info(f"  Goods: {agent.endowment.goods}")
            logger.info(f"  Cash: {agent.endowment.cash:.2f}")
            logger.info(f"  Utility: {agent.calculate_individual_surplus():.2f}")
        logger.info(f"Total Market Utility: {self.total_utility:.2f}")
        logger.info(f"Remaining Trade Opportunities: {self.remaining_trade_opportunities}")
        logger.info(f"Remaining Surplus: {self.remaining_surplus:.2f}")
        logger.info(f"Market Efficiency: {self.efficiency:.2%}")

    def calculate_equilibrium(self, initial: bool = True) -> Tuple[float, float, float, float, float]:
        """Calculate the market equilibrium."""
        demand_curve = self.initial_demand_curve if initial else self.current_demand_curve
        supply_curve = self.initial_supply_curve if initial else self.current_supply_curve

        demand_points = sorted(demand_curve.points, key=lambda p: p.quantity)
        supply_points = sorted(supply_curve.points, key=lambda p: p.quantity)

        ce_quantity = 0
        ce_price = 0
        d_index = 0
        s_index = 0

        while d_index < len(demand_points) and s_index < len(supply_points):
            if demand_points[d_index].price >= supply_points[s_index].price:
                ce_quantity = min(demand_points[d_index].quantity, supply_points[s_index].quantity)
                ce_price = (demand_points[d_index].price + supply_points[s_index].price) / 2
                if demand_points[d_index].quantity < supply_points[s_index].quantity:
                    d_index += 1
                else:
                    s_index += 1
            else:
                break

        buyer_surplus = sum(max(p.price - ce_price, 0) * (p.quantity - prev_q)
                            for prev_q, p in zip([0] + [p.quantity for p in demand_points[:-1]], demand_points)
                            if p.quantity <= ce_quantity)

        seller_surplus = sum(max(ce_price - p.price, 0) * (p.quantity - prev_q)
                             for prev_q, p in zip([0] + [p.quantity for p in supply_points[:-1]], supply_points)
                             if p.quantity <= ce_quantity)

        total_surplus = buyer_surplus + seller_surplus

        return ce_price, ce_quantity, buyer_surplus, seller_surplus, total_surplus

    def step(self, agent_actions: Dict[str, Any]) -> Dict[str, Any]:
        """Update the global state based on agent actions."""
        # Process agent actions and update the auction
        parsed_actions = {agent_id: self.parse_action(action) for agent_id, action in agent_actions.items()}
        self.auction.process_actions(parsed_actions)
        
        # Execute trades
        trades = self.auction.execute_trades()
        
        # Update agent observations based on trade results
        for agent in self.agents:
            observation = self.get_observation(agent.id)
            agent.perceive(observation)
        
        # Update the environment state
        self.current_step += 1
        
        return self.get_global_state()

    def get_observation(self, agent_id: str):
        """Send observation to an agent using the provided protocol."""
        observation = self.auction.get_current_trade_execution(agent_id)
        message = self.protocol.create_message(
            performative=Performative.INFORM,
            sender="market",
            receiver=agent_id,
            content=observation,
            protocol="double-auction",
            ontology="market-ontology",
            conversation_id=f"observation-{self.current_step}-{agent_id}"
        )
        return message

    def reset(self) -> Dict[str, Any]:
        """Reset the environment and return the initial global state."""
        self.current_step = 0
        self.auction.reset()
        for agent in self.agents:
            agent.reset()
        return self.get_global_state()

    def render(self):
        """Render the environment."""
        self.print_market_state()

    def get_action_space(self) -> Dict[str, Any]:
        """Return the action space of the environment."""
        return {
            "type": "continuous",
            "shape": (2,),
            "low": [0, 0],
            "high": [float('inf'), float('inf')],
            "description": "A tuple of (price, quantity) for bid or ask"
        }

    def get_observation_space(self) -> Any:
        """Return the observation space of the environment."""
        return self.auction.get_observation_space()

    def parse_action(self, action: Union[str, Dict[str, Any], ACLMessage]) -> Dict[str, Any]:
        """Parse an action into a format the environment can use."""
        if isinstance(action, ACLMessage):
            return action.parse_to_market_action()
        elif isinstance(action, dict):
            return action
        elif isinstance(action, str):
            # Parse string action (e.g., "BID 100 5" or "ASK 90 3")
            parts = action.split()
            if len(parts) != 3:
                raise ValueError(f"Invalid action format: {action}")
            action_type, price, quantity = parts
            return {
                "type": action_type.lower(),
                "price": float(price),
                "quantity": int(quantity)
            }
        else:
            raise ValueError(f"Unsupported action type: {type(action)}")

    def update(self, agent_actions: Dict[str, Any]) -> Dict[str, Any]:
        """Update the global state based on agent actions."""
        # Process agent actions and update the auction
        parsed_actions = {agent_id: self.parse_action(action) for agent_id, action in agent_actions.items()}
        self.auction.process_actions(parsed_actions)
        
        # Execute trades
        trades = self.auction.execute_trades()
        
        # Update agent observations based on trade results
        for agent in self.agents:
            observation = self.get_observation(agent)
            agent.perceive(observation)
        
        # Update the environment state
        self.current_step += 1
        
        return self.get_global_state()

    def get_action_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for the MarketActionSchema."""
        return MarketActionSchema.model_json_schema()

def generate_llm_market_agents(num_agents: int, num_units: int, buyer_base_value: int, seller_base_value: int, spread: float, use_llm: bool = False, llm_config: Dict[str, Any] = None, initial_cash: float = 1000, initial_goods: int = 0, noise_factor: float = 0.1) -> List[MarketAgent]:
    agents = []
    protocol = ACLMessage()  # Pass the ACLMessage class, not an instance
    for i in range(num_agents):
        is_buyer = i < num_agents // 2
        base_value = buyer_base_value if is_buyer else seller_base_value
        agent_initial_cash = initial_cash if is_buyer else 0
        agent_initial_goods = 0 if is_buyer else num_units
        
        market_agent = MarketAgent.create(
            agent_id=i,
            is_buyer=is_buyer,
            num_units=num_units,
            base_value=base_value,
            use_llm=use_llm,
            initial_cash=agent_initial_cash,
            initial_goods=agent_initial_goods,
            noise_factor=noise_factor,
            max_relative_spread=spread,
            llm_config=llm_config,
            protocol=protocol,  # Pass the ACLMessage class as the protocol
            environments={}  # Add an empty dict for environments
        )
        agents.append(market_agent)
    
    return agents