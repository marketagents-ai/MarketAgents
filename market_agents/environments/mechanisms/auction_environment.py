import logging
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
from pydantic import BaseModel, Field, computed_field, model_validator
from environments.environment import Environment
from environments.auction.auction import Ask, Bid, DoubleAuction, MarketAction

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
    max_steps: int = Field(..., description="Maximum number of steps in the auction")
    auction_type: Literal['double'] = Field(..., description="Type of auction")
    current_step: int = Field(default=0, description="Current step in the auction")
    auction: Any = Field(None, description="The auction mechanism")
    protocol: Type[Protocol] = Field(..., description="Communication protocol class")
    
    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.auction = self._create_auction()

    def _create_auction(self):
        if self.auction_type == 'double':
            return DoubleAuction(max_rounds=self.max_steps)
        else:
            raise ValueError(f"Unsupported auction type: {self.auction_type}")

    def get_observation(self, agent_id: str):
        """Get observation for an agent."""
        observation = self.auction.get_current_trade_execution(int(agent_id))
        if self.protocol:
            return self.protocol.create_observation(
                sender="market",
                agent_id=agent_id,
                content=observation,
                step=self.current_step
            )
        return observation

    def parse_action(self, action: ACLMessage) -> Dict[str, Any]:
        """Parse an action into a format the environment can use."""
        if not isinstance(action, ACLMessage):
            raise ValueError(f"Expected ACLMessage, got {type(action)}")
        
        if action.performative not in [Performative.PROPOSE, Performative.REQUEST]:
            return {"type": "hold", "price": 0, "quantity": 0}
        
        content = action.content
        if not isinstance(content, dict):
            return {"type": "hold", "price": 0, "quantity": 0}
        
        action_type = content.get("type", "hold")
        if action_type not in ["bid", "ask", "hold"]:
            return {"type": "hold", "price": 0, "quantity": 0}
        
        return {
            "type": action_type,
            "price": float(content.get("price", 0)),
            "quantity": int(content.get("quantity", 0))
        }

    @property
    def current_demand_curve(self) -> BaseCurve:
        """Generate the current demand curve."""
        points = []
        for bid in sorted(self.auction.order_book.bids, key=lambda x: x.market_action.price, reverse=True):
            points.append(CurvePoint(quantity=bid.market_action.quantity, price=bid.market_action.price))
        return BaseCurve(points=points)

    @property
    def current_supply_curve(self) -> BaseCurve:
        """Generate the current supply curve."""
        points = []
        for ask in sorted(self.auction.order_book.asks, key=lambda x: x.market_action.price):
            points.append(CurvePoint(quantity=ask.market_action.quantity, price=ask.market_action.price))
        return BaseCurve(points=points)

    @property
    def remaining_trade_opportunities(self) -> int:
        """Calculate the number of remaining trade opportunities."""
        return min(len(self.auction.order_book.bids), len(self.auction.order_book.asks))

    @property
    def remaining_surplus(self) -> float:
        """Calculate the remaining surplus in the market."""
        remaining_surplus = 0
        for bid in self.auction.order_book.bids:
            for ask in self.auction.order_book.asks:
                if bid.market_action.price >= ask.market_action.price:
                    remaining_surplus += bid.base_value - ask.base_cost
        return remaining_surplus

    @property
    def total_utility(self) -> float:
        """Calculate the total utility of all agents."""
        return self.auction.total_surplus_extracted

    @property
    def ce_price(self) -> float:
        """Get the competitive equilibrium price."""
        demand_curve = self.current_demand_curve
        supply_curve = self.current_supply_curve
        
        for d_point, s_point in zip(demand_curve.points, supply_curve.points):
            if d_point.price >= s_point.price:
                return (d_point.price + s_point.price) / 2
        
        return 0  # No equilibrium found

    @property
    def ce_quantity(self) -> float:
        """Get the competitive equilibrium quantity."""
        demand_curve = self.current_demand_curve
        supply_curve = self.current_supply_curve
        
        ce_quantity = 0
        for d_point, s_point in zip(demand_curve.points, supply_curve.points):
            if d_point.price >= s_point.price:
                ce_quantity += min(d_point.quantity, s_point.quantity)
            else:
                break
        
        return ce_quantity

    @property
    def efficiency(self) -> float:
        """Calculate the market efficiency."""
        total_possible_surplus = self.auction.total_surplus_extracted + self.remaining_surplus
        if total_possible_surplus == 0:
            return 1.0  # Avoid division by zero
        return self.auction.total_surplus_extracted / total_possible_surplus

    def print_market_state(self):
        """Print the current market state."""
        logger.info("\n=== Current Market State ===")
        logger.info(f"Current Round: {self.current_step}")
        logger.info(f"Total Trades: {len(self.auction.successful_trades)}")
        logger.info(f"Total Surplus Extracted: {self.auction.total_surplus_extracted:.2f}")
        logger.info(f"Current Efficiency: {self.efficiency:.2%}")
        logger.info(f"Remaining Trade Opportunities: {self.remaining_trade_opportunities}")
        logger.info(f"CE Price: {self.ce_price:.2f}")
        logger.info(f"CE Quantity: {self.ce_quantity:.2f}")

    def calculate_equilibrium(self, initial: bool = True) -> Tuple[float, float, float, float, float]:
        """Calculate the market equilibrium."""
        ce_price = self.ce_price
        ce_quantity = self.ce_quantity
        efficiency = self.efficiency
        total_surplus = self.auction.total_surplus_extracted
        remaining_surplus = self.remaining_surplus
        
        return ce_price, ce_quantity, efficiency, total_surplus, remaining_surplus

    def step(self, agent_actions):
        parsed_actions = {agent_id: self.parse_action(action) for agent_id, action in agent_actions.items()}
        
        # Process actions and update the auction state
        for agent_id, action in parsed_actions.items():
            if action['type'] == 'bid':
                self.auction.process_action(int(agent_id), action, True, action['price'])
            elif action['type'] == 'ask':
                self.auction.process_action(int(agent_id), action, False, action['price'])
        
        # Match orders and execute trades
        trade_info = self.auction.update_auction_state()
        
        # Update market info
        market_info = self.auction.get_market_info()
        
        # Prepare observations for each agent
        observations = {agent_id: self.get_observation(agent_id) for agent_id in agent_actions.keys()}
        
        # Check if the auction has ended
        done = self.auction.is_auction_complete()
        
        self.current_step += 1
        self.auction.advance_round()
        
        return {
            "observations": observations,
            "market_info": market_info,
            "trade_info": trade_info,
            "done": done,
            "current_step": self.current_step
        }

    def update(self, agent_actions: Dict[str, Any]) -> Dict[str, Any]:
        return self.step(agent_actions)

    def get_global_state(self) -> Dict[str, Any]:
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
            "efficiency": self.efficiency,
            "order_book": self.auction.order_book.dict(),
            "trade_history": [trade.dict() for trade in self.auction.trade_history],
            "successful_trades": [trade.dict() for trade in self.auction.successful_trades],
            "current_round": self.auction.current_round,
            "total_surplus_extracted": self.auction.total_surplus_extracted,
            "average_prices": self.auction.average_prices
        }
    def reset(self) -> Dict[str, Any]:
        """Reset the environment and return the initial global state."""
        self.current_step = 0
        self.auction.reset()
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
            action_type = action.get('type', 'hold')
            if action_type in ["bid", "ask"]:
                return {
                    "type": action_type,
                    "price": float(action['price']),
                    "quantity": int(action['quantity'])
                }
            else:
                return {"type": "hold"}
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
        return self.step(agent_actions)

    def get_action_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for the MarketActionSchema."""
        return MarketActionSchema.model_json_schema()
