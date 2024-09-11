import logging
import traceback
from typing import Any, List, Optional, Dict, Union
from pydantic import BaseModel, Field, computed_field
from colorama import Fore, Style
from market_agents.environments.environment import (
    Mechanism, LocalAction, GlobalAction, LocalObservation, GlobalObservation,
    LocalEnvironmentStep, EnvironmentStep
)

# Set up logger
logger = logging.getLogger(__name__)

class MarketAction(BaseModel):
    price: float = Field(..., description="Price of the order")
    quantity: int = Field(default=1, ge=1, description="Quantity of the order")

class Order(BaseModel):
    agent_id: str = Field(..., description="Unique identifier of the agent placing the order")
    market_action: MarketAction = Field(..., description="Market action of the order")

class Bid(Order):
    """Represents a bid order."""
    base_value: float = Field(..., description="Base value of the item for the buyer")

class Ask(Order):
    """Represents an ask order."""
    base_cost: float = Field(..., description="Base cost of the item for the seller")

class AuctionLocalAction(LocalAction):
    action: Union[Bid, Ask]

class AuctionGlobalAction(GlobalAction):
    actions: Dict[str, AuctionLocalAction]

class Trade(BaseModel):
    trade_id: int = Field(..., description="Unique identifier for the trade")
    buyer_id: str = Field(..., description="ID of the buyer")
    seller_id: str = Field(..., description="ID of the seller")
    price: float = Field(..., description="The price at which the trade was executed")
    quantity: int = Field(default=1, description="The quantity traded")

class AuctionLocalObservation(LocalObservation):
    observation: Optional[Trade]

class AuctionGlobalObservation(GlobalObservation):
    observations: Dict[str, AuctionLocalObservation]
    @computed_field
    @property
    def global_obs(self) -> Optional[List[Trade]]:
        return list(set([value.observation for value in self.observations.values() if value.observation is not None])) if self.observations else None

class DoubleAuction(Mechanism):
    max_rounds: int = Field(..., description="Maximum number of auction rounds")
    current_round: int = Field(default=0, description="Current round number")
    trades: List[Trade] = Field(default_factory=list, description="List of executed trades")

    @property
    def sequential(self) -> bool:
        return False

    def step(self, action: AuctionGlobalAction) -> EnvironmentStep:
        self.current_round += 1
        new_trades = self._match_orders(action.actions)
        self.trades.extend(new_trades)

        observations = self._create_observations(new_trades)
        done = self.current_round >= self.max_rounds

        return EnvironmentStep(
            global_observation=AuctionGlobalObservation(observations=observations),
            done=done,
            info={"current_round": self.current_round}
        )

    def _match_orders(self, actions: Dict[str, AuctionLocalAction]) -> List[Trade]:
        bids = sorted([(agent_id, action.action) for agent_id, action in actions.items() if isinstance(action.action, Bid)],
                      key=lambda x: x[1].market_action.price, reverse=True)
        asks = sorted([(agent_id, action.action) for agent_id, action in actions.items() if isinstance(action.action, Ask)],
                      key=lambda x: x[1].market_action.price)
        
        trades = []
        trade_id = len(self.trades)

        for (bid_id, bid) in bids:
            for (ask_id, ask) in asks:
                if bid.market_action.price >= ask.market_action.price:
                    price = (bid.market_action.price + ask.market_action.price) / 2
                    trade = Trade(
                        trade_id=trade_id,
                        buyer_id=bid_id,
                        seller_id=ask_id,
                        price=price,
                        quantity=min(bid.market_action.quantity, ask.market_action.quantity)
                    )
                    trades.append(trade)
                    trade_id += 1
                    asks.remove((ask_id, ask))
                    break
                
        return trades

    def _create_observations(self, new_trades: List[Trade]) -> Dict[str, AuctionLocalObservation]:
        observations = {}
        for trade in new_trades:
            buyer_obs = AuctionLocalObservation(agent_id=trade.buyer_id, observation=trade)
            seller_obs = AuctionLocalObservation(agent_id=trade.seller_id, observation=trade)
            observations[trade.buyer_id] = buyer_obs
            observations[trade.seller_id] = seller_obs
        return observations

    def get_global_state(self) -> Dict[str, Any]:
        return {
            "current_round": self.current_round,
            "trades": [trade.dict() for trade in self.trades]
        }

    def reset(self) -> None:
        self.current_round = 0
        self.trades = []
