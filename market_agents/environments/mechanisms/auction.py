import logging
import traceback
from typing import Any, List, Optional, Dict, Union
from pydantic import BaseModel, Field, computed_field
from colorama import Fore, Style
from market_agents.environments.environment import (
    Mechanism, LocalAction, GlobalAction, LocalObservation, GlobalObservation,
    LocalEnvironmentStep, EnvironmentStep, ActionSpace, ObservationSpace,
    FloatAction
)
import random
from typing import Type
from market_agents.economics.econ_models import Bid, Ask, Trade
# Set up logger
logger = logging.getLogger(__name__)


class AuctionAction(LocalAction):
    action: Union[Bid, Ask]

    @classmethod
    def sample(cls, agent_id: str) -> 'AuctionAction':
        is_buyer = random.choice([True, False])
        random_price = random.uniform(0, 100)
        random_quantity = random.randint(1, 5)
        action = Bid(price=random_price, quantity=random_quantity) if is_buyer else Ask(price=random_price, quantity=random_quantity)
        return cls(agent_id=agent_id, action=action)

class GlobalAuctionAction(GlobalAction):
    actions: Dict[str, AuctionAction]


class AuctionObservation(BaseModel):
    trades: List[Trade] = Field(default_factory=list, description="List of trades the agent participated in")
    market_summary: Dict[str, Any] = Field(default_factory=dict, description="Summary of market activity")
    waiting_orders: List[Union[Ask, Bid]] = Field(default_factory=list, description="List of orders waiting to be executed")

class AuctionLocalObservation(LocalObservation):
    observation: AuctionObservation

class AuctionGlobalObservation(GlobalObservation):
    observations: Dict[str, AuctionLocalObservation]
    all_trades: List[Trade] = Field(default_factory=list, description="All trades executed in this round")
    market_summary: Dict[str, Any] = Field(default_factory=dict, description="Summary of market activity")

    @computed_field
    @property
    def participants_ids(self) -> List[str]:
        return list(set([trade.buyer_id for trade in self.all_trades] + [trade.seller_id for trade in self.all_trades]))

class AuctionActionSpace(ActionSpace):
    allowed_actions: List[Type[LocalAction]] = [AuctionAction]

class AuctionObservationSpace(ObservationSpace):
    allowed_observations: List[Type[LocalObservation]] = [AuctionLocalObservation]

class DoubleAuction(Mechanism):
    max_rounds: int = Field(..., description="Maximum number of auction rounds")
    current_round: int = Field(default=0, description="Current round number")
    trades: List[Trade] = Field(default_factory=list, description="List of executed trades")
    action_space: AuctionActionSpace = Field(default_factory=AuctionActionSpace)
    observation_space: AuctionObservationSpace = Field(default_factory=AuctionObservationSpace)
    waiting_orders: List[AuctionAction] = Field(default_factory=list, description="List of waiting orders")

    sequential: bool = Field(default=False, description="Whether the mechanism is sequential")
    
    @computed_field
    @property
    def participants_ids(self) -> List[str]:
        return list(set([order.agent_id for order in self.waiting_orders] + self.trade_participants_ids))
    
    @computed_field
    @property
    def trade_participants_ids(self) -> List[str]:
        return list(set([trade.buyer_id for trade in self.trades] + [trade.seller_id for trade in self.trades]))

    def step(self, action: GlobalAuctionAction) -> EnvironmentStep:
        self.current_round += 1
        self._update_waiting_orders(action.actions)
        new_trades = self._match_orders()
        self.trades.extend(new_trades)

        market_summary = self._create_market_summary(new_trades)
        observations = self._create_observations(new_trades, market_summary)
        done = self.current_round >= self.max_rounds

        return EnvironmentStep(
            global_observation=AuctionGlobalObservation(
                observations=observations,
                all_trades=new_trades,
                market_summary=market_summary
            ),
            done=done,
            info={"current_round": self.current_round}
        )

    def _update_waiting_orders(self, actions: Dict[str, AuctionAction]):
        self.waiting_orders.extend(actions.values())

    def _match_orders(self) -> List[Trade]:
        bids = sorted([order for order in self.waiting_orders if isinstance(order.action, Bid)],
                      key=lambda x: x.action.price, reverse=True)
        asks = sorted([order for order in self.waiting_orders if isinstance(order.action, Ask)],
                      key=lambda x: x.action.price)
        
        trades = []
        trade_id = len(self.trades)

        for bid in bids[:]:  # Create a copy of the list to iterate over
            for ask in asks[:]:  # Create a copy of the list to iterate over
                if bid.action.price >= ask.action.price:
                    price = (bid.action.price + ask.action.price) / 2
                    quantity = min(bid.action.quantity, ask.action.quantity)
                    trade = Trade(
                        trade_id=trade_id,
                        buyer_id=bid.agent_id,
                        seller_id=ask.agent_id,
                        price=price,
                        quantity=quantity
                    )
                    trades.append(trade)
                    trade_id += 1

                    # Update quantities
                    bid.action.quantity -= quantity
                    ask.action.quantity -= quantity

                    # Remove orders if fully executed
                    if bid.action.quantity == 0:
                        bids.remove(bid)
                        self.waiting_orders.remove(bid)
                    if ask.action.quantity == 0:
                        asks.remove(ask)
                        self.waiting_orders.remove(ask)

                    if bid.action.quantity == 0:
                        break  # Move to next bid if current bid is fully executed
                
        return trades

    def _create_observations(self, new_trades: List[Trade], market_summary: Dict[str, Any]) -> Dict[str, AuctionLocalObservation]:
        observations = {}
        for agent_id in self.participants_ids:
            agent_trades = [trade for trade in new_trades if trade.buyer_id == agent_id or trade.seller_id == agent_id]
            agent_waiting_orders = [order.action for order in self.waiting_orders if order.agent_id == agent_id]
            
            observation = AuctionObservation(
                trades=agent_trades,  # Changed from single trade to list of trades
                market_summary=market_summary,
                waiting_orders=agent_waiting_orders
            )
            
            observations[agent_id] = AuctionLocalObservation(
                agent_id=agent_id,
                observation=observation
            )
        
        return observations

    def get_global_state(self) -> Dict[str, Any]:
        return {
            "current_round": self.current_round,
            "trades": [trade.model_dump() for trade in self.trades],
            "waiting_orders": [order.action.model_dump() for order in self.waiting_orders]
        }

    def reset(self) -> None:
        self.current_round = 0
        self.trades = []
        self.waiting_orders = []

    def _create_market_summary(self, trades: List[Trade]) -> Dict[str, Any]:
        if not trades:
            return {
                "trades_count": 0,
                "average_price": None,
                "total_volume": 0,
                "price_range": None
            }
        
        prices = [trade.price for trade in trades]
        volumes = [trade.quantity for trade in trades]
        return {
            "trades_count": len(trades),
            "average_price": sum(prices) / len(prices),
            "total_volume": sum(volumes),
            "price_range": (min(prices), max(prices))
        }
