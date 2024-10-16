# stock_market.py

import logging
import random
from typing import Any, List, Dict, Union, Type, Optional, Tuple
from market_agents.stock_market.stock_agent import StockEconomicAgent
from pydantic import BaseModel, Field, field_validator
from market_agents.environments.environment import (
    EnvironmentHistory, Mechanism, LocalAction, GlobalAction, LocalObservation, GlobalObservation,
    EnvironmentStep, ActionSpace, ObservationSpace, MultiAgentEnvironment
)
from market_agents.stock_market.stock_models import OrderType, MarketAction, StockOrder, Trade
logger = logging.getLogger(__name__)


class MarketSummary(BaseModel):
    trades_count: int = Field(default=0, description="Number of trades executed")
    average_price: float = Field(default=0.0, description="Average price of trades")
    total_volume: int = Field(default=0, description="Total volume of trades")
    price_range: Tuple[float, float] = Field(default=(0.0, 0.0), description="Range of prices")


class StockMarketAction(LocalAction):
    action: MarketAction

    @field_validator('action')
    def validate_action(cls, v):
        if v.order_type in [OrderType.BUY, OrderType.SELL] and (v.quantity <= 0 or v.price <= 0):
            raise ValueError("Quantity and price must be positive for buy and sell orders")
        return v

    @classmethod
    def sample(cls, agent_id: str) -> 'StockMarketAction':
        order_type = random.choice(list(OrderType))
        if order_type == OrderType.HOLD:
            action = MarketAction(order_type=order_type)
        else:
            random_price = random.uniform(50, 150)
            random_quantity = random.randint(1, 10)
            action = MarketAction(order_type=order_type, price=random_price, quantity=random_quantity)
        return cls(agent_id=agent_id, action=action)

    @classmethod
    def action_schema(cls) -> Dict[str, Any]:
        return MarketAction.model_json_schema()


class GlobalStockMarketAction(GlobalAction):
    actions: Dict[str, StockMarketAction]


class StockMarketObservation(BaseModel):
    trades: List[Trade] = Field(default_factory=list, description="List of trades the agent participated in")
    market_summary: MarketSummary = Field(default_factory=MarketSummary, description="Summary of market activity")
    order_book_summary: Dict[str, List[Tuple[float, int]]] = Field(default_factory=dict, description="Summary of order book")
    current_price: float = Field(default=100.0, description="Current market price")
    portfolio_value: float = Field(default=0.0, description="Total value of the agent's portfolio")


class StockMarketLocalObservation(LocalObservation):
    observation: StockMarketObservation


class StockMarketGlobalObservation(GlobalObservation):
    observations: Dict[str, StockMarketLocalObservation]
    all_trades: List[Trade] = Field(default_factory=list, description="All trades executed in this round")
    market_summary: MarketSummary = Field(default_factory=MarketSummary, description="Summary of market activity")
    order_book_summary: Dict[str, List[Tuple[float, int]]] = Field(default_factory=dict, description="Summary of order book")
    current_price: float = Field(default=100.0, description="Current market price")


class StockMarketActionSpace(ActionSpace):
    allowed_actions: List[Type[LocalAction]] = [StockMarketAction]

    @classmethod
    def get_action_schema(cls) -> Dict[str, Any]:
        return MarketAction.model_json_schema()


class StockMarketObservationSpace(ObservationSpace):
    allowed_observations: List[Type[LocalObservation]] = [StockMarketLocalObservation]


class StockMarketMechanism(Mechanism):
    max_rounds: int = Field(default=100, description="Maximum number of trading rounds")
    current_round: int = Field(default=0, description="Current round number")
    trades: List[Trade] = Field(default_factory=list, description="List of executed trades")
    order_book_buy: List[StockOrder] = Field(default_factory=list, description="List of buy orders")
    order_book_sell: List[StockOrder] = Field(default_factory=list, description="List of sell orders")
    stock_symbol: str = Field(default="AAPL", description="Stock symbol being traded")
    current_price: float = Field(default=100.0, description="Current market price")
    price_history: List[float] = Field(default_factory=lambda: [100.0])
    sequential: bool = Field(default=False, description="Whether the mechanism is sequential")
    agent_registry: Dict[str, Any] = Field(default_factory=dict, description="Registry of agents")

    def step(self, action: GlobalStockMarketAction) -> EnvironmentStep:
        self.current_round += 1
        self._update_order_book(action.actions)
        new_trades = self._match_orders()
        self.trades.extend(new_trades)
        self._update_price(new_trades)

        market_summary = self._create_market_summary(new_trades)
        order_book_summary = self._get_order_book_summary()
        observations = self._create_observations(new_trades, market_summary, order_book_summary)
        done = self.current_round >= self.max_rounds

        return EnvironmentStep(
            global_observation=StockMarketGlobalObservation(
                observations=observations,
                all_trades=new_trades,
                market_summary=market_summary,
                order_book_summary=order_book_summary,
                current_price=self.current_price
            ),
            done=done,
            info={"current_round": self.current_round}
        )

    def _update_order_book(self, actions: Dict[str, StockMarketAction]):
        for agent_id, action in actions.items():
            order = StockOrder(
                agent_id=agent_id,
                order_type=action.action.order_type,
                price=action.action.price,
                quantity=action.action.quantity
            )
            if order.is_buy_order:
                self.order_book_buy.append(order)
            elif order.order_type == OrderType.SELL:
                self.order_book_sell.append(order)

    def _match_orders(self) -> List[Trade]:
        trades = []
        trade_id = len(self.trades)

        # Sort buy and sell orders
        self.order_book_buy.sort(key=lambda x: (-x.price, x.agent_id))
        self.order_book_sell.sort(key=lambda x: (x.price, x.agent_id))

        while self.order_book_buy and self.order_book_sell:
            best_buy = self.order_book_buy[0]
            best_sell = self.order_book_sell[0]

            if best_buy.price >= best_sell.price:
                trade_price = (best_buy.price + best_sell.price) / 2
                trade_quantity = min(best_buy.quantity, best_sell.quantity)

                trade = Trade(
                    trade_id=trade_id,
                    buyer_id=best_buy.agent_id,
                    seller_id=best_sell.agent_id,
                    price=trade_price,
                    bid_price=best_buy.price,
                    ask_price=best_sell.price,
                    quantity=trade_quantity,
                    stock_symbol=self.stock_symbol
                )
                trades.append(trade)
                trade_id += 1

                # Update order quantities
                best_buy.quantity -= trade_quantity
                best_sell.quantity -= trade_quantity

                if best_buy.quantity == 0:
                    self.order_book_buy.pop(0)
                if best_sell.quantity == 0:
                    self.order_book_sell.pop(0)
            else:
                break

        return trades

    def _get_order_book_summary(self) -> Dict[str, List[Tuple[float, int]]]:
        buy_orders = {}
        for order in self.order_book_buy:
            price = order.price
            quantity = order.quantity
            buy_orders[price] = buy_orders.get(price, 0) + quantity
        sell_orders = {}
        for order in self.order_book_sell:
            price = order.price
            quantity = order.quantity
            sell_orders[price] = sell_orders.get(price, 0) + quantity
        return {
            'buy': sorted(buy_orders.items(), key=lambda x: -x[0]),
            'sell': sorted(sell_orders.items(), key=lambda x: x[0])
        }

    def _update_price(self, trades: List[Trade]):
        if trades:
            prices = [trade.price for trade in trades]
            self.current_price = sum(prices) / len(prices)
            self.price_history.append(self.current_price)

    def _create_observations(self, new_trades: List[Trade], market_summary: MarketSummary, order_book_summary: Dict[str, List[Tuple[float, int]]]) -> Dict[str, StockMarketLocalObservation]:
        observations = {}
        agent_ids = set([trade.buyer_id for trade in new_trades] + [trade.seller_id for trade in new_trades])

        for agent_id in agent_ids:
            agent_trades = [trade for trade in new_trades if trade.buyer_id == agent_id or trade.seller_id == agent_id]
            agent = self.agent_registry.get(agent_id)
            if agent:
                portfolio_value = agent.calculate_portfolio_value(self.current_price)
            else:
                portfolio_value = 0.0

            observation = StockMarketObservation(
                trades=agent_trades,
                market_summary=market_summary,
                order_book_summary=order_book_summary,
                current_price=self.current_price,
                portfolio_value=portfolio_value
            )

            observations[agent_id] = StockMarketLocalObservation(
                agent_id=agent_id,
                observation=observation
            )

        return observations

    def _create_market_summary(self, trades: List[Trade]) -> MarketSummary:
        if not trades:
            return MarketSummary()

        prices = [trade.price for trade in trades]
        total_volume = sum(trade.quantity for trade in trades)
        return MarketSummary(
            trades_count=len(trades),
            average_price=sum(prices) / len(prices),
            total_volume=total_volume,
            price_range=(min(prices), max(prices))
        )

    def get_global_state(self) -> Dict[str, Any]:
        return {
            "current_round": self.current_round,
            "current_price": self.current_price,
            "price_history": self.price_history,
            "trades": [trade.model_dump() for trade in self.trades],
            "order_book_summary": self._get_order_book_summary()
        }

    def reset(self) -> None:
        self.current_round = 0
        self.trades = []
        self.order_book_buy = []
        self.order_book_sell = []
        self.current_price = 100.0
        self.price_history = [self.current_price]


class StockMarket(MultiAgentEnvironment):
    name: str = Field(default="Stock Market", description="Name of the stock market")
    action_space: StockMarketActionSpace = Field(default_factory=StockMarketActionSpace, description="Action space of the stock market")
    observation_space: StockMarketObservationSpace = Field(default_factory=StockMarketObservationSpace, description="Observation space of the stock market")
    mechanism: StockMarketMechanism = Field(default_factory=StockMarketMechanism, description="Mechanism of the stock market")
    agents: Dict[str, StockEconomicAgent] = Field(default_factory=dict, description="Dictionary of agents in the market")

    def __init__(self, **data):
        super().__init__(**data)
        self.mechanism.agent_registry = self.agents

    def __init__(self, agents: Dict[str, StockEconomicAgent], **kwargs):
        super().__init__(**kwargs)
        self.agents = agents
        self.mechanism.agent_registry = self.agents

    def reset(self) -> GlobalObservation:
        self.current_step = 0
        self.history = EnvironmentHistory()
        self.mechanism.reset()
        observations = {}

        for agent_id, agent in self.agents.items():
            portfolio_value = agent.calculate_portfolio_value(self.mechanism.current_price)
            observation = StockMarketObservation(
                trades=[],
                market_summary=MarketSummary(),
                order_book_summary=self.mechanism._get_order_book_summary(),
                current_price=self.mechanism.current_price,
                portfolio_value=portfolio_value
            )
            observations[agent_id] = StockMarketLocalObservation(
                agent_id=agent_id,
                observation=observation
            )

        return StockMarketGlobalObservation(
            observations=observations,
            all_trades=[],
            market_summary=MarketSummary(),
            order_book_summary=self.mechanism._get_order_book_summary(),
            current_price=self.mechanism.current_price
        )

    def step(self, actions: GlobalAction) -> EnvironmentStep:
        step_result = self.mechanism.step(actions)
        self.current_step += 1
        self.update_history(actions, step_result)
        return step_result

    def render(self):
        pass
