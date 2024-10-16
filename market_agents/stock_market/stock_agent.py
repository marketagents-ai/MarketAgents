from typing import List, Dict, Optional
from pydantic import BaseModel, Field, computed_field
import random
import logging
from market_agents.stock_market.stock_models import (
    MarketAction,
    OrderType,
    StockOrder,
    Trade,
    Endowment,
    Portfolio,
    Stock,
)
from market_agents.economics.econ_agent import EconomicAgent as BaseEconomicAgent

# Set up logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class StockEconomicAgent(BaseEconomicAgent):
    endowment: Endowment
    max_relative_spread: float = Field(default=0.05)
    pending_orders: List[MarketAction] = Field(default_factory=list)
    archived_endowments: List[Endowment] = Field(default_factory=list)
    risk_aversion: float = Field(default=0.5)
    expected_return: float = Field(default=0.05)
    stock_symbol: str = Field(default="AAPL")

    def archive_endowment(self, new_portfolio: Optional[Portfolio] = None):
        self.archived_endowments.append(self.endowment.model_copy(deep=True))
        if new_portfolio is None:
            new_endowment = self.endowment.model_copy(deep=True, update={"trades": []})
        else:
            new_endowment = self.endowment.model_copy(deep=True, update={"trades": [], "initial_portfolio": new_portfolio})
        self.endowment = new_endowment

    @computed_field
    @property
    def current_portfolio(self) -> Portfolio:
        return self.endowment.current_portfolio

    @computed_field
    @property
    def current_cash(self) -> float:
        return self.current_portfolio.cash

    @computed_field
    @property
    def current_stock_quantity(self) -> int:
        return self.current_portfolio.get_stock_quantity(self.stock_symbol)

    def generate_order(self, market_price: float) -> Optional[MarketAction]:
        expected_value = market_price * (1 + self.expected_return)
        price_variation = market_price * self.max_relative_spread

        if expected_value > market_price:
            max_buy_price = market_price + price_variation
            price = random.uniform(market_price, max_buy_price)
            quantity = self.determine_quantity_to_buy(price)
            if quantity > 0:
                order = MarketAction(order_type=OrderType.BUY, price=price, quantity=quantity)
                self.pending_orders.append(order)
                return order
        elif expected_value < market_price:
            min_sell_price = market_price - price_variation
            price = random.uniform(min_sell_price, market_price)
            quantity = self.determine_quantity_to_sell()
            if quantity > 0:
                order = MarketAction(order_type=OrderType.SELL, price=price, quantity=quantity)
                self.pending_orders.append(order)
                return order
        else:
            return None

    def determine_quantity_to_buy(self, price: float) -> int:
        affordable_quantity = int(self.current_cash // price)
        if affordable_quantity <= 0:
            return 0
        quantity = max(1, int(affordable_quantity * (1 - self.risk_aversion)))
        return quantity

    def determine_quantity_to_sell(self) -> int:
        current_quantity = self.current_stock_quantity
        if current_quantity <= 0:
            return 0
        quantity = max(1, int(current_quantity * self.risk_aversion))
        return quantity

    def process_trade(self, trade: Trade):
        matching_order = next((order for order in self.pending_orders if order.price == (trade.bid_price if trade.buyer_id == self.id else trade.ask_price) and order.quantity == trade.quantity), None)
        if matching_order:
            self.pending_orders.remove(matching_order)
        else:
            logger.warning(f"Trade processed but matching order not found for agent {self.id}")

        self.endowment.add_trade(trade)

    def reset_pending_orders(self):
        self.pending_orders = []

    def calculate_portfolio_value(self, market_price: float) -> float:
        stock_value = self.current_stock_quantity * market_price
        total_value = self.current_cash + stock_value
        return total_value

    def print_status(self, market_price: float):
        print(f"\nAgent ID: {self.id}")
        print(f"Current Portfolio:")
        print(f"  Cash: {self.current_cash:.2f}")
        print(f"  {self.stock_symbol} Stocks: {self.current_stock_quantity}")
        portfolio_value = self.calculate_portfolio_value(market_price)
        print(f"Total Portfolio Value: {portfolio_value:.2f}")

    # Override methods from BaseEconomicAgent that are not applicable to stock trading
    def is_buyer(self, good_name: str) -> bool:
        return good_name == self.stock_symbol

    def is_seller(self, good_name: str) -> bool:
        return good_name == self.stock_symbol

    def get_current_value(self, good_name: str) -> Optional[float]:
        if good_name == self.stock_symbol:
            return self.calculate_portfolio_value(market_price=1.0)  # Use a placeholder market price
        return None

    def get_current_cost(self, good_name: str) -> Optional[float]:
        if good_name == self.stock_symbol:
            return self.calculate_portfolio_value(market_price=1.0)  # Use a placeholder market price
        return None

    def calculate_utility(self, portfolio: Portfolio) -> float:
        # For simplicity, we'll use the portfolio value as utility
        return self.calculate_portfolio_value(market_price=1.0)  # Use a placeholder market price

    def calculate_individual_surplus(self) -> float:
        # For simplicity, we'll return 0 as we don't have a clear notion of surplus in this context
        return 0.0