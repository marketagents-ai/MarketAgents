# stock_models.py

from pydantic import BaseModel, Field, computed_field, model_validator
from functools import cached_property
from typing import List, Dict, Self, Optional
from enum import Enum
from copy import deepcopy
from datetime import datetime
import os
import json
import tempfile
from pathlib import Path


class SavableBaseModel(BaseModel):
    name: str

    def save_to_json(self, folder_path: str) -> str:
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.name.replace(' ', '_')}_{timestamp}.json"
        file_path = os.path.join(folder_path, filename)

        try:
            data = self.model_dump(mode='json')
            with tempfile.NamedTemporaryFile('w', delete=False) as temp_file:
                json.dump(data, temp_file, indent=2)
            os.replace(temp_file.name, file_path)
            print(f"State saved to {file_path}")
        except Exception as e:
            print(f"Error saving state to {file_path}")
            print(f"Error message: {str(e)}")
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
            raise

        return file_path

    @classmethod
    def load_from_json(cls, file_path: str) -> Self:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return cls.model_validate(data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}")
            print(f"Error message: {str(e)}")
            with open(file_path, 'r') as f:
                print(f"File contents:\n{f.read()}")
            raise


class OrderType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class MarketAction(BaseModel):
    order_type: OrderType = Field(..., description="Type of order: 'buy', 'sell', or 'hold'")
    price: Optional[float] = Field(None, description="Price of the order (not applicable for 'hold')")
    quantity: Optional[int] = Field(None, ge=0, description="Quantity of the order (not applicable for 'hold')")

    @model_validator(mode='after')
    def validate_order_type_and_fields(self):
        if self.order_type == OrderType.HOLD:
            if self.price is not None or self.quantity is not None:
                raise ValueError("Price and quantity must be None for 'hold' orders")
        else:
            if self.price is None or self.quantity is None:
                raise ValueError("Price and quantity must be specified for 'buy' and 'sell' orders")
            if self.price <= 0 or self.quantity <= 0:
                raise ValueError("Price and quantity must be positive for 'buy' and 'sell' orders")
        return self


class StockOrder(MarketAction):
    agent_id: str

    @computed_field
    @property
    def is_buy_order(self) -> bool:
        return self.order_type == OrderType.BUY


class Trade(BaseModel):
    trade_id: int = Field(..., description="Unique identifier for the trade")
    buyer_id: str = Field(..., description="ID of the buyer")
    seller_id: str = Field(..., description="ID of the seller")
    price: float = Field(..., description="The price at which the trade was executed")
    bid_price: float = Field(ge=0, description="The bid price")
    ask_price: float = Field(ge=0, description="The ask price")
    quantity: int = Field(default=1, description="The quantity traded")
    stock_symbol: str = Field(default="AAPL", description="The symbol of the stock traded")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the trade")

    @model_validator(mode='after')
    def rational_trade(self):
        if self.ask_price > self.bid_price:
            raise ValueError(f"Ask price {self.ask_price} is more than bid price {self.bid_price}")
        return self


class Position(BaseModel):
    quantity: int
    purchase_price: float


class Stock(BaseModel):
    symbol: str
    positions: List[Position] = Field(default_factory=list)

    def total_quantity(self) -> int:
        return sum(position.quantity for position in self.positions)

    def average_cost(self) -> float:
        total_quantity = self.total_quantity()
        if total_quantity == 0:
            return 0.0
        total_cost = sum(position.quantity * position.purchase_price for position in self.positions)
        return total_cost / total_quantity


class Portfolio(BaseModel):
    cash: float
    stocks: List[Stock] = Field(default_factory=list)

    @computed_field
    @cached_property
    def stocks_dict(self) -> Dict[str, int]:
        return {stock.symbol: stock.total_quantity() for stock in self.stocks}

    def update_stock(self, symbol: str, quantity: int, purchase_price: Optional[float] = None):
        stock = next((s for s in self.stocks if s.symbol == symbol), None)
        if stock is None:
            if quantity > 0:
                stock = Stock(symbol=symbol, positions=[])
                self.stocks.append(stock)
            else:
                return
        if stock is not None:
            if purchase_price is not None and quantity > 0:
                stock.positions.append(Position(quantity=quantity, purchase_price=purchase_price))
            elif quantity < 0:
                self.sell_stock_positions(stock, -quantity)
            if stock.total_quantity() == 0:
                self.stocks.remove(stock)

    def sell_stock_positions(self, stock: Stock, quantity_to_sell: int):
        positions = stock.positions
        quantity_remaining = quantity_to_sell
        while quantity_remaining > 0 and positions:
            position = positions[0]
            if position.quantity > quantity_remaining:
                position.quantity -= quantity_remaining
                quantity_remaining = 0
            else:
                quantity_remaining -= position.quantity
                positions.pop(0)

    def get_stock_quantity(self, symbol: str) -> int:
        stock = next((s for s in self.stocks if s.symbol == symbol), None)
        if stock:
            return stock.total_quantity()
        return 0


class Endowment(BaseModel):
    initial_portfolio: Portfolio
    trades: List[Trade] = Field(default_factory=list)
    agent_id: str

    @computed_field
    @property
    def current_portfolio(self) -> Portfolio:
        temp_portfolio = deepcopy(self.initial_portfolio)

        for trade in self.trades:
            if trade.buyer_id == self.agent_id:
                temp_portfolio.cash -= trade.price * trade.quantity
                temp_portfolio.update_stock(
                    symbol=trade.stock_symbol,
                    quantity=trade.quantity,
                    purchase_price=trade.price
                )
            elif trade.seller_id == self.agent_id:
                temp_portfolio.cash += trade.price * trade.quantity
                temp_portfolio.update_stock(
                    symbol=trade.stock_symbol,
                    quantity=-trade.quantity
                )
            else:
                raise ValueError(f"Trade {trade} not for agent {self.agent_id}")

        return temp_portfolio

    def add_trade(self, trade: Trade):
        self.trades.append(trade)
        if 'current_portfolio' in self.__dict__:
            del self.__dict__['current_portfolio']

    def simulate_trade(self, trade: Trade) -> Portfolio:
        temp_portfolio = deepcopy(self.current_portfolio)

        if trade.buyer_id == self.agent_id:
            temp_portfolio.cash -= trade.price * trade.quantity
            temp_portfolio.update_stock(
                symbol=trade.stock_symbol,
                quantity=trade.quantity,
                purchase_price=trade.price
            )
        elif trade.seller_id == self.agent_id:
            temp_portfolio.cash += trade.price * trade.quantity
            temp_portfolio.update_stock(
                symbol=trade.stock_symbol,
                quantity=-trade.quantity
            )

        return temp_portfolio
