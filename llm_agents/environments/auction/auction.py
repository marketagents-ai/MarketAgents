import logging
import traceback
from typing import Any, List, Optional, Dict, Tuple
from pydantic import BaseModel, Field, computed_field, ValidationError
from colorama import Fore, Style

# Set up logger
logger = logging.getLogger(__name__)

class MarketAction(BaseModel):
    """Represents a market action (bid or ask)."""
    price: float = Field(..., description="Price of the order")
    quantity: int = Field(default=1, ge=1, le=1, description="Quantity of the order, constrained to 1")

class Order(BaseModel):
    """Base class for orders."""
    agent_id: int = Field(..., description="Unique identifier of the agent placing the order")
    market_action: MarketAction = Field(..., description="Market action of the order")

    @computed_field
    @property
    def is_buy(self) -> bool:
        """Determine if the order is a buy order."""
        return isinstance(self, Bid)

class Bid(Order):
    """Represents a bid order."""
    base_value: float = Field(..., description="Base value of the item for the buyer")
    
    @computed_field
    def check_bid_validity(self) -> None:
        """Validate that the bid price is not higher than the base value."""
        if self.market_action.price > self.base_value:
            raise ValueError("Bid price is higher than base value")

class Ask(Order):
    """Represents an ask order."""
    base_cost: float = Field(..., description="Base cost of the item for the seller")

    @computed_field
    def check_ask_validity(self) -> None:
        """Validate that the ask price is not lower than the base cost."""
        if self.market_action.price < self.base_cost:
            raise ValueError("Ask price is lower than base cost")

class Trade(BaseModel):
    """Represents a completed trade."""
    trade_id: int = Field(..., description="Unique identifier for the trade")
    bid: Bid = Field(..., description="The bid involved in the trade")
    ask: Ask = Field(..., description="The ask involved in the trade")
    price: float = Field(..., description="The price at which the trade was executed")
    round: int = Field(..., description="The round in which the trade occurred")
    quantity: int = 1
    buyer_value: float = Field(..., description="The buyer's value for the traded item")
    seller_cost: float = Field(..., description="The seller's cost for the traded item")

    @computed_field
    @property
    def buyer_surplus(self) -> float:
        """Calculate the buyer's surplus from the trade."""
        return self.buyer_value - self.price

    @computed_field
    @property
    def seller_surplus(self) -> float:
        """Calculate the seller's surplus from the trade."""
        return self.price - self.seller_cost

    @computed_field
    @property
    def total_surplus(self) -> float:
        """Calculate the total surplus from the trade."""
        return self.buyer_surplus + self.seller_surplus

    @computed_field
    def check_trade_validity(self) -> None:
        """Validate that the bid price is not lower than the ask price."""
        if self.bid.market_action.price < self.ask.market_action.price:
            raise ValueError("Bid price is lower than Ask price")

class MarketInfo(BaseModel):
    """Represents current market information."""
    last_trade_price: Optional[float] = Field(None, description="Price of the last executed trade")
    average_price: Optional[float] = Field(None, description="Average price of all executed trades")
    total_trades: int = Field(0, description="Total number of executed trades")
    current_round: int = Field(1, description="Current round number")

class OrderBook(BaseModel):
    """Represents the order book for the double auction."""

    bids: List[Bid] = Field(default_factory=list, description="List of current bids")
    asks: List[Ask] = Field(default_factory=list, description="List of current asks")

    def add_bid(self, bid: Bid) -> None:
        """Add a bid to the order book."""
        self.bids.append(bid)

    def add_ask(self, ask: Ask) -> None:
        """Add an ask to the order book."""
        self.asks.append(ask)

    def match_orders(self, round_num: int) -> List[Trade]:
        """Match bids and asks to create trades.

        Args:
            round_num: The current round number.

        Returns:
            A list of matched trades.
        """
        trades = []
        trade_counter = 0
        
        sorted_bids = sorted(self.bids, key=lambda x: x.market_action.price, reverse=True)
        sorted_asks = sorted(self.asks, key=lambda x: x.market_action.price)
        
        bid_index, ask_index = 0, 0
        
        while bid_index < len(sorted_bids) and ask_index < len(sorted_asks):
            bid, ask = sorted_bids[bid_index], sorted_asks[ask_index]
            
            if bid.market_action.price >= ask.market_action.price:
                trade_price = (bid.market_action.price + ask.market_action.price) / 2
                trade = Trade(
                    trade_id=trade_counter,
                    bid=bid,
                    ask=ask,
                    price=trade_price,
                    round=round_num,
                    quantity=1,
                    buyer_value=bid.base_value,
                    seller_cost=ask.base_cost
                )
                trades.append(trade)
                trade_counter += 1
                bid_index += 1
                ask_index += 1
            else:
                if bid.market_action.price < ask.market_action.price:
                    bid_index += 1
                else:
                    ask_index += 1
        
        self._remove_matched_orders(trades)
        
        return trades

    def _remove_matched_orders(self, trades: List[Trade]) -> None:
        """Remove matched orders from the order book.

        Args:
            trades: List of executed trades.
        """
        matched_bid_ids = set((trade.bid.agent_id, trade.bid.market_action.price) for trade in trades)
        matched_ask_ids = set((trade.ask.agent_id, trade.ask.market_action.price) for trade in trades)
        
        self.bids = [bid for bid in self.bids if (bid.agent_id, bid.market_action.price) not in matched_bid_ids]
        self.asks = [ask for ask in self.asks if (ask.agent_id, ask.market_action.price) not in matched_ask_ids]

class DoubleAuction(BaseModel):
    """Represents a double auction mechanism."""
    max_rounds: int = Field(..., description="Maximum number of auction rounds")
    current_round: int = Field(default=0, description="Current round number")
    successful_trades: List[Trade] = Field(default_factory=list, description="List of successful trades")
    total_surplus_extracted: float = Field(default=0.0, description="Total surplus extracted from trades")
    average_prices: List[float] = Field(default_factory=list, description="List of average prices per round")
    order_book: OrderBook = Field(default_factory=OrderBook, description="Current order book")
    trade_history: List[Trade] = Field(default_factory=list, description="Complete trade history")

    @computed_field
    @property
    def trade_counter(self) -> int:
        """Return the total number of trades executed."""
        return len(self.trade_history)
    
    def process_action(self, agent_id: int, action: Dict[str, Any], is_buyer: bool, base_value: float) -> None:
        """Process a single agent's action and update the order book.

        Args:
            agent_id: The ID of the agent taking the action.
            action: The parsed action from the agent.
            is_buyer: Whether the agent is a buyer or seller.
            base_value: The base value or cost for the agent.
        """
        market_action = MarketAction(price=action['price'], quantity=action['quantity'])
        
        if is_buyer:
            bid = Bid(agent_id=agent_id, market_action=market_action, base_value=base_value)
            self.order_book.add_bid(bid)
            logger.info(f"{Fore.BLUE}Buyer {Fore.CYAN}{agent_id}{Fore.BLUE} bid: ${Fore.GREEN}{action['price']:.2f}{Fore.BLUE} for {Fore.YELLOW}{action['quantity']}{Fore.BLUE} unit(s){Style.RESET_ALL}")
        else:
            ask = Ask(agent_id=agent_id, market_action=market_action, base_cost=base_value)
            self.order_book.add_ask(ask)
            logger.info(f"{Fore.RED}Seller {Fore.CYAN}{agent_id}{Fore.RED} ask: ${Fore.GREEN}{action['price']:.2f}{Fore.RED} for {Fore.YELLOW}{action['quantity']}{Fore.RED} unit(s){Style.RESET_ALL}")

    def execute_trades(self, trades: List[Trade]) -> List[Dict[str, Any]]:
        """Execute a list of trades and return trade information.

        Args:
            trades: List of trades to execute.

        Returns:
            List of dictionaries containing trade information for each agent involved.
        """
        trade_info = []
        for trade in trades:
            self.total_surplus_extracted += trade.total_surplus
            self.average_prices.append(trade.price)
            self.successful_trades.append(trade)
            self.trade_history.append(trade)

            logger.info(f"Executing trade: Buyer {trade.bid.agent_id} - Surplus: {trade.buyer_surplus:.2f}, "
                        f"Seller {trade.ask.agent_id} - Surplus: {trade.seller_surplus:.2f}")

            buyer_info = {
                "agent_id": trade.bid.agent_id,
                "is_buyer": True,
                "price": trade.price,
                "quantity": trade.quantity,
                "surplus": trade.buyer_surplus
            }
            seller_info = {
                "agent_id": trade.ask.agent_id,
                "is_buyer": False,
                "price": trade.price,
                "quantity": trade.quantity,
                "surplus": trade.seller_surplus
            }
            trade_info.extend([buyer_info, seller_info])

        return trade_info

    def update_auction_state(self) -> List[Dict[str, Any]]:
        """Update the auction state by matching orders and executing trades.

        Returns:
            List of dictionaries containing trade information for each agent involved.
        """
        trades = self.order_book.match_orders(self.current_round)
        if trades:
            return self.execute_trades(trades)
        return []

    def get_market_info(self) -> MarketInfo:
        """Get current market information."""
        last_trade_price = self.average_prices[-1] if self.average_prices else None
        average_price = sum(self.average_prices) / len(self.average_prices) if self.average_prices else None
        
        return MarketInfo(
            last_trade_price=last_trade_price,
            average_price=average_price,
            total_trades=len(self.successful_trades),
            current_round=self.current_round,
        )

    def get_current_trade_execution(self, agent_id: int) -> Dict[str, Any]:
        """Get the current round trade execution information for a specific agent.

        Args:
            agent_id (int): The ID of the agent requesting information.

        Returns:
            Dict[str, Any]: A dictionary containing trade execution information for the agent.
        """
        agent_trades = [trade for trade in self.successful_trades if trade.bid.agent_id == agent_id or trade.ask.agent_id == agent_id]
        
        if not agent_trades:
            return {
                "agent_id": agent_id,
                "round": self.current_round,
                "price": 0,
                "quantity": 0,
                "reward": 0
            }
        
        latest_trade = agent_trades[-1]  # Get the most recent trade for this agent
        is_buyer = latest_trade.bid.agent_id == agent_id
        
        return {
            "agent_id": agent_id,
            "round": self.current_round,
            "price": latest_trade.price,
            "quantity": latest_trade.quantity,
            "reward": latest_trade.buyer_surplus if is_buyer else latest_trade.seller_surplus
        }

    def advance_round(self) -> None:
        """Advance the auction to the next round."""
        self.current_round += 1
        logger.info(f"\n=== Starting Round {self.current_round} ===")

    def is_auction_complete(self) -> bool:
        """Check if the auction is complete."""
        return self.current_round >= self.max_rounds

    def summarize_results(self) -> None:
        """Summarize and log the auction results."""
        total_trades = len(self.successful_trades)
        avg_price = sum(self.average_prices) / total_trades if total_trades > 0 else 0

        logger.info("\n=== Auction Summary ===")
        logger.info(f"Total Successful Trades: {total_trades}")
        logger.info(f"Total Surplus Extracted: {self.total_surplus_extracted:.2f}")
        logger.info(f"Average Price: {avg_price:.2f}")

    def reset(self) -> None:
        """Reset the auction to its initial state."""
        self.current_round = 0
        self.successful_trades = []
        self.total_surplus_extracted = 0.0
        self.average_prices = []
        self.order_book = OrderBook()
        self.trade_history = []
        logger.info("Auction has been reset to its initial state.")
