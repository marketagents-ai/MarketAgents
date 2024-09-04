import logging
import traceback
from typing import List, Optional

from pydantic import BaseModel, Field, computed_field, ValidationError
from colorama import Fore, Style

from llm_agents.environments.environment import Environment, generate_market_agents
from ziagents import Order, Trade, ZIAgent, Bid, Ask, MarketInfo, MarketAction
from acl_message.acl_message import ACLMessage, Performative

# Set up logger
logger = logging.getLogger(__name__)


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

    environment: Environment = Field(..., description="The market environment")
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

    def execute_trades(self, trades: List[Trade]) -> None:
        """Execute a list of trades.

        Args:
            trades: List of trades to execute.
        """
        for trade in trades:
            buyer = self.environment.get_agent(trade.bid.agent_id)
            seller = self.environment.get_agent(trade.ask.agent_id)

            assert buyer is not None, "Buyer not found"
            assert seller is not None, "Seller not found"

            buyer.finalize_trade(trade)
            seller.finalize_trade(trade)
            self.total_surplus_extracted += trade.total_surplus
            self.average_prices.append(trade.price)
            self.successful_trades.append(trade)
            self.trade_history.append(trade)

            logger.info(f"Executing trade: Buyer {buyer.zi_agent.id} - Surplus: {trade.buyer_surplus:.2f}, "
                        f"Seller {seller.zi_agent.id} - Surplus: {trade.seller_surplus:.2f}")

            buyer_message = self.create_trade_message(trade, is_buyer=True)
            seller_message = self.create_trade_message(trade, is_buyer=False)
            buyer.receive_message(buyer_message)
            seller.receive_message(seller_message)

    def generate_bids(self, market_info: MarketInfo) -> List[Bid]:
        """Generate bids from buyers.

        Args:
            market_info: Current market information.

        Returns:
            List of generated bids.
        """
        bids = []
        for buyer in self.environment.buyers:
            try:
                if buyer.zi_agent.allocation.goods < buyer.zi_agent.preference_schedule.values.get(len(buyer.zi_agent.preference_schedule.values), 0):
                    bid = buyer.generate_bid(market_info, self.current_round)
                    if bid and isinstance(bid, ACLMessage):
                        market_action = MarketAction(action='bid', price=bid.content.price, quantity=bid.content.quantity)
                        base_value = buyer.zi_agent.preference_schedule.get_value(buyer.zi_agent.allocation.goods + 1)
                        bids.append(Bid(agent_id=buyer.zi_agent.id, market_action=market_action, base_value=base_value, base_cost=0))
                        logger.info(f"{Fore.BLUE}Buyer {Fore.CYAN}{buyer.zi_agent.id}{Fore.BLUE} bid: ${Fore.GREEN}{bid.content.price:.2f}{Fore.BLUE} for {Fore.YELLOW}{bid.content.quantity}{Fore.BLUE} unit(s){Style.RESET_ALL}")
            except Exception as e:
                logger.error(f"Error generating bid for buyer {buyer.zi_agent.id}: {e}")
                logger.error(traceback.format_exc())
        return bids

    def generate_asks(self, market_info: MarketInfo) -> List[Ask]:
        """Generate asks from sellers.

        Args:
            market_info: Current market information.

        Returns:
            List of generated asks.
        """
        asks = []
        for seller in self.environment.sellers:
            try:
                if seller.zi_agent.allocation.goods > 0:
                    ask = seller.generate_bid(market_info, self.current_round)
                    if ask and isinstance(ask, ACLMessage):
                        market_action = MarketAction(action='ask', price=ask.content.price, quantity=ask.content.quantity)
                        base_cost = seller.zi_agent.preference_schedule.get_value(seller.zi_agent.allocation.goods)
                        try:
                            new_ask = Ask(agent_id=seller.zi_agent.id, market_action=market_action, base_value=0, base_cost=base_cost)
                            if new_ask.market_action.price >= new_ask.base_cost:
                                asks.append(new_ask)
                                logger.info(f"{Fore.RED}Seller {Fore.CYAN}{seller.zi_agent.id}{Fore.RED} ask: ${Fore.GREEN}{ask.content.price:.2f}{Fore.RED} for {Fore.YELLOW}{ask.content.quantity}{Fore.RED} unit(s){Style.RESET_ALL}")
                            else:
                                logger.warning(f"Ask price ${ask.content.price:.2f} is lower than base cost ${base_cost:.2f} for seller {seller.zi_agent.id}. Skipping this ask.")
                        except ValidationError as ve:
                            logger.error(f"Validation error for seller {seller.zi_agent.id}: {ve}")
            except Exception as e:
                logger.error(f"Error generating ask for seller {seller.zi_agent.id}: {e}")
                logger.error(traceback.format_exc())
        return asks
    
    def run_auction(self):
        """Run the double auction simulation."""
        if self.current_round >= self.max_rounds:
            logger.info("Max rounds reached. Auction has ended.")
            return
        
        for round_num in range(self.current_round + 1, self.max_rounds + 1):
            logger.info(f"\n=== Starting Round {round_num} ===")
            self.current_round = round_num

            market_info = self._get_market_info()

            bids = self.generate_bids(market_info)
            asks = self.generate_asks(market_info)

            logger.info(f"Generated {len(bids)} bids and {len(asks)} asks")

            for bid in bids:
                self.order_book.add_bid(bid)
            for ask in asks:
                self.order_book.add_ask(ask)

            trades = self.order_book.match_orders(round_num)
            if trades:
                self.execute_trades(trades)
            
            logger.info(f"=== Finished Round {round_num} ===\n")

        logger.info("Auction completed. Summarizing results.")
        self.summarize_results()

    def _get_market_info(self) -> MarketInfo:
        """Get current market information.

        Returns:
            Current market information.
        """
        last_trade_price = self.average_prices[-1] if self.average_prices else None
        average_price = sum(self.average_prices) / len(self.average_prices) if self.average_prices else None
        
        if last_trade_price is None or average_price is None:
            buyer_base_value = max(agent.zi_agent.preference_schedule.get_value(agent.zi_agent.allocation.goods + 1)
                                   for agent in self.environment.buyers)
            seller_base_value = min(agent.zi_agent.preference_schedule.get_value(agent.zi_agent.allocation.goods + 1)
                                    for agent in self.environment.sellers)
            initial_price_estimate = (buyer_base_value + seller_base_value) / 2
            
            last_trade_price = last_trade_price or initial_price_estimate
            average_price = average_price or initial_price_estimate

        return MarketInfo(
            last_trade_price=last_trade_price,
            average_price=average_price,
            total_trades=len(self.successful_trades),
            current_round=self.current_round,
        )

    def summarize_results(self) -> None:
        """Summarize and log the auction results."""
        total_trades = len(self.successful_trades)
        avg_price = sum(self.average_prices) / total_trades if total_trades > 0 else 0

        logger.info("\n=== Auction Summary ===")
        logger.info(f"Total Successful Trades: {total_trades}")
        logger.info(f"Total Surplus Extracted: {self.total_surplus_extracted:.2f}")
        logger.info(f"Average Price: {avg_price:.2f}")

        ce_price, ce_quantity, theoretical_buyer_surplus, theoretical_seller_surplus, theoretical_total_surplus = (
            self.environment.calculate_equilibrium())
        logger.info("\n=== Theoretical vs. Practical Surplus ===")
        logger.info(f"Theoretical Total Surplus: {theoretical_total_surplus:.2f}")
        logger.info(f"Practical Total Surplus: {self.total_surplus_extracted:.2f}")
        logger.info(f"Difference (Practical - Theoretical): "
                    f"{self.total_surplus_extracted - theoretical_total_surplus:.2f}")

    def create_trade_message(self, trade: Trade, is_buyer: bool) -> ACLMessage:
        """Create a trade message for the given trade.

        Args:
            trade: The trade to create a message for.
            is_buyer: Whether the message is for the buyer or seller.

        Returns:
            The created trade message.
        """
        content = {
            "trade_id": trade.trade_id,
            "price": trade.price,
            "quantity": trade.quantity,
            "round": trade.round
        }
        
        message = ACLMessage.create_message(
            performative=Performative.INFORM,
            sender="market",
            receiver=str(trade.bid.agent_id if is_buyer else trade.ask.agent_id),
            content=content,
            protocol="double-auction",
            ontology="market-ontology",
            conversation_id=f"trade-{trade.trade_id}"
        )
        logger.info(f"Created trade message: {message}")
        return message


def run_simulation():
    """Run a sample double auction simulation."""
    num_buyers = 5
    num_sellers = 5
    spread = 0.5

    agents = generate_market_agents(
        num_agents=num_buyers + num_sellers, 
        num_units=5, 
        buyer_base_value=100, 
        seller_base_value=80, 
        spread=spread
    )
    
    env = Environment(agents=agents)
    env.print_market_state()

    auction = DoubleAuction(environment=env, max_rounds=5)
    auction.run_auction()

    from analysis import analyze_and_plot_auction_results
    analyze_and_plot_auction_results(auction, env)

if __name__ == "__main__":
    run_simulation()
