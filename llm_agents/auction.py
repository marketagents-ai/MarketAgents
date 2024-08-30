import logging
from typing import List, Optional
from pydantic import BaseModel, Field, computed_field
from environment import Environment, generate_market_agents
from ziagents import Order, Trade, ZIAgent, Bid, Ask, MarketInfo

from colorama import Fore, Style

# Set up logger
logger = logging.getLogger(__name__)

class OrderBook(BaseModel):
    bids: List[Bid] = Field(default_factory=list, description="List of current bids")
    asks: List[Ask] = Field(default_factory=list, description="List of current asks")

    def add_bid(self, bid: Bid):
        self.bids.append(bid)
        self.bids.sort(key=lambda x: x.market_action.price, reverse=True)

    def add_ask(self, ask: Ask):
        self.asks.append(ask)
        self.asks.sort(key=lambda x: x.market_action.price)

    def match_orders(self, round_num: int) -> List[Trade]:
        trades = []
        trade_counter = 0
        while self.bids and self.asks and self.bids[0].market_action.price >= self.asks[0].market_action.price:
            bid = self.bids.pop(0)
            ask = self.asks.pop(0)
            trade_price = (bid.market_action.price + ask.market_action.price) / 2  # Midpoint price
            trade = Trade(
                trade_id=trade_counter,
                bid=bid,
                ask=ask,
                price=trade_price,
                round=round_num
            )
            trades.append(trade)
            trade_counter += 1
        return trades

class DoubleAuction(BaseModel):
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
        return len(self.trade_history)

    def execute_trades(self, trades: List[Trade]):
        for trade in trades:
            buyer = self.environment.get_agent(trade.bid.agent_id)
            seller = self.environment.get_agent(trade.ask.agent_id)

            assert buyer is not None, "Buyer not found"
            assert seller is not None, "Seller not found"

            buyer.finalize_bid(trade)
            seller.finalize_ask(trade)
            self.total_surplus_extracted += trade.total_surplus
            self.average_prices.append(trade.price)
            self.successful_trades.append(trade)
            self.trade_history.append(trade)

            logger.info(f"Executing trade: Buyer {buyer.id} - Surplus: {trade.buyer_surplus:.2f}, Seller {seller.id} - Surplus: {trade.seller_surplus:.2f}")

    def generate_bids(self, market_info: MarketInfo) -> List[Bid]:
        bids = []
        for buyer in self.environment.buyers:
            bid = buyer.generate_bid(market_info)
            if bid:
                bids.append(bid)
                logger.info(f"{Fore.BLUE}Buyer {Fore.CYAN}{buyer.id}{Fore.BLUE} bid: ${Fore.GREEN}{bid.market_action.price:.2f}{Fore.BLUE} for 1 unit(s){Style.RESET_ALL}")
        return bids
    
    def generate_asks(self, market_info: MarketInfo) -> List[Ask]:
        asks = []
        for seller in self.environment.sellers:
            ask = seller.generate_ask(market_info)
            if ask:
                asks.append(ask)
                logger.info(f"{Fore.RED}Seller {Fore.CYAN}{seller.id}{Fore.RED} ask: ${Fore.GREEN}{ask.market_action.price:.2f}{Fore.RED} for 1 unit(s){Style.RESET_ALL}")
        return asks
    
    def run_auction(self):
        if self.current_round >= self.max_rounds:
            logger.info("Max rounds reached. Auction has ended.")
            return
        
        for round_num in range(self.current_round + 1, self.max_rounds + 1):
            logger.info(f"\n=== Round {round_num} ===")
            self.current_round = round_num

            # Prepare market info
            market_info = self._get_market_info()

            # Generate bids from buyers
            bids = self.generate_bids(market_info)
            # Generate asks from sellers
            asks = self.generate_asks(market_info)

            # Add bids and asks to the order book
            for bid in bids:
                self.order_book.add_bid(bid)
            for ask in asks:
                self.order_book.add_ask(ask)

            trades = self.order_book.match_orders(round_num)
            if trades:
                self.execute_trades(trades)

        self.summarize_results()

    def _get_market_info(self) -> MarketInfo:
        last_trade_price = self.average_prices[-1] if self.average_prices else None
        average_price = sum(self.average_prices) / len(self.average_prices) if self.average_prices else None
        
        # If no trades have occurred, use the midpoint of buyer and seller base values - this is a crime leaking hidden information
        if last_trade_price is None or average_price is None:
            buyer_base_value = max(agent.base_value for agent in self.environment.buyers)
            seller_base_value = min(agent.base_value for agent in self.environment.sellers)
            initial_price_estimate = (buyer_base_value + seller_base_value) / 2
            
            last_trade_price = last_trade_price or initial_price_estimate
            average_price = average_price or initial_price_estimate

        return MarketInfo(
            last_trade_price=last_trade_price,
            average_price=average_price,
            total_trades=len(self.successful_trades),
            current_round=self.current_round,
        )

    def summarize_results(self):
        total_trades = len(self.successful_trades)
        avg_price = sum(self.average_prices) / total_trades if total_trades > 0 else 0

        logger.info(f"\n=== Auction Summary ===")
        logger.info(f"Total Successful Trades: {total_trades}")
        logger.info(f"Total Surplus Extracted: {self.total_surplus_extracted:.2f}")
        logger.info(f"Average Price: {avg_price:.2f}")

        # Compare theoretical and practical surplus
        ce_price, ce_quantity, theoretical_buyer_surplus, theoretical_seller_surplus, theoretical_total_surplus = self.environment.calculate_equilibrium()
        logger.info(f"\n=== Theoretical vs. Practical Surplus ===")
        logger.info(f"Theoretical Total Surplus: {theoretical_total_surplus:.2f}")
        logger.info(f"Practical Total Surplus: {self.total_surplus_extracted:.2f}")
        logger.info(f"Difference (Practical - Theoretical): {self.total_surplus_extracted - theoretical_total_surplus:.2f}")

if __name__ == "__main__":
    # Generate ZI agents
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
    
    # Create the environment
    env = Environment(agents=agents)

    # Print initial market state
    env.print_market_state()

    # Run the auction
    auction = DoubleAuction(environment=env, max_rounds=5)
    auction.run_auction()

    # Analyze and plot results
    from analysis import analyze_and_plot_auction_results
    analyze_and_plot_auction_results(auction, env)