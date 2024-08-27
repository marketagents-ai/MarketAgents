import logging
from typing import List
from environment import Environment, generate_market_agents
from ziagents import Order, Trade, ZIAgent, Bid, Ask

from colorama import Fore, Style

# Set up logger
logger = logging.getLogger(__name__)

class DoubleAuction:
    def __init__(self, environment: Environment, max_rounds: int):
        self.environment = environment
        self.max_rounds = max_rounds
        self.current_round = 0
        self.successful_trades: List[Trade] = []
        self.total_surplus_extracted = 0.0
        self.average_prices: List[float] = []
        self.order_book = []  # Store current order book
        self.trade_history = []  # Store trade history
        self.trade_counter = 0

    def match_orders(self, bids: List[Bid], asks: List[Ask], round_num: int) -> List[Trade]:
        trades = []
        bids.sort(key=lambda x: x.price, reverse=True)  # Highest bids first
        asks.sort(key=lambda x: x.price)  # Lowest asks first

        while bids and asks and bids[0].price >= asks[0].price:
            bid = bids.pop(0)
            ask = asks.pop(0)
            trade_price = (bid.price + ask.price) / 2  # Midpoint price
            trade = Trade(
                trade_id=self.trade_counter,
                bid=bid,
                ask=ask,
                price=trade_price,
                round=round_num
            )
            trades.append(trade)
            self.trade_counter += 1
            self.order_book.append({'price': trade_price, 'shares': 1, 'total': trade_price})  # Update order book
        return trades

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
            self.trade_history.append(trade)  # Update trade history

            logger.info(f"Executing trade: Buyer {buyer.id} - Surplus: {trade.buyer_surplus:.2f}, Seller {seller.id} - Surplus: {trade.seller_surplus:.2f}")

    def generate_bids(self, market_info: dict) -> List[Bid]:
        bids = []
        for buyer in self.environment.buyers:
            bid = buyer.generate_bid(market_info)
            if bid:
                bids.append(bid)
                logger.info(f"{Fore.BLUE}Buyer {Fore.CYAN}{buyer.id}{Fore.BLUE} bid: ${Fore.GREEN}{bid.price:.2f}{Fore.BLUE} for 1 unit(s){Style.RESET_ALL}")
        return bids
    
    def generate_asks(self, market_info: dict) -> List[Ask]:
        asks = []
        for seller in self.environment.sellers:
            ask = seller.generate_ask(market_info)
            if ask:
                asks.append(ask)
                logger.info(f"{Fore.RED}Seller {Fore.CYAN}{seller.id}{Fore.RED} ask: ${Fore.GREEN}{ask.price:.2f}{Fore.RED} for 1 unit(s){Style.RESET_ALL}")
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

            trades = self.match_orders(bids, asks, round_num)
            if trades:
                self.execute_trades(trades)

        self.summarize_results()

    def _get_market_info(self) -> dict:
        last_trade_price = self.average_prices[-1] if self.average_prices else None
        average_price = sum(self.average_prices) / len(self.average_prices) if self.average_prices else None
        
        # If no trades have occurred, use the midpoint of buyer and seller base values
        if last_trade_price is None or average_price is None:
            buyer_base_value = max(agent.base_value for agent in self.environment.buyers)
            seller_base_value = min(agent.base_value for agent in self.environment.sellers)
            initial_price_estimate = (buyer_base_value + seller_base_value) / 2
            
            last_trade_price = last_trade_price or initial_price_estimate
            average_price = average_price or initial_price_estimate

        return {
            "last_trade_price": last_trade_price,
            "average_price": average_price,
            "total_trades": len(self.successful_trades),
            "current_round": self.current_round,
        }

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

    def get_order_book(self):
        return self.order_book

    def get_trade_history(self):
        return self.trade_history


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