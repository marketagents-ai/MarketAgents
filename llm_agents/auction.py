import logging
import traceback
from typing import List
from environment import Environment, generate_llm_market_agents
from acl_message import ACLMessage, Performative
from ziagents import Order, Trade
from plotter import analyze_and_plot_auction_results

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

    def match_orders(self, bids: List[Order], asks: List[Order], round_num: int) -> List[Trade]:
        trades = []
        bids.sort(key=lambda x: x.price, reverse=True)  # Highest bids first
        asks.sort(key=lambda x: x.price)  # Lowest asks first

        while bids and asks and bids[0].price >= asks[0].price:
            bid = bids.pop(0)
            ask = asks.pop(0)
            trade_price = (bid.price + ask.price) / 2  # Midpoint price
            trade_quantity = min(bid.quantity, ask.quantity)
            trade = Trade(
                trade_id=self.trade_counter,
                buyer_id=bid.agent_id,
                seller_id=ask.agent_id,
                quantity=trade_quantity,
                price=trade_price,
                buyer_value=bid.base_value,
                seller_cost=ask.base_cost,
                round=round_num
            )
            trades.append(trade)
            self.trade_counter += 1
            self.order_book.append({'price': trade_price, 'shares': trade_quantity, 'total': trade_price * trade_quantity})

            logger.info(f"Trade matched: Buyer {trade.buyer_id}, Seller {trade.seller_id}, Price: {trade_price}, Quantity: {trade_quantity}")

            # Send messages immediately after trade is completed
            buyer_message = self.create_trade_message(trade, is_buyer=True)
            seller_message = self.create_trade_message(trade, is_buyer=False)
            
            buyer = next(agent for agent in self.environment.agents if agent.zi_agent.id == trade.buyer_id)
            seller = next(agent for agent in self.environment.agents if agent.zi_agent.id == trade.seller_id)
            
            logger.info(f"Sending trade message to buyer {buyer.zi_agent.id}")
            buyer.receive_message(buyer_message)
            logger.info(f"Sending trade message to seller {seller.zi_agent.id}")
            seller.receive_message(seller_message)

        logger.info(f"Matched {len(trades)} trades")
        return trades

    def execute_trades(self, trades: List[Trade]):
        for trade in trades:
            buyer = self.environment.get_agent(trade.buyer_id)
            seller = self.environment.get_agent(trade.seller_id)

            buyer_surplus = trade.buyer_value - trade.price
            seller_surplus = trade.price - trade.seller_cost

            if buyer_surplus < 0 or seller_surplus < 0:
                logger.warning(f"Trade rejected due to negative surplus: Buyer Surplus = {buyer_surplus}, Seller Surplus = {seller_surplus}")
                continue

            buyer.finalize_trade(trade)
            seller.finalize_trade(trade)
            self.total_surplus_extracted += buyer_surplus + seller_surplus
            self.average_prices.append(trade.price)
            self.successful_trades.append(trade)
            self.trade_history.append(trade)  # Update trade history

            logger.info(f"Executing trade: Buyer {buyer.zi_agent.id} - Surplus: {buyer_surplus:.2f}, Seller {seller.zi_agent.id} - Surplus: {seller_surplus:.2f}")

    def run_auction(self):
        if self.current_round >= self.max_rounds:
            logger.info("Max rounds reached. Auction has ended.")
            return
        
        for round_num in range(self.current_round + 1, self.max_rounds + 1):
            logger.info(f"\n=== Starting Round {round_num} ===")
            self.current_round = round_num

            # Prepare market info
            market_info = self._get_market_info()

            # Generate bids from buyers
            bids = []
            for buyer in self.environment.buyers:
                try:
                    if buyer.zi_agent.allocation.goods < buyer.zi_agent.preference_schedule.values.get(len(buyer.zi_agent.preference_schedule.values), 0):
                        order = buyer.generate_bid(market_info, round_num)
                        if order and order.is_buy:
                            bids.append(order)
                            logger.info(f"{Fore.BLUE}Buyer {Fore.CYAN}{buyer.zi_agent.id}{Fore.BLUE} bid: ${Fore.GREEN}{order.price:.2f}{Fore.BLUE} for {Fore.YELLOW}{order.quantity}{Fore.BLUE} unit(s){Style.RESET_ALL}")
                except Exception as e:
                    logger.error(f"Error generating bid for buyer {buyer.zi_agent.id}: {e}")
                    logger.error(traceback.format_exc())

            # Generate asks from sellers
            asks = []
            for seller in self.environment.sellers:
                try:
                    if seller.zi_agent.allocation.goods > 0:
                        order = seller.generate_bid(market_info, round_num)
                        if order and not order.is_buy:
                            asks.append(order)
                            logger.info(f"{Fore.RED}Seller {Fore.CYAN}{seller.zi_agent.id}{Fore.RED} ask: ${Fore.GREEN}{order.price:.2f}{Fore.RED} for {Fore.YELLOW}{order.quantity}{Fore.RED} unit(s){Style.RESET_ALL}")
                except Exception as e:
                    logger.error(f"Error generating ask for seller {seller.zi_agent.id}: {e}")
                    logger.error(traceback.format_exc())

            logger.info(f"Generated {len(bids)} bids and {len(asks)} asks")

            # Match orders and execute trades
            trades = self.match_orders(bids, asks, round_num)
            if trades:
                self.execute_trades(trades)
            
            logger.info(f"=== Finished Round {round_num} ===\n")

        logger.info("Auction completed. Summarizing results.")
        self.summarize_results()
        
    def create_trade_message(self, trade: Trade, is_buyer: bool) -> ACLMessage:
        content = {
            "trade_id": trade.trade_id,
            "price": trade.price,
            "quantity": trade.quantity,
            "round": trade.round
        }
        
        message = ACLMessage.create_message(
            performative=Performative.INFORM,
            sender="market",
            receiver=str(trade.buyer_id if is_buyer else trade.seller_id),
            content=content,
            protocol="double-auction",
            ontology="market-ontology",
            conversation_id=f"trade-{trade.trade_id}"
        )
        logger.info(f"Created trade message: {message}")
        return message

    def _get_market_info(self) -> dict:
        last_trade_price = self.average_prices[-1] if self.average_prices else None
        average_price = sum(self.average_prices) / len(self.average_prices) if self.average_prices else None
        
        # If no trades have occurred, use the midpoint of buyer and seller base values
        if last_trade_price is None or average_price is None:
            buyer_base_value = max(agent.zi_agent.preference_schedule.get_value(1) for agent in self.environment.buyers)
            seller_base_value = min(agent.zi_agent.preference_schedule.get_value(1) for agent in self.environment.sellers)
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

        # Detecting and explaining potential negative surplus
        if self.total_surplus_extracted < 0:
            logger.warning(f"Warning: Negative practical surplus detected. Possible causes include:")
            logger.warning(f"  1. Mismatch between bid/ask values and agent utilities.")
            logger.warning(f"  2. Overestimated initial utilities.")
            logger.warning(f"  3. High frictions or spread preventing trades.")

    def get_order_book(self):
        return self.order_book

    def get_trade_history(self):
        return self.trade_history

def run_market_simulation(num_buyers: int, num_sellers: int, num_units: int, buyer_base_value: int, seller_base_value: int, spread: float, max_rounds: int):
    # Generate test agents
    agents = generate_llm_market_agents(num_agents=num_buyers + num_sellers, num_units=num_units, buyer_base_value=buyer_base_value, seller_base_value=seller_base_value, spread=spread)
    
    # Create the environment
    env = Environment(agents=agents)

    # Print initial market state
    env.print_market_state()

    # Calculate and print initial utilities
    logger.info("\nInitial Utilities:")
    for agent in env.agents:
        initial_utility = env.get_agent_utility(agent)
        logger.info(f"Agent {agent.zi_agent.id} ({'Buyer' if agent.preference_schedule.is_buyer else 'Seller'}): {initial_utility:.2f}")

    # Run the auction
    auction = DoubleAuction(environment=env, max_rounds=max_rounds)
    auction.run_auction()

    # Analyze the auction results and plot
    analyze_and_plot_auction_results(auction, env)

if __name__ == "__main__":
    # Generate test agents
    num_buyers = 5
    num_sellers = 5
    spread = 0.5

    llm_config= {
        "client": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.5,
        "response_format": {
            "type": "json_object"
        }
    }
    agents = generate_llm_market_agents(
        num_agents=num_buyers + num_sellers, 
        num_units=5, 
        buyer_base_value=100, 
        seller_base_value=80, 
        spread=spread, 
        use_llm=True,
        llm_config=llm_config)
    
    # Create the environment
    env = Environment(agents=agents)

    # Print initial market state
    env.print_market_state()

    # Calculate and print initial utilities
    logger.info("\nInitial Utilities:")
    for agent in env.agents:
        initial_utility = env.get_agent_utility(agent)
        logger.info(f"Agent {agent.zi_agent.id} ({'Buyer' if agent.zi_agent.preference_schedule.is_buyer else 'Seller'}): {initial_utility:.2f}")

    # Run the auction
    auction = DoubleAuction(environment=env, max_rounds=5)
    auction.run_auction()

    # Analyze and plot results
    analyze_and_plot_auction_results(auction, env)