# equilibrium_da_example.py

import random
import logging
from typing import List, Dict, Tuple
from market_agents.economics.econ_agent import create_economic_agent, EconomicAgent
from market_agents.economics.equilibrium import Equilibrium
from market_agents.environments.mechanisms.auction import DoubleAuction, AuctionAction, GlobalAuctionAction, AuctionGlobalObservation
from market_agents.economics.econ_models import Bid, Ask, Trade
from market_agents.economics.analysis import analyze_and_plot_market_results

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_agents(num_buyers: int, num_sellers: int, num_units: int) -> List[EconomicAgent]:
    buyers = [
        create_economic_agent(
            agent_id=f"buyer_{i}",
            goods=["apple"],
            buy_goods=["apple"],
            sell_goods=[],
            base_values={"apple": 100},
            initial_cash=1000,
            initial_goods={"apple": 0},
            num_units=num_units,
            noise_factor=0.1,
            max_relative_spread=0.2
        ) for i in range(num_buyers)
    ]
    
    sellers = [
        create_economic_agent(
            agent_id=f"seller_{i}",
            goods=["apple"],
            buy_goods=[],
            sell_goods=["apple"],
            base_values={"apple": 80},
            initial_cash=0,
            initial_goods={"apple": num_units},
            num_units=num_units,
            noise_factor=0.1,
            max_relative_spread=0.2
        ) for i in range(num_sellers)
    ]
    
    return buyers + sellers

def run_auction(agents: List[EconomicAgent], max_rounds: int) -> Tuple[List[Trade], List[float]]:
    auction = DoubleAuction(max_rounds=max_rounds)
    all_trades = []
    surplus = []

    for round_num in range(max_rounds):
        actions = {}
        for agent in agents:
            bid = agent.generate_bid("apple")
            ask = agent.generate_ask("apple")
            if bid:
                actions[agent.id] = AuctionAction(agent_id=agent.id, action=bid)
            elif ask:
                actions[agent.id] = AuctionAction(agent_id=agent.id, action=ask)

        if not actions:
            logger.info(f"No more actions generated. Ending auction at round {round_num}")
            break

        global_action = GlobalAuctionAction(actions=actions)
        environment_step = auction.step(global_action)
        
        assert isinstance(environment_step.global_observation, AuctionGlobalObservation)
        new_trades = environment_step.global_observation.all_trades
        
        # Sort trades by trade_id to ensure chronological order
        new_trades.sort(key=lambda x: x.trade_id)
        
        for trade in new_trades:
            buyer = next(agent for agent in agents if agent.id == trade.buyer_id)
            seller = next(agent for agent in agents if agent.id == trade.seller_id)

            if buyer is None or seller is None:
                raise ValueError(f"Trade {trade} has invalid agent IDs")
            
            if buyer.would_accept_trade(trade) and seller.would_accept_trade(trade):
                buyer_utility_before = buyer.calculate_utility(buyer.endowment.current_basket)
                seller_utility_before = seller.calculate_utility(seller.endowment.current_basket)
                buyer.process_trade(trade)
                seller.process_trade(trade)
                buyer_utility_after = buyer.calculate_utility(buyer.endowment.current_basket)
                seller_utility_after = seller.calculate_utility(seller.endowment.current_basket)
                trade_surplus = buyer_utility_after - buyer_utility_before + seller_utility_after - seller_utility_before
                surplus.append(trade_surplus)
                all_trades.append(trade)
                logger.info(f"Trade executed: {trade}")
            else:
                logger.warning(f"Trade rejected: {trade}")

    return all_trades, surplus

def calculate_surplus(agents: List[EconomicAgent]) -> Dict[str, float]:
    buyer_surplus = sum(agent.calculate_individual_surplus() for agent in agents if agent.is_buyer("apple"))
    seller_surplus = sum(agent.calculate_individual_surplus() for agent in agents if agent.is_seller("apple"))
    total_surplus = buyer_surplus + seller_surplus
    return {
        "buyer_surplus": buyer_surplus,
        "seller_surplus": seller_surplus,
        "total_surplus": total_surplus
    }

def main():
    random.seed(42)
    
    num_buyers = 10
    num_sellers = 10
    num_units_per_agent = 10
    max_rounds = 100
    
    agents = create_agents(num_buyers, num_sellers, num_units_per_agent)
    
    # Calculate theoretical equilibrium
    equilibrium = Equilibrium(agents=agents, goods=["apple"])
    theoretical_results = equilibrium.calculate_equilibrium()
    logger.info("Theoretical Equilibrium Results:")
    logger.info(theoretical_results)
    theoretical_total_surplus = theoretical_results["apple"]["total_surplus"]
    
    # Run auction
    trades,surplus = run_auction(agents, max_rounds)
    
    # Calculate empirical results
    empirical_surplus = calculate_surplus(agents)
    efficiency = (empirical_surplus["total_surplus"] / theoretical_total_surplus) * 100 if theoretical_total_surplus > 0 else 0
    
    logger.info("\nEmpirical Results:")
    logger.info(f"Total Trades Executed: {len(trades)}")
    logger.info(f"Buyer Surplus: {empirical_surplus['buyer_surplus']:.2f}")
    logger.info(f"Seller Surplus: {empirical_surplus['seller_surplus']:.2f}")
    logger.info(f"Total Surplus: {empirical_surplus['total_surplus']:.2f}")
    logger.info(f"Efficiency: {efficiency:.2f}%")
    #compute comulative surplus from list of trades
    cumulative_surplus = [sum(surplus[:i+1]) for i in range(len(surplus))]
    # Generate market report
    analyze_and_plot_market_results(
        trades=trades,
        agents=agents,
        equilibrium=equilibrium,
        goods=["apple"],
        max_rounds=max_rounds,
        cumulative_quantities=[sum(trade.quantity for trade in trades[:i+1]) for i in range(len(trades))],
        cumulative_surplus=cumulative_surplus
    )

if __name__ == "__main__":
    main()
