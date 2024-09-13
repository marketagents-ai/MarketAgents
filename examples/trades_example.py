from market_agents.economics.econ_agent import EconomicAgent, create_economic_agent
from market_agents.economics.equilibrium import Equilibrium
from market_agents.economics.econ_models import Trade
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_agents() -> list[EconomicAgent]:
    buyer = create_economic_agent(
        agent_id="buyer",
        goods=["apple"],
        buy_goods=["apple"],
        sell_goods=[],
        base_values={"apple": 100},
        initial_cash=1000,
        initial_goods={"apple": 0},
        num_units=10,
        noise_factor=0.1,
        max_relative_spread=0.2
    )
    seller = create_economic_agent(
        agent_id="seller",
        goods=["apple"],
        buy_goods=[],
        sell_goods=["apple"],
        base_values={"apple": 80},
        initial_cash=0,
        initial_goods={"apple": 10},
        num_units=10,
        noise_factor=0.1,
        max_relative_spread=0.2
    )
    return [buyer, seller]

def calculate_efficiency(agents: list[EconomicAgent], equilibrium: Equilibrium) -> float:
    practical_total_surplus = sum(agent.calculate_individual_surplus() for agent in agents)
    theoretical_results = equilibrium.calculate_equilibrium()["apple"]
    theoretical_total_surplus = theoretical_results["total_surplus"]
    
    return (practical_total_surplus / theoretical_total_surplus) * 100 if theoretical_total_surplus > 0 else 0

def run_trades_example():
    print("\n=== Simplified Trades Example ===\n")
    
    num_rounds = 15
    agents = create_agents()
    buyer, seller = agents
    
    # Create Equilibrium object for analysis
    equilibrium = Equilibrium(agents=agents, goods=["apple"])
    
    print("Theoretical Equilibrium Results:")
    result = equilibrium.calculate_equilibrium()
    for good, data in result.items():
        print(f"\nGood: {good}")
        for key, value in data.items():
            print(f"  {key}: {value}")
    theoretical_total_surplus = sum(data['total_surplus'] for data in result.values())
    print(f"\nTheoretical Total Surplus: {theoretical_total_surplus:.2f}")
    
    # Print initial utilities
    print(f"Initial utility of buyer: {buyer.initial_utility:.2f}")
    print(f"Initial utility of seller: {seller.initial_utility:.2f}")
    
    # Run trades
    trade_id = 1
    successful_trades = 0
    total_trade_price = 0
    
    for round in range(num_rounds):
        # Generate bid and ask
        buyer.reset_pending_orders("apple")
        seller.reset_pending_orders("apple")
        bid = buyer.generate_bid("apple")
        ask = seller.generate_ask("apple")
        
        if bid and ask and bid.price >= ask.price:
            trade_price = (bid.price + ask.price) / 2
            trade = Trade(
                trade_id=trade_id,
                buyer_id=buyer.id,
                seller_id=seller.id,
                price=trade_price,
                quantity=1,
                good_name="apple",
                bid_price=bid.price,
                ask_price=ask.price
            )
            
            buyer_utility_before = buyer.calculate_utility(buyer.endowment.current_basket)
            seller_utility_before = seller.calculate_utility(seller.endowment.current_basket)
            
            buyer_accepts = buyer.would_accept_trade(trade)
            seller_accepts = seller.would_accept_trade(trade)
            
            if buyer_accepts and seller_accepts:
                buyer.process_trade(trade)
                seller.process_trade(trade)
                
                buyer_utility_after = buyer.calculate_utility(buyer.endowment.current_basket)
                seller_utility_after = seller.calculate_utility(seller.endowment.current_basket)
                
                buyer_surplus = buyer_utility_after - buyer_utility_before
                seller_surplus = seller_utility_after - seller_utility_before
                
                print(f"Round {round + 1}: Trade executed at price {trade_price:.2f}")
                print(f"  Buyer surplus: {buyer_surplus:.2f}")
                print(f"  Seller surplus: {seller_surplus:.2f}")
                
                trade_id += 1
                successful_trades += 1
                total_trade_price += trade_price
        else:
            print(f"Round {round + 1}: No trade executed")

    # Calculate final efficiency
    final_efficiency = calculate_efficiency(agents, equilibrium)
    print("\nFinal agent states:")
    for agent in agents:
        agent.print_status()
        surplus = agent.calculate_individual_surplus()
        print(f"Agent {agent.id} surplus: {surplus:.2f}")

    # Print summary statistics
    print("\nTrade Summary:")
    print(f"Total Rounds: {num_rounds}")
    print(f"Successful Trades: {successful_trades}")
    print(f"Average Trade Price: {total_trade_price / successful_trades:.2f}" if successful_trades else "No trades occurred")
    print(f"\nFinal Efficiency: {final_efficiency:.2f}%")

if __name__ == "__main__":
    run_trades_example()
