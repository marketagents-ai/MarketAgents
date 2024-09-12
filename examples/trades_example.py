from market_agents.economics.econ_agent import EconomicAgent, create_economic_agent
from market_agents.economics.equilibrium import Equilibrium
from market_agents.economics.econ_models import Trade
import logging

# Set up logging
logging.basicConfig(level=logging.WARNING)
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

def calculate_efficiency(agents: list[EconomicAgent], equilibrium: Equilibrium, actual_quantity: int) -> float:
    practical_total_surplus = sum(agent.calculate_individual_surplus() for agent in agents)
    theoretical_results = equilibrium.calculate_equilibrium()["apple"]
    theoretical_total_surplus = theoretical_results["total_surplus"]
    theoretical_quantity = theoretical_results["quantity"]
    
    # Adjust theoretical surplus based on the actual quantity traded
    adjusted_theoretical_surplus = (theoretical_total_surplus / theoretical_quantity) * actual_quantity if theoretical_quantity > 0 else 0
    
    return (practical_total_surplus / adjusted_theoretical_surplus) * 100 if adjusted_theoretical_surplus > 0 else 0

def run_trades_example():
    print("\n=== Simplified Trades Example ===\n")
    
    num_rounds = 200
    agents = create_agents()
    buyer, seller = agents
    
    # Create Equilibrium object for analysis
    equilibrium = Equilibrium(agents=agents, goods=["apple"])
    
    # Calculate initial equilibrium
    eq_results = equilibrium.calculate_equilibrium()["apple"]
    ce_price, ce_quantity = eq_results["price"], int(eq_results["quantity"])
    theoretical_total_surplus = eq_results["total_surplus"]
    
    print(f"Competitive Equilibrium Price: {ce_price:.2f}")
    print(f"Competitive Equilibrium Quantity: {ce_quantity}")
    print(f"Theoretical Total Surplus: {theoretical_total_surplus:.2f}")
    
    # Run trades
    trade_id = 1
    successful_trades = 0
    total_trade_price = 0
    
    for round in range(num_rounds):
        # Generate bid and ask
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
                good_name="apple"
            )
            
            buyer_success = buyer.process_trade(trade)
            seller_success = seller.process_trade(trade)
            
            if buyer_success and seller_success:
                trade_id += 1
                successful_trades += 1
                total_trade_price += trade_price

    # Calculate final efficiency
    final_efficiency = calculate_efficiency(agents, equilibrium, successful_trades)
    print("\nFinal agent states:")
    for agent in agents:
        agent.print_status()

    # Print summary statistics
    print("\nTrade Summary:")
    print(f"Total Rounds: {num_rounds}")
    print(f"Successful Trades: {successful_trades}")
    print(f"Average Trade Price: {total_trade_price / successful_trades:.2f}" if successful_trades else "No trades occurred")
    print(f"\nFinal Efficiency: {final_efficiency:.2f}%")


if __name__ == "__main__":
    run_trades_example()