from market_agents.economics.equilibrium import Equilibrium
from market_agents.economics.econ_agent import EconomicAgent, create_economic_agent
from market_agents.economics.analysis import analyze_and_plot_market_results
from market_agents.economics.econ_models import Trade
import random
from typing import Tuple, List
from market_agents.economics.econ_models import Bid, Ask

if __name__ == "__main__":
   # Set random seed for reproducibility
    random.seed(42)
    
    # Create multiple buyers and sellers
    num_buyers = 10
    num_sellers = 10
    num_units_per_agent = 10
    goods = ["apple"]
    
    buyers = [
        create_economic_agent(
            agent_id=f"buyer_{i}",
            goods=goods,
            buy_goods=goods,
            sell_goods=[],
            base_values={"apple": 100},
            initial_cash=500,
            initial_goods={"apple": 0},
            num_units=num_units_per_agent,
            noise_factor=0.1,
            max_relative_spread=0.2
        ) for i in range(num_buyers)
    ]
    
    sellers = [
        create_economic_agent(
            agent_id=f"seller_{i}",
            goods=goods,
            buy_goods=[],
            sell_goods=goods,
            base_values={"apple": 80},
            initial_cash=0,
            initial_goods={"apple": num_units_per_agent},
            num_units=num_units_per_agent,
            noise_factor=0.1,
            max_relative_spread=0.2
        ) for i in range(num_sellers)
    ]
    
    # Create the Equilibrium object
    equilibrium = Equilibrium(agents=buyers + sellers, goods=goods)
    
    # Calculate and print the theoretical equilibrium
    result = equilibrium.calculate_equilibrium()
    print("Theoretical Equilibrium Results:")
    for good, data in result.items():
        print(f"\nGood: {good}")
        for key, value in data.items():
            print(f"  {key}: {value}")
    theoretical_total_surplus = sum(data['total_surplus'] for data in result.values())
    print(f"\nTheoretical Total Surplus: {theoretical_total_surplus:.2f}")
    
    # Plot the supply and demand curves
    equilibrium.plot_supply_demand("apple")
    
    # Simulate the market trading process
    print("\nSimulating market trading...")
    all_agents = buyers + sellers
    trades = []
    trade_id = 1
    max_rounds = 100  # Number of trading rounds
    # Initialize variables to track cumulative surplus
    cumulative_quantities = []
    cumulative_surplus = []
    total_quantity = 0
    total_surplus = 0.0

    for round_num in range(max_rounds):
        # Collect bids and asks from agents
        bids : List[Tuple[EconomicAgent, Bid]] = []
        asks :List[Tuple[EconomicAgent, Ask]] = []
        for agent in all_agents:
            for good in goods:
                bid = agent.generate_bid(good)
                if bid:
                    bids.append((agent, bid))
                ask = agent.generate_ask(good)
                if ask:
                    asks.append((agent, ask))
        
        # Sort bids and asks by price
        bids.sort(key=lambda x: x[1].price, reverse=True)  # Highest bids first
        asks.sort(key=lambda x: x[1].price)  # Lowest asks first
        # Attempt to match bids and asks
        
        while bids and asks:
            highest_bidder, highest_bid = bids[0]
            lowest_asker, lowest_ask = asks[0]
            if highest_bid.price >= lowest_ask.price:
                # Compute utilities before the trade for surplus calculation
                buyer_utility_before = highest_bidder.calculate_utility(highest_bidder.endowment.current_basket)
                seller_utility_before = lowest_asker.calculate_utility(lowest_asker.endowment.current_basket)
                
                # Execute trade
                trade_price = (highest_bid.price + lowest_ask.price) / 2
                trade = Trade(
                    trade_id=trade_id,
                    buyer_id=highest_bidder.id,
                    seller_id=lowest_asker.id,
                    price=trade_price,
                    quantity=1,
                    good_name=good,
                    ask_price=lowest_ask.price,
                    bid_price=highest_bid.price
                )
                # Process trade for both buyer and seller
                buyer_success = highest_bidder.would_accept_trade(trade)
                seller_success = lowest_asker.would_accept_trade(trade)
                
                if buyer_success and seller_success:
                    highest_bidder.process_trade(trade)
                    lowest_asker.process_trade(trade)
                    # Compute utilities after the trade
                    buyer_utility_after = highest_bidder.calculate_utility(highest_bidder.endowment.current_basket)
                    seller_utility_after = lowest_asker.calculate_utility(lowest_asker.endowment.current_basket)
                    
                    # Compute surplus changes
                    buyer_surplus_change = buyer_utility_after - buyer_utility_before
                    seller_surplus_change = seller_utility_after - seller_utility_before
                    trade_surplus = buyer_surplus_change + seller_surplus_change
                    total_surplus += trade_surplus
                    total_quantity += trade.quantity
                    cumulative_surplus.append(total_surplus)
                    cumulative_quantities.append(total_quantity)
                    
                    trades.append(trade)
                    trade_id += 1
                    print(f"Round {round_num + 1}: Trade executed at price {trade_price:.2f}")
                    # Remove the bid and ask since they have been fulfilled
                    bids.pop(0)
                    asks.pop(0)
                else:
                    print(f"buyer_success: {buyer_success}, seller_success: {seller_success}")
                    # If trade was not successful, remove the bid/ask and continue
                    bids.pop(0)
                    asks.pop(0)
            else:
                # No more matches possible in this round
                
                break
        else:
            print(f"Round {round_num + 1}: No trade executed")
    
    # After trading rounds, compute the empirical surplus
    print("\nComputing empirical surplus...")
    total_buyer_surplus = sum(agent.calculate_individual_surplus() for agent in buyers)
    total_seller_surplus = sum(agent.calculate_individual_surplus() for agent in sellers)
    total_empirical_surplus = total_buyer_surplus + total_seller_surplus
    print(f"Total Empirical Buyer Surplus: {total_buyer_surplus:.2f}")
    print(f"Total Empirical Seller Surplus: {total_seller_surplus:.2f}")
    print(f"Total Empirical Surplus: {total_empirical_surplus:.2f}")

    # Compute and print the empirical efficiency (% of theoretical surplus achieved)
    efficiency = (total_empirical_surplus / theoretical_total_surplus) * 100 if theoretical_total_surplus > 0 else 0
    print(f"\nEmpirical Efficiency: {efficiency:.2f}%")

    print("\nGenerating market report...")
    analyze_and_plot_market_results(
        trades=trades,
        agents=all_agents,
        equilibrium=equilibrium,
        goods=goods,
        max_rounds=max_rounds,
        cumulative_quantities=cumulative_quantities,
        cumulative_surplus=cumulative_surplus
    )
    print(trade_id)




