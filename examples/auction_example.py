# auction_example.py

import logging
from market_agents.environments.mechanisms.auction import DoubleAuction, AuctionAction, GlobalAuctionAction, AuctionGlobalObservation
from market_agents.economics.econ_models import Bid, Ask, Trade

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Initialize the auction mechanism
    auction = DoubleAuction(max_rounds=5)

    # Define test scenarios
    scenarios = [
        {
            'description': 'Simple matching bid and ask',
            'actions': {
                'agent_1': AuctionAction(agent_id='agent_1', action=Bid(price=50, quantity=1)),
                'agent_2': AuctionAction(agent_id='agent_2', action=Ask(price=45, quantity=1))
            }
        },
        {
            'description': 'No matching bid and ask',
            'actions': {
                'agent_3': AuctionAction(agent_id='agent_3', action=Bid(price=40, quantity=1)),
                'agent_4': AuctionAction(agent_id='agent_4', action=Ask(price=60, quantity=1))
            }
        },
        {
            'description': 'Multiple bids and asks',
            'actions': {
                'agent_5': AuctionAction(agent_id='agent_5', action=Bid(price=55, quantity=1)),
                'agent_6': AuctionAction(agent_id='agent_6', action=Ask(price=50, quantity=1)),
                'agent_7': AuctionAction(agent_id='agent_7', action=Bid(price=52, quantity=1)),
                'agent_8': AuctionAction(agent_id='agent_8', action=Ask(price=53, quantity=1))
            }
        },
        {
            'description': 'Unmatched orders carried over',
            'actions': {
                'agent_9': AuctionAction(agent_id='agent_9', action=Bid(price=48, quantity=1)),
                'agent_10': AuctionAction(agent_id='agent_10', action=Ask(price=49, quantity=1))
            }
        }
    ]

    # Run scenarios
    for i, scenario in enumerate(scenarios):
        print(f"\nScenario {i+1}: {scenario['description']}")
        actions = scenario['actions']
        global_action = GlobalAuctionAction(actions=actions)

        # Step the auction
        environment_step = auction.step(global_action)

        # Print the market summary
        assert isinstance(environment_step.global_observation, AuctionGlobalObservation)
        market_summary = environment_step.global_observation.market_summary
        print(f"Market Summary: {market_summary}")

        # Print trades executed
        trades = environment_step.global_observation.all_trades
        if trades:
            print("Trades executed:")
            for trade in trades:
                print(f"  Trade ID: {trade.trade_id}, Buyer: {trade.buyer_id}, Seller: {trade.seller_id}, Price: {trade.price}, Quantity: {trade.quantity}")
        else:
            print("No trades executed in this round.")

        # Print waiting orders
        print("Remaining Waiting Orders:")
        waiting_bids = auction.waiting_bids
        waiting_asks = auction.waiting_asks
        if waiting_bids:
            print("  Bids:")
            for bid in waiting_bids:
                print(f"    Agent ID: {bid.agent_id}, Price: {bid.action.price}, Quantity: {bid.action.quantity}")
        else:
            print("  No waiting bids.")
        if waiting_asks:
            print("  Asks:")
            for ask in waiting_asks:
                print(f"    Agent ID: {ask.agent_id}, Price: {ask.action.price}, Quantity: {ask.action.quantity}")
        else:
            print("  No waiting asks.")

    # End of scenarios
    print("\nFinal Auction State:")
    print(f"Total Trades Executed: {len(auction.trades)}")
    for trade in auction.trades:
        print(f"  Trade ID: {trade.trade_id}, Buyer: {trade.buyer_id}, Seller: {trade.seller_id}, Price: {trade.price}, Quantity: {trade.quantity}")

if __name__ == "__main__":
    main()
