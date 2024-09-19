from market_agents.environments.environment import MultiAgentEnvironment
from market_agents.environments.mechanisms.auction import (
    DoubleAuction, AuctionAction, AuctionActionSpace, AuctionObservationSpace
)

def run_random_auction():
    print("\n=== Random Double Auction Example ===\n")
    
    # Create a MultiAgentEnvironment with a DoubleAuction mechanism
    env = MultiAgentEnvironment(
        name="RandomAuction",
        address="random_auction_address",
        max_steps=10,
        mechanism=DoubleAuction(max_rounds=10),
        action_space=AuctionActionSpace(),
        observation_space=AuctionObservationSpace()
    )

    # Run the random action test
    env.random_action_test(num_agents=4, num_steps=10)

def main():
    run_random_auction()

if __name__ == "__main__":
    main()
