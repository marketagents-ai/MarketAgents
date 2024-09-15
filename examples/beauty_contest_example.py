from market_agents.environments.environment import MultiAgentEnvironment
from market_agents.environments.mechanisms.beauty import (
    BeautyContestMechanism, BeautyContestAction, BeautyContestActionSpace,
    BeautyContestObservationSpace, BeautyContestGlobalObservation
)

def run_beauty_contest_example():
    print("\n=== Beauty Contest Game Example (Batched) ===\n")
    
    # Create a MultiAgentEnvironment with a BeautyContestMechanism
    env = MultiAgentEnvironment(
        name="BeautyContest",
        address="beauty_contest_address",
        max_steps=5,
        mechanism=BeautyContestMechanism(),
        action_space=BeautyContestActionSpace(),
        observation_space=BeautyContestObservationSpace()
    )

    # Run the random action test
    env.random_action_test(num_agents=4, num_steps=5)

def main():
    run_beauty_contest_example()

if __name__ == "__main__":
    main()