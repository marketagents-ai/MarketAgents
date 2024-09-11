from llm_agents.environments.environment import (
    MultiAgentEnvironment, Notebook, BeautyContestMechanism,
    LocalAction, GlobalAction, IntAction, StrAction
)

def run_notebook_example():
    print("\n=== Notebook Environment Example (Sequential) ===\n")
    
    # Create a MultiAgentEnvironment with a Notebook mechanism
    env = MultiAgentEnvironment(
        name="TestNotebook",
        address="test_address",
        max_steps=5,
        mechanism=Notebook(),
        action_space=StrAction(min_length=5, max_length=20)
    )

    # Define some test agents
    agent_ids = ["Alice", "Bob", "Charlie"]

    # Run the environment for max_steps
    for step in range(env.max_steps):
        print(f"\nStep {step + 1}:")
        
        # Generate random actions for each agent
        actions = {}
        for agent_id in agent_ids:
            random_action = env.action_space.sample()
            actions[agent_id] = LocalAction(agent_id=agent_id, action=random_action.content)

        # Create a GlobalAction from the local actions
        global_action = GlobalAction.from_local_actions(actions)

        # Step the environment
        step_result = env.step(global_action)

        # Print the results
        for agent_id in agent_ids:
            local_step = step_result.get_local_step(agent_id)
            print(f"{agent_id} wrote: {actions[agent_id].action}")

        print("\nCurrent Notebook Content:")
        print(env.get_global_state())
        print("\n" + "="*50)

    # Close the environment
    env.close()

def run_beauty_contest_example():
    print("\n=== Beauty Contest Game Example (Batched) ===\n")
    
    # Create a MultiAgentEnvironment with a BeautyContestMechanism
    env = MultiAgentEnvironment(
        name="BeautyContest",
        address="beauty_contest_address",
        max_steps=5,
        mechanism=BeautyContestMechanism(),
        action_space=IntAction(min_value=0, max_value=100)
    )

    # Define some test agents
    agent_ids = ["Alice", "Bob", "Charlie", "David"]

    # Run the environment for max_steps
    for step in range(env.max_steps):
        print(f"\nRound {step + 1}:")
        
        # Generate random actions for each agent
        actions = {}
        for agent_id in agent_ids:
            random_action = env.action_space.sample()
            actions[agent_id] = LocalAction(agent_id=agent_id, action=random_action.value)

        # Create a GlobalAction from the local actions
        global_action = GlobalAction.from_local_actions(actions)

        # Step the environment
        step_result = env.step(global_action)

        # Print the results
        print("Guesses:")
        for agent_id in agent_ids:
            local_step = step_result.get_local_step(agent_id)
            print(f"  {agent_id}: {actions[agent_id].action}")

        target = step_result.info['target']
        average = step_result.info['average']
        print(f"\nAverage guess: {average:.2f}")
        print(f"Target number (2/3 of average): {target:.2f}")

        print("\nRewards:")
        for agent_id in agent_ids:
            local_step = step_result.get_local_step(agent_id)
            reward = local_step.reward.rewards[agent_id]
            print(f"  {agent_id}: {reward:.2f}")

        print("\n" + "="*50)

    # Close the environment
    env.close()

def main():
    run_notebook_example()
    run_beauty_contest_example()

if __name__ == "__main__":
    main()
