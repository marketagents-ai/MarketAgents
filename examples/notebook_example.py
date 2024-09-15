from market_agents.environments.environment import (
    MultiAgentEnvironment, LocalAction, GlobalAction, StrAction, IntAction, Notebook, ActionSpace, ObservationSpace,
    StrObservation, GlobalObservation, LocalObservation, NotebookActionSpace, NotebookObservationSpace
)
from typing import List, Type
import random
import string


def run_notebook_example():
    env = MultiAgentEnvironment(
        name="TestNotebook",
        address="test_address",
        max_steps=5,
        mechanism=Notebook(),
        action_space=NotebookActionSpace(),
        observation_space=NotebookObservationSpace()
    )

    env.random_action_test(num_agents=3, num_steps=5)

if __name__ == "__main__":
    run_notebook_example()