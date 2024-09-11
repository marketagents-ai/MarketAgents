from typing import Dict, Any, List, Optional, Type, Union, Tuple
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from datetime import datetime
import random
import string
from statistics import mean

class LocalAction(BaseModel):
    """Represents an action for a single agent."""
    agent_id: str
    action: Any

class GlobalAction(BaseModel):
    """Represents actions for all agents."""
    actions: Dict[str, LocalAction]

    def locals(self) -> Dict[str, LocalAction]:
        """Get the local actions for all agents."""
        return self.actions

    @classmethod
    def from_local_actions(cls, local_actions: Dict[str, LocalAction]) -> "GlobalAction":
        """Create a global action from local actions."""
        return cls(actions=local_actions)

class LocalObservation(BaseModel):
    """Represents an observation for a single agent."""
    agent_id: str
    observation: Any

class GlobalObservation(BaseModel):
    """Represents observations for all agents."""
    observations: Dict[str, LocalObservation]
    global_obs: Optional[Any] = None

    def locals(self) -> Dict[str, LocalObservation]:
        """Get the local observations for all agents."""
        return self.observations

    @classmethod
    def from_local_observations(cls, local_observations: Dict[str, LocalObservation]) -> "GlobalObservation":
        """Create a global observation from local observations."""
        return cls(observations=local_observations)

    def to_local(self, agent_id: str) -> LocalObservation:
        """Convert global observation to local observation for a specific agent."""
        return self.observations[agent_id]

class LocalReward(BaseModel):
    """Represents rewards for all agents."""
    rewards: Dict[str, float]

class GlobalReward(BaseModel):
    """Represents rewards for all agents."""
    rewards: Dict[str, LocalReward]

    def locals(self) -> Dict[str, LocalReward]:
        """Get the local rewards for all agents."""
        return self.rewards

    @classmethod
    def from_local_rewards(cls, local_rewards: Dict[str, LocalReward]) -> "GlobalReward":
        """Create a global reward from local rewards."""
        return cls(rewards=local_rewards)

class LocalEnvironmentStep(BaseModel):
    """Represents the output of a single environment step for a single agent."""
    observation: LocalObservation
    reward: LocalReward
    done: bool
    info: Dict[str, Any]

class EnvironmentStep(BaseModel):
    """Represents the output of a single environment step."""
    global_observation: GlobalObservation
    reward: GlobalReward
    done: bool
    info: Dict[str, Any]

    @classmethod
    def from_local_steps(cls, local_steps: Dict[str, LocalEnvironmentStep]) -> "EnvironmentStep":
        """Create a global environment step from local steps."""
        observations = {agent_id: step.observation for agent_id, step in local_steps.items()}
        rewards = {agent_id: step.reward for agent_id, step in local_steps.items()}
        done = all(step.done for step in local_steps.values())
        info = {}
        return cls(
            global_observation=GlobalObservation.from_local_observations(observations),
            reward=GlobalReward.from_local_rewards(rewards),
            done=done,
            info=info
        )
        

    def get_local_step(self, agent_id: str) -> LocalEnvironmentStep:
        """Get the local step for a single agent."""
        return LocalEnvironmentStep(
            observation=self.global_observation.to_local(agent_id),
            reward=self.reward.rewards[agent_id],
            done=self.done,
            info=self.info
        )

class EnvironmentHistory(BaseModel):
    """Represents the history of environment steps."""
    steps: List[Tuple[GlobalAction, EnvironmentStep]] = Field(default_factory=list)

    def add_step(self, action: GlobalAction, step: EnvironmentStep):
        """Add a step to the history."""
        self.steps.append((action, step))

class ActionSpace(BaseModel, ABC):
    @abstractmethod
    def sample(self) -> Any:
        """Sample a random action from the action space."""
        pass

class ObservationSpace(BaseModel, ABC):
    @abstractmethod
    def sample(self) -> Any:
        """Sample a random observation from the observation space."""
        pass

class StrAction(ActionSpace):
    min_length: int = Field(default=1, description="Minimum length of the string action")
    max_length: int = Field(default=10, description="Maximum length of the string action")
    content: str = Field(default="", description="Content of the string action")

    def sample(self) -> 'StrAction':
        """Sample a random string action."""
        length = random.randint(self.min_length, self.max_length)
        return StrAction(content=''.join(random.choices(string.ascii_letters + string.digits, k=length)))

class StrObservation(ObservationSpace):
    min_length: int = Field(default=1, description="Minimum length of the string observation")
    max_length: int = Field(default=100, description="Maximum length of the string observation")
    content: str = Field(default="", description="Content of the string observation")

    def sample(self) -> 'StrObservation':
        """Sample a random string observation."""
        length = random.randint(self.min_length, self.max_length)
        return StrObservation(content=''.join(random.choices(string.ascii_letters + string.digits + string.punctuation + ' ', k=length)))

class IntAction(ActionSpace):
    min_value: int = Field(default=0, description="Minimum value of the integer action")
    max_value: int = Field(default=100, description="Maximum value of the integer action")
    value: int = Field(default=0, description="Value of the integer action")

    def sample(self) -> 'IntAction':
        """Sample a random integer action."""
        return IntAction(value=random.randint(self.min_value, self.max_value))

class Mechanism(BaseModel, ABC):
    @abstractmethod
    def step(self, action: Union[LocalAction, GlobalAction]) -> Union[LocalEnvironmentStep, EnvironmentStep]:
        """Execute a step in the mechanism."""
        pass

    @property
    @abstractmethod
    def sequential(self) -> bool:
        """Whether the mechanism is sequential or not."""
        pass
    @abstractmethod
    def get_global_state(self) -> Any:
        """Get the global state of the mechanism."""
        pass

class Notebook(Mechanism):
    text: str = Field(default="", description="The notebook's text content")
    
    @property
    def sequential(self) -> bool:
        return True

    def step(self, action: LocalAction) -> LocalEnvironmentStep:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        header = f"\n[{timestamp}] Agent {action.agent_id}:"
        self.text += f"{header}\n{action.action}\n"
        
        observation = LocalObservation(agent_id=action.agent_id, observation=self.text)
        reward = LocalReward(rewards={action.agent_id: 0.0})  # No reward system for this simple mechanism
        done = False  # The notebook never ends
        info = {}

        return LocalEnvironmentStep(
            observation=observation,
            reward=reward,
            done=done,
            info=info
        )
    def get_global_state(self) -> str:
        return self.text

class BeautyContestMechanism(Mechanism):
    target_factor: float = Field(default=2/3, description="The factor to multiply the average by")
    last_actions: Dict[str, int] = Field(default_factory=dict, description="The last actions taken by each agent")
    last_target: float = Field(default=0, description="The last target number")

    @property
    def sequential(self) -> bool:
        return False

    def step(self, action: GlobalAction) -> EnvironmentStep:
        # Extract the integer values from the actions
        action_values = {agent_id: int(local_action.action) for agent_id, local_action in action.locals().items()}
        
        # Calculate the target number
        average = mean(action_values.values())
        self.last_target = self.target_factor * average

        # Calculate rewards (negative distance from target)
        rewards = {agent_id: -abs(value - self.last_target) for agent_id, value in action_values.items()}

        # Update last actions
        self.last_actions = action_values

        # Prepare observations (last actions of all agents)
        observations = {agent_id: LocalObservation(agent_id=agent_id, observation=self.last_actions) 
                        for agent_id in action_values.keys()}

        # Prepare rewards
        local_rewards = {agent_id: LocalReward(rewards={agent_id: reward}) for agent_id, reward in rewards.items()}

        return EnvironmentStep(
            global_observation=GlobalObservation(observations=observations, global_obs=self.last_actions),
            reward=GlobalReward(rewards=local_rewards),
            done=False,  # The beauty contest can continue indefinitely
            info={"target": self.last_target, "average": average}
        )

    def get_global_state(self) -> Dict[str, Any]:
        return {
            "last_actions": self.last_actions,
            "last_target": self.last_target
        }

class MultiAgentEnvironment(BaseModel):
    """
    Base class for multi-agent environments. With batched or sequential actions.
    """
    name: str = Field(..., description="Name of the environment")
    address: str = Field(..., description="Address of the environment for orchestrator linking")
    current_step: int = Field(default=0, description="Current step/round of the simulation")
    max_steps: int = Field(..., description="Maximum number of steps/rounds for this environment")
    action_space: ActionSpace = Field(default_factory=StrAction, description="Action space of the environment")
    observation_space: ObservationSpace = Field(default_factory=StrObservation, description="Observation space of the environment")
    history: EnvironmentHistory = Field(default_factory=EnvironmentHistory, description="History of environment steps")
    mechanism: Mechanism = Field(default_factory=Notebook, description="Mechanism of the environment that determines the rules of the game P(s, a, s')")

    def step(self, actions: GlobalAction) -> EnvironmentStep:
        """
        Run one timestep of the environment's dynamics using the batched agent actions.
        
        Args:
            actions (GlobalAction): A batched action containing actions for each agent.

        Returns:
            EnvironmentStep: The result of taking a step in the environment.
        """
        if self.mechanism.sequential:
            # if it is sequential, we need to run the mechanism for each agent
            local_steps: Dict[str, LocalEnvironmentStep] = {}  # Correct type annotation
            for agent_id, local_action in actions.locals().items():
                local_step = self.mechanism.step(local_action)
                assert isinstance(local_step, LocalEnvironmentStep)
                local_steps[agent_id] = local_step
            global_step = EnvironmentStep.from_local_steps(local_steps)
        else:
            global_step = self.mechanism.step(actions)
            assert isinstance(global_step, EnvironmentStep)
        self.current_step += 1
        self.update_history(actions, global_step)
        return global_step

    def reset(self) -> GlobalObservation:
        """
        Reset the environment and return the initial global observation.

        Returns:
            GlobalObservation: Initial global observation of the environment.
        """
        self.current_step = 0
        self.global_state = {}
        self.history = EnvironmentHistory()
        if isinstance(self.mechanism, Notebook):
            self.mechanism.text = ""
        return GlobalObservation(observations={}, global_obs={})

    def render(self):
        """
        Render the environment.

        Args:
            mode (str): The mode to render with.

        """
        print(self.get_global_state())

    def close(self):
        """
        Close the environment, do any necessary cleanup.
        """
        pass  # No specific cleanup needed for the basic environment

    def get_global_state(self) -> Any:
        """
        Return a summary of the global state.

        Returns:
            Dict[str, Any]: The global state.
        """
        return self.mechanism.get_global_state()
    def get_current_step(self) -> int:
        """
        Return the current step/round of the simulation.

        Returns:
            int: The current step.
        """
        return self.current_step

    def update_history(self, action: GlobalAction, step: EnvironmentStep):
        """
        Update the environment history with the latest step.
        """
        self.history.add_step(action, step)

def main():
    # Create a MultiAgentEnvironment with a BeautyContestMechanism
    env = MultiAgentEnvironment(
        name="BeautyContest",
        address="beauty_contest_address",
        max_steps=5,
        mechanism=BeautyContestMechanism(),
        action_space=IntAction(min_value=0, max_value=100)
    )

    # Define some test agents
    agent_ids = ["Agent1", "Agent2", "Agent3", "Agent4"]

    # Run the environment for max_steps
    for step in range(env.max_steps):
        print(f"\nStep {step + 1}:")

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
        for agent_id in agent_ids:
            local_step = step_result.get_local_step(agent_id)
            print(f"{agent_id} action: {actions[agent_id].action}")
            print(f"{agent_id} observation: {local_step.observation.observation}")
            print(f"{agent_id} reward: {local_step.reward.rewards[agent_id]}")
            print()

        print("Global state:")
        print(env.get_global_state())
        print(f"Target: {step_result.info['target']:.2f}")
        print(f"Average: {step_result.info['average']:.2f}")
        print("\n" + "="*50)

    # Close the environment
    env.close()

if __name__ == "__main__":
    main()

