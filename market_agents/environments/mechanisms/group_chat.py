from typing import List, Dict, Any, Union, Type, Optional
from pydantic import BaseModel, Field, PrivateAttr
from datetime import datetime
import logging

from market_agents.environments.environment import (
    Mechanism, LocalAction, GlobalAction, LocalObservation, GlobalObservation,
    EnvironmentStep, ActionSpace, ObservationSpace, LocalEnvironmentStep
)

logger = logging.getLogger(__name__)

from typing import List, Dict, Any, Union, Type, Optional
from pydantic import BaseModel, Field
import logging

from market_agents.environments.environment import (
    Mechanism, LocalAction, GlobalAction, LocalObservation, GlobalObservation,
    EnvironmentStep, ActionSpace, ObservationSpace, LocalEnvironmentStep
)

logger = logging.getLogger(__name__)

class GroupChatMessage(BaseModel):
    """Message generated by an agent in the group chat"""
    content: str = Field(
        description="Agent's message content"
    )
    message_type: Optional[str] = Field(
        default=None,
        description="Type of message (e.g., propose_topic, reflection)"
    )

class GroupChatAction(LocalAction):
    action: GroupChatMessage = Field(
        description="The message action taken by the agent"
    )

    @classmethod
    def sample(cls, agent_id: str) -> 'GroupChatAction':
        return cls(
            agent_id=agent_id,
            action=GroupChatMessage(content="Sample message")
        )

    @classmethod
    def action_schema(cls) -> Dict[str, Any]:
        return GroupChatMessage.model_json_schema()

class GroupChatGlobalAction(GlobalAction):
    actions: Dict[str, GroupChatAction] = Field(
        description="Dictionary of actions by agent"
    )

class GroupChatObservation(BaseModel):
    current_topic: str = Field(
        default="",
        description="Current topic of discussion"
    )
    agent_message: Optional[GroupChatMessage] = Field(
        default=None,
        description="Agent's own message from this round"
    )

class GroupChatLocalObservation(LocalObservation):
    observation: GroupChatObservation = Field(
        description="Local observation for a specific agent"
    )

class GroupChatGlobalObservation(GlobalObservation):
    observations: Dict[str, GroupChatLocalObservation] = Field(
        description="Dictionary of local observations by agent"
    )
    round_messages: List[GroupChatMessage] = Field(
        default_factory=list,
        description="Messages from the current round"
    )
    current_topic: str = Field(
        default="",
        description="Current topic of discussion"
    )

class GroupChatActionSpace(ActionSpace):
    allowed_actions: List[Type[LocalAction]] = Field(
        default=[GroupChatAction],
        description="Allowed actions in the group chat"
    )

class GroupChatObservationSpace(ObservationSpace):
    allowed_observations: List[Type[LocalObservation]] = Field(
        default=[GroupChatLocalObservation],
        description="Allowed observations in the group chat"
    )

class GroupChat(Mechanism):
    max_rounds: int = Field(
        ...,
        description="Maximum number of chat rounds"
    )
    current_round: int = Field(
        default=0,
        description="Current round number"
    )
    current_topic: str = Field(
        default="",
        description="Current topic of discussion"
    )
    round_messages: List[GroupChatMessage] = Field(
        default_factory=list,
        description="Messages from current round only"
    )
    topics: Dict[int, str] = Field(
        default_factory=dict,
        description="Topics by round number"
    )
    sequential: bool = Field(
        default=False,
        description="Whether agents speak in sequential order"
    )
    speaker_order: Optional[List[str]] = Field(
        default=None,
        description="Order of speakers (if sequential)"
    )
    current_speaker_index: int = Field(
        default=0,
        description="Current speaker index for sequential mode"
    )

    _last_step: Optional[Union[LocalEnvironmentStep, EnvironmentStep]] = PrivateAttr(default=None)

    def step(
        self,
        action: Union[GroupChatAction, GroupChatGlobalAction]
    ) -> Union[LocalEnvironmentStep, EnvironmentStep]:
        """Execute a step in the group chat"""
        if self.sequential:
            if not isinstance(action, GroupChatAction):
                raise TypeError("Sequential mode requires GroupChatAction")
            
            # Verify speaker order
            if self.speaker_order:
                cur_speaker = self.speaker_order[self.current_speaker_index]
                if action.agent_id != cur_speaker:
                    raise ValueError(f"Not {action.agent_id}'s turn to speak")

            # Create local observation
            local_obs = GroupChatLocalObservation(
                agent_id=action.agent_id,
                observation=GroupChatObservation(
                    current_topic=self.current_topic,
                    agent_message=action.action
                )
            )
            
            # Update messages
            self.round_messages.append(action.action)
            
            # Update speaker index
            if self.speaker_order:
                self.current_speaker_index = (self.current_speaker_index + 1) % len(self.speaker_order)
                if self.current_speaker_index == 0:
                    self.current_round += 1

            # Create step result
            step_result = LocalEnvironmentStep(
                observation=local_obs,
                done=(self.current_round >= self.max_rounds),
                info={
                    "current_round": self.current_round,
                    "current_topic": self.current_topic,
                    "current_speaker": cur_speaker if self.speaker_order else None
                }
            )
            self._last_step = step_result
            return step_result

        else:
            if not isinstance(action, GroupChatGlobalAction):
                raise TypeError("Non-sequential mode requires GroupChatGlobalAction")

            # Process all messages
            new_messages = []
            for agent_id, local_action in action.actions.items():
                if isinstance(local_action, dict):
                    local_action = GroupChatAction.model_validate(local_action)
                new_messages.append(local_action.action)
                
                # Check for topic proposals
                if local_action.action.message_type == "propose_topic":
                    self._update_topic(local_action.action.content)

            # Update round messages
            self.round_messages = new_messages

            # Create observations
            observations = {}
            for agent_id, local_action in action.actions.items():
                observations[agent_id] = GroupChatLocalObservation(
                    agent_id=agent_id,
                    observation=GroupChatObservation(
                        current_topic=self.current_topic,
                        agent_message=local_action.action
                    )
                )

            # Increment round in non-sequential mode
            self.current_round += 1

            # Create step result
            step_result = EnvironmentStep(
                global_observation=GroupChatGlobalObservation(
                    observations=observations,
                    round_messages=self.round_messages,
                    current_topic=self.current_topic
                ),
                done=(self.current_round >= self.max_rounds),
                info={
                    "current_round": self.current_round,
                    "current_topic": self.current_topic,
                    "agent_rewards": {agent_id: 0.0 for agent_id in action.actions}
                }
            )
            self._last_step = step_result
            return step_result

    def _update_topic(self, new_topic: str) -> None:
        """Update the current topic"""
        self.topics[self.current_round] = new_topic
        self.current_topic = new_topic
        logger.debug(f"Updated topic to: {new_topic}")

    def get_global_state(self) -> Dict[str, Any]:
        """Return relevant global state for agent context"""
        return {
            "current_round": self.current_round,
            "current_topic": self.current_topic,
            "round_messages": [
                {
                    "content": msg.content,
                    "message_type": msg.message_type
                } for msg in self.round_messages
            ]
        }

    def reset(self) -> None:
        """Reset the chat state"""
        self.current_round = 0
        self.round_messages = []
        self.topics = {}
        self.current_topic = ""
        self.current_speaker_index = 0
        self._last_step = None
        logger.info("GroupChat mechanism has been reset")