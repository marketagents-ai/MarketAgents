import random
from typing import List, Dict, Any, Union, Type, Literal
from pydantic import BaseModel, Field
from market_agents.environments.environment import (
    Mechanism, LocalAction, GlobalAction, LocalObservation, GlobalObservation,
    EnvironmentStep, ActionSpace, ObservationSpace, LocalEnvironmentStep
)
import logging

logger = logging.getLogger(__name__)

class GroupChatMessage(BaseModel):
    content: str
    message_type: Literal["propose_topic", "group_message"]
    agent_id: str

class GroupChatAction(LocalAction):
    action: GroupChatMessage

    @classmethod
    def sample(cls, agent_id: str) -> 'GroupChatAction':
        return cls(
            agent_id=agent_id, 
            action=GroupChatMessage(
                content="Sample message", 
                message_type="group_message"
            )
        )

    @classmethod
    def action_schema(cls) -> Type[BaseModel]:
        return cls.model_json_schema()

class GroupChatGlobalAction(GlobalAction):
    actions: Dict[str, Dict[str, Any]]

class GroupChatObservation(BaseModel):
    messages: List[GroupChatMessage]
    current_topic: str
    current_speaker: str

class GroupChatLocalObservation(LocalObservation):
    observation: GroupChatObservation

class GroupChatGlobalObservation(GlobalObservation):
    observations: Dict[str, GroupChatLocalObservation]
    all_messages: List[GroupChatMessage]
    current_topic: str
    speaker_order: List[str]

class GroupChatActionSpace(ActionSpace):
    allowed_actions: List[Type[LocalAction]] = [GroupChatAction]

class GroupChatObservationSpace(ObservationSpace):
    allowed_observations: List[Type[LocalObservation]] = [GroupChatLocalObservation]

class GroupChat(Mechanism):
    max_rounds: int = Field(..., description="Maximum number of chat rounds")
    current_round: int = Field(default=0, description="Current round number")
    messages: List[GroupChatMessage] = Field(default_factory=list)
    topics: List[str] = Field(default_factory=list)
    current_topic: str = Field(..., description="Current discussion topic")
    speaker_order: List[str] = Field(default_factory=list)
    current_speaker_index: int = Field(default=0)
    actions: GroupChatGlobalAction = Field(default=None)

    sequential: bool = Field(default=False, description="Whether the mechanism is sequential")
    def step(self, actions: Union[GroupChatGlobalAction, Dict[str, Any]]) -> LocalEnvironmentStep:
        logger.debug(f"Received actions of type: {type(actions).__name__}")
        logger.debug(f"Actions content: {actions}")
        
        # Handle if actions is a dict (possibly due to serialization)
        if isinstance(actions, dict):
            try:
                actions = GroupChatGlobalAction(actions=actions)
                logger.debug("Parsed actions into GroupChatGlobalAction.")
            except Exception as e:
                logger.error(f"Failed to parse actions into GroupChatGlobalAction: {e}")
                raise

        # Ensure actions is GroupChatGlobalAction
        if not isinstance(actions, GroupChatGlobalAction):
            logger.error(f"Expected GroupChatGlobalAction, got {type(actions).__name__}")
            raise TypeError(f"Expected GroupChatGlobalAction, got {type(actions).__name__}")

        self.current_round += 1
        logger.debug(f"Processing round {self.current_round} with actions: {actions}")

        new_messages = self._process_actions(actions)
        self.messages.extend(new_messages)

        observations = self._create_observations(new_messages)
        done = self.current_round >= self.max_rounds

        # Select next speaker
        if self.sequential and self.speaker_order:
            next_speaker = self._select_next_speaker()
            logger.debug(f"Next speaker selected: {next_speaker}")
        else:
            next_speaker = random.choice(self.speaker_order) if self.speaker_order else None

        # Optionally, update topic if a propose_topic message is found
        for message in new_messages:
            if message.message_type == "propose_topic":
                self._update_topic(message.content)

        global_observation = GroupChatGlobalObservation(
            observations=observations,
            all_messages=self.messages,
            current_topic=self.current_topic,
            speaker_order=self.speaker_order
        )

        local_steps = {}
        for agent_id in self.speaker_order:
            local_steps[agent_id] = LocalEnvironmentStep(
                observation=observations[agent_id],
                reward=0,  # You may want to implement a reward function
                done=done,
                info={
                    "current_round": self.current_round,
                    "current_topic": self.current_topic,
                    "all_messages": self.messages,
                    "speaker_order": self.speaker_order
                }
            )

        return local_steps
    
    def _process_actions(self, global_action: GroupChatGlobalAction) -> List[GroupChatMessage]:
        new_messages = []
        for agent_id, action_dict in global_action.actions.items():
            try:
                action = GroupChatAction.parse_obj(action_dict)
                new_messages.append(action.action)
            except Exception as e:
                logger.error(f"Failed to parse action for agent {agent_id}: {e}")
                continue  # Skip invalid actions
        return new_messages

    def _update_topic(self, new_topic: str):
        self.topics.append(new_topic)
        self.current_topic = new_topic
        logger.info(f"Updated topic to: {new_topic}")

    def _create_observations(self, new_messages: List[GroupChatMessage]) -> Dict[str, GroupChatLocalObservation]:
        observations = {}
        for agent_id in self.speaker_order:
            observation = GroupChatObservation(
                messages=new_messages,
                current_topic=self.current_topic,
                current_speaker=self.speaker_order[self.current_speaker_index]
            )
            observations[agent_id] = GroupChatLocalObservation(
                agent_id=agent_id,
                observation=observation
            )
        return observations

    def get_global_state(self) -> Dict[str, Any]:
        return {
            "current_round": self.current_round,
            "messages": [message.model_dump() for message in self.messages],
            "current_topic": self.current_topic,
            "speaker_order": self.speaker_order,
            "current_speaker_index": self.current_speaker_index
        }

    def reset(self) -> None:
        self.current_round = 0
        self.messages = []
        self.current_speaker_index = 0
        logger.info("GroupChat mechanism has been reset.")

    def _select_next_speaker(self) -> str:
        self.current_speaker_index = (self.current_speaker_index + 1) % len(self.speaker_order)
        return self.speaker_order[self.current_speaker_index]
