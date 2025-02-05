from typing import List, Dict, Any, Optional, Union, Type
from pydantic import BaseModel, Field, PrivateAttr
from market_agents.environments.environment import (
    Mechanism,
    LocalAction,
    GlobalAction,
    LocalEnvironmentStep,
    EnvironmentStep,
    LocalObservation,
    GlobalObservation,
    ActionSpace,
    ObservationSpace
)
import logging

logger = logging.getLogger(__name__)

class GroupChatMessage(BaseModel):
    content: str = Field(
        description="agent's opinions & arguments on the topic. express yourself with emojis"
    )
    message_type: Optional[str] = Field(
        default=None,
        description="Type of message (e.g., propose_topic)."
    )

class GroupChatAction(LocalAction):
    action: GroupChatMessage = Field(
        description="The message action taken by the agent."
    )

    def sample(self) -> "GroupChatAction":
        return GroupChatAction(
            agent_id="sample_agent",
            action=GroupChatMessage(content="sample content")
        )

class GroupChatGlobalAction(GlobalAction):
    actions: Dict[str, Dict[str, Any]] = Field(
        description="Dictionary of actions by agent."
    )

    def sample(self) -> "GroupChatGlobalAction":
        return GroupChatGlobalAction(
            actions={
                "sample_agent": {
                    "agent_id": "sample_agent",
                    "action": {"content": "sample global content"}
                }
            }
        )

class GroupChatObservation(BaseModel):
    messages: List[GroupChatMessage] = Field(
        description="List of messages in the current observation."
    )
    current_topic: str = Field(
        description="The current topic of discussion."
    )

class GroupChatLocalObservation(LocalObservation):
    observation: GroupChatObservation = Field(
        description="Local observation for a specific agent."
    )

class GroupChatGlobalObservation(GlobalObservation):
    observations: Dict[str, GroupChatLocalObservation] = Field(
        description="Dictionary of local observations by agent."
    )
    all_messages: List[GroupChatMessage] = Field(
        description="All messages exchanged in the chat."
    )
    current_topic: str = Field(
        description="The current topic of discussion."
    )

class GroupChatActionSpace(ActionSpace):
    allowed_actions: List[Type[LocalAction]] = Field(
        default=[GroupChatAction],
        description="Allowed actions"
    )

class GroupChatObservationSpace(ObservationSpace):
    allowed_observations: List[Type[LocalObservation]] = Field(
        default=[GroupChatLocalObservation],
        description="Allowed observations"
    )

class GroupChat(Mechanism):
    max_rounds: int = Field(
        ...,
        description="Maximum number of rounds"
    )
    current_round: int = Field(
        default=0,
        description="Current round number"
    )
    messages: List[GroupChatMessage] = Field(
        default_factory=list,
        description="List of messages"
    )
    topics: Dict[int, str] = Field(
        default_factory=dict,
        description="Topics by round"
    )
    current_topic: str = Field(
        default="",
        description="Current topic"
    )
    speaker_order: Optional[List[str]] = Field(
        default=None,
        description="Order of speakers"
    )
    current_speaker_index: int = Field(
        default=0,
        description="Index of the current speaker"
    )

    _last_step: Optional[Union[LocalEnvironmentStep, EnvironmentStep]] = PrivateAttr(
        default=None
    )

    @property
    def last_step(self) -> Optional[Union[LocalEnvironmentStep, EnvironmentStep]]:
        return self._last_step

    @last_step.setter
    def last_step(self, step: Union[LocalEnvironmentStep, EnvironmentStep, None]):
        self._last_step = step

    def step(
        self,
        action: Union[GroupChatAction, GroupChatGlobalAction, Dict[str, Any]]
    ) -> Union[LocalEnvironmentStep, EnvironmentStep]:
        logger.debug(f"Received action: {action}")

        if self.sequential:
            if isinstance(action, dict):
                action = GroupChatAction.model_validate(action)
            if not isinstance(action, GroupChatAction):
                raise TypeError(f"Expected GroupChatAction, got {type(action).__name__}")

            cur_speaker = self.speaker_order[self.current_speaker_index]
            if action.agent_id != cur_speaker:
                raise ValueError(f"It's not {action.agent_id}'s turn")

            self.current_round += 1
            self.messages.append(action.action)

            if action.action.message_type == "propose_topic":
                self._update_topic(action.action.content)

            observation = self._create_local_observation(agent_id=action.agent_id, msg=action.action)
            done = (self.current_round >= self.max_rounds)
            self._select_next_speaker()

            local_step = LocalEnvironmentStep(
                observation=observation,
                reward=1.0,
                done=done,
                info={
                    "agent_rewards": {action.agent_id: 1.0},
                    "current_round": self.current_round,
                    "current_topic": self.current_topic,
                    "all_messages": [m.model_dump() for m in self.messages],
                    "speaker_order": self.speaker_order
                }
            )
            self.last_step = local_step
            return local_step

        else:
            if isinstance(action, dict):
                action = GroupChatGlobalAction.model_validate(action)
            if not isinstance(action, GroupChatGlobalAction):
                raise TypeError(f"Expected GroupChatGlobalAction, got {type(action).__name__}")

            self.current_round += 1
            new_items = self._process_actions(action)

            for agent_id, msg in new_items:
                self.messages.append(msg)

            done = (self.current_round >= self.max_rounds)

            for agent_id, msg in new_items:
                if msg.message_type == "propose_topic":
                    self._update_topic(msg.content)

            observations = self._create_observations(new_items)
            global_observation = GroupChatGlobalObservation(
                observations=observations,
                all_messages=self.messages,
                current_topic=self.current_topic
            )

            env_step = EnvironmentStep(
                global_observation=global_observation,
                done=done,
                info={
                    "agent_rewards": {agent_id: 1.0 for agent_id in action.actions},
                    "current_round": self.current_round,
                    "all_messages": [m.model_dump() for m in self.messages],
                }
            )
            self.last_step = env_step
            return env_step

    def _update_topic(self, new_topic: str):
        self.topics[self.current_round] = new_topic
        self.current_topic = new_topic

    def _process_actions(
        self, global_action: GroupChatGlobalAction
    ) -> List[tuple[str, GroupChatMessage]]:
        new_items = []
        for agent_id, action_dict in global_action.actions.items():
            try:
                local_action = GroupChatAction.model_validate(action_dict)
                new_items.append((agent_id, local_action.action))
            except Exception as e:
                logger.error(f"Failed to parse action for agent {agent_id}: {e}")
        return new_items

    def _create_local_observation(
        self, agent_id: str, msg: GroupChatMessage
    ) -> GroupChatLocalObservation:
        return GroupChatLocalObservation(
            agent_id=agent_id,
            observation=GroupChatObservation(
                messages=[msg],
                current_topic=self.current_topic
            )
        )

    def _create_observations(
        self, items: List[tuple[str, GroupChatMessage]]
    ) -> Dict[str, GroupChatLocalObservation]:
        observations = {}
        for agent_id, msg in items:
            ob = GroupChatObservation(
                messages=[m for _, m in items],
                current_topic=self.current_topic
            )
            observations[agent_id] = GroupChatLocalObservation(
                agent_id=agent_id,
                observation=ob
            )
        return observations

    def _select_next_speaker(self) -> None:
        self.current_speaker_index += 1
        if self.speaker_order and self.current_speaker_index >= len(self.speaker_order):
            self.current_speaker_index = 0

    def get_global_state(self) -> Dict[str, Any]:
        return {
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "messages": [m.model_dump() for m in self.messages],
            "topics_by_round": self.topics,
            "current_topic": self.current_topic,
            "speaker_order": self.speaker_order,
            "current_speaker_index": self.current_speaker_index
        }