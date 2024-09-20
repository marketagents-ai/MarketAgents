from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel, Field
from datetime import datetime
from market_agents.environments.environment import (
    Mechanism, LocalAction, GlobalAction, LocalObservation, GlobalObservation,
    EnvironmentStep, ActionSpace, ObservationSpace
)

class GroupChatMessage(BaseModel):
    agent_id: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

class GroupChatAction(LocalAction):
    message: str
    propose_topic: Optional[str] = None

    @classmethod
    def sample(cls, agent_id: str) -> 'GroupChatAction':
        return cls(agent_id=agent_id, message="Sample message", propose_topic=None)

    @classmethod
    def action_schema(cls) -> Type[BaseModel]:
        return cls.model_json_schema()

class GroupChatGlobalAction(GlobalAction):
    actions: Dict[str, GroupChatAction]

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
    current_topic: str = Field(default="", description="Current discussion topic")
    speaker_order: List[str] = Field(default_factory=list)
    current_speaker_index: int = Field(default=0)

    sequential: bool = Field(default=True, description="Whether the mechanism is sequential")

    def step(self, action: GroupChatGlobalAction) -> EnvironmentStep:
        self.current_round += 1
        new_messages = self._process_actions(action.actions)
        self.messages.extend(new_messages)

        observations = self._create_observations(new_messages)
        done = self.current_round >= self.max_rounds

        return EnvironmentStep(
            global_observation=GroupChatGlobalObservation(
                observations=observations,
                all_messages=self.messages,
                current_topic=self.current_topic,
                speaker_order=self.speaker_order
            ),
            done=done,
            info={"current_round": self.current_round}
        )

    def _process_actions(self, actions: Dict[str, GroupChatAction]) -> List[GroupChatMessage]:
        new_messages = []
        for agent_id, action in actions.items():
            if action.propose_topic:
                self._update_topic(action.propose_topic)
            new_messages.append(GroupChatMessage(agent_id=agent_id, content=action.message))
        return new_messages

    def _update_topic(self, new_topic: str):
        self.topics.append(new_topic)
        self.current_topic = new_topic

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
        self.current_topic = ""
        self.current_speaker_index = 0

    def _select_next_speaker(self) -> str:
        self.current_speaker_index = (self.current_speaker_index + 1) % len(self.speaker_order)
        return self.speaker_order[self.current_speaker_index]
