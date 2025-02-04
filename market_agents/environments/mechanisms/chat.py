from datetime import datetime
from typing import Dict, Any, List, Type
from pydantic import BaseModel, Field
from market_agents.agents.cognitive_schemas import ChainOfThoughtSchema, ThoughtStep
from market_agents.environments.environment import (
    Mechanism, LocalAction, LocalObservation, ActionSpace, 
    ObservationSpace, MultiAgentEnvironment, LocalEnvironmentStep
)

class ChatMessage(BaseModel):
    content: str
    timestamp: str
    role: str = "user"

class ChatAction(LocalAction):
    """Response action for chat using ChainOfThoughtSchema"""
    agent_id: str
    action: ChainOfThoughtSchema = Field(
        description="Response containing thought process and actual response"
    )

    @classmethod
    def sample(cls, agent_id: str) -> 'ChatAction':
        return cls(
            agent_id=agent_id,
            action=ChainOfThoughtSchema(
                thoughts=[ThoughtStep(reasoning="Sample thinking")],
                final_answer="Sample response"
            )
        )

class ChatObservation(LocalObservation):
    """Message observation for chat"""
    agent_id: str
    observation: ChatMessage
    chat_history: List[ChatMessage] = Field(default_factory=list)

    @classmethod
    def sample(cls, agent_id: str) -> 'ChatObservation':
        return cls(
            agent_id=agent_id,
            observation=ChatMessage(
                content="Sample message",
                timestamp="2024-01-01",
                role="user"
            ),
            chat_history=[]
        )

class ChatMechanism(Mechanism):
    chat_history: List[ChatMessage] = Field(default_factory=list)
    sequential: bool = Field(default=True)
    
    def step(self, action: LocalAction) -> LocalEnvironmentStep:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        self.chat_history.append(ChatMessage(
            content=action.action.final_answer,
            timestamp=timestamp,
            role="assistant"
        ))
        
        last_user_message = next(
            (msg for msg in reversed(self.chat_history) 
             if msg.role == "user"), 
            ChatMessage(content="No user message found", timestamp=timestamp, role="user")
        )
        
        observation = ChatObservation(
            agent_id=action.agent_id,
            observation=last_user_message,
            chat_history=self.chat_history.copy()
        )
        
        return LocalEnvironmentStep(
            observation=observation,
            done=False,
            info={"chat_history": self.chat_history}
        )

    def get_global_state(self) -> List[ChatMessage]:
        return self.chat_history

    def add_user_message(self, content: str):
        """Add a user message to the chat history"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.chat_history.append(ChatMessage(
            content=content,
            timestamp=timestamp,
            role="user"
        ))

class ChatActionSpace(ActionSpace):
    allowed_actions: List[Type[LocalAction]] = [ChatAction]

class ChatObservationSpace(ObservationSpace):
    allowed_observations: List[Type[LocalObservation]] = [ChatObservation]

class ChatEnvironment(MultiAgentEnvironment):
    name: str = "chat_environment"
    action_space: ActionSpace = Field(default_factory=ChatActionSpace)
    observation_space: ObservationSpace = Field(default_factory=ChatObservationSpace)
    mechanism: Mechanism = Field(default_factory=ChatMechanism)