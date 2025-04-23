from datetime import datetime
from typing import Dict, Any, List, Type, Optional, Union
from pydantic import BaseModel, Field
import logging

from market_agents.agents.cognitive_schemas import ChainOfThoughtSchema, ThoughtStep
from market_agents.environments.environment import (
    Mechanism, LocalAction, LocalObservation, ActionSpace, 
    ObservationSpace, MultiAgentEnvironment, LocalEnvironmentStep, StrAction
)
from minference.lite.models import CallableTool, StructuredTool

logger = logging.getLogger(__name__)

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
        
        # Extract final answer from the action
        final_answer = ""
        if hasattr(action, 'action') and hasattr(action.action, 'final_answer'):
            final_answer = action.action.final_answer
        elif isinstance(action.action, dict) and 'final_answer' in action.action:
            final_answer = action.action['final_answer']
        else:
            final_answer = str(action.action)
        
        self.chat_history.append(ChatMessage(
            content=final_answer,
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

    def get_global_state(self, agent_id: Optional[str] = None) -> List[ChatMessage]:
        """Get the global state of the chat, optionally filtered for an agent."""
        return self.chat_history

    def add_user_message(self, content: str):
        """Add a user message to the chat history"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.chat_history.append(ChatMessage(
            content=content,
            timestamp=timestamp,
            role="user"
        ))

class ToolEnabledChatActionSpace(ActionSpace):
    """ActionSpace that supports both chat actions and tools"""
    allowed_actions: List[Any] = Field(default_factory=list)
    tools: List[Union[CallableTool, StructuredTool]] = Field(
        default_factory=list,
        description="Tools available to the agent in this action space"
    )
    workflow: bool = Field(
        default=False,
        description="Whether tools should be executed in sequence"
    )
    
    def __init__(self, tools=None, **data):
        # Initialize with empty lists
        super().__init__()
        
        # Set workflow flag explicitly
        self.workflow = False
        
        # Process tools
        if tools and len(tools) > 0:
            # Create speak_to_user tool
            speak_to_user_tool = StructuredTool(
                json_schema=ChainOfThoughtSchema.model_json_schema(),
                name="speak_to_user",
                description="Respond directly to the user with reasoning and final answer"
            )
            
            # Add speak_to_user to tools list - put it FIRST for priority
            all_tools = [speak_to_user_tool] + list(tools)
            
            # Set both tools and allowed_actions to ensure they're found by ActionStep
            self.tools = all_tools
            self.allowed_actions = all_tools
            
            logger.info(f"Created action space with {len(all_tools)} tools (including speak_to_user)")
            for tool in all_tools:
                if hasattr(tool, 'name'):
                    logger.info(f"  - Tool: {tool.name}")
        else:
            # When no tools are provided, use the standard ChatAction
            self.allowed_actions = [ChatAction]
            self.tools = []
            logger.info("Created action space with only ChatAction")
    
    def set_tools(self, tools):
        """Update available tools"""
        if tools and len(tools) > 0:
            # Create speak_to_user tool
            speak_to_user_tool = StructuredTool(
                json_schema=ChainOfThoughtSchema.model_json_schema(),
                name="speak_to_user",
                description="Respond directly to the user with reasoning and final answer"
            )
            
            # Create combined tools list with speak_to_user first
            all_tools = [speak_to_user_tool] + list(tools)
            
            # Update both tools and allowed_actions
            self.tools = all_tools
            self.allowed_actions = all_tools
            
            logger.info(f"Updated action space with {len(all_tools)} tools (including speak_to_user)")
            for tool in all_tools:
                if hasattr(tool, 'name'):
                    logger.info(f"  - Tool: {tool.name}")
        else:
            # When no tools are provided, use the standard ChatAction
            self.allowed_actions = [ChatAction]
            self.tools = []
            logger.info("Updated action space with only ChatAction")
    
    def get_action_schema(self) -> Dict[str, Any]:
        """Return JSON schema for all available tools"""
        schemas = {}
        
        # Add schemas for all tools
        for tool in self.tools:
            if hasattr(tool, 'name') and hasattr(tool, 'json_schema'):
                schemas[tool.name] = tool.json_schema()
                
        return schemas

class ChatObservationSpace(ObservationSpace):
    allowed_observations: List[Type[LocalObservation]] = [ChatObservation]

class ChatEnvironment(MultiAgentEnvironment):
    """Chat environment with support for both conversation and tools"""
    name: str = "chat_environment"
    action_space: ActionSpace = Field(default_factory=lambda: ToolEnabledChatActionSpace())
    observation_space: ObservationSpace = Field(default_factory=ChatObservationSpace)
    mechanism: Mechanism = Field(default_factory=ChatMechanism)

    def __init__(self, name="tool_enabled_chat", tools=None):
        super().__init__(
            name=name,
            action_space=ToolEnabledChatActionSpace(tools=tools),
            observation_space=ChatObservationSpace(),
            mechanism=ChatMechanism()
        )
        logger.info(f"Created ToolEnabledChatEnvironment with {len(tools or [])} tools")

    def update_tools(self, tools):
        """Update tools in the action space"""
        if isinstance(self.action_space, ToolEnabledChatActionSpace):
            self.action_space.set_tools(tools)

    def get_global_state(self, agent_id: Optional[str] = None) -> Any:
        """Get the global state, optionally filtered for an agent."""
        history = self.mechanism.get_global_state(agent_id)
        # Include tools in the global state
        if hasattr(self.action_space, 'tools'):
            tool_names = [t.name for t in self.action_space.tools if hasattr(t, 'name')]
            return {
                "chat_history": history,
                "available_tools": tool_names
            }
        return history