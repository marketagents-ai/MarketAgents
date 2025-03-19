import asyncio
from enum import Enum
from typing import List, Dict, Any, Union, Type, Optional
from pydantic import BaseModel, Field, PrivateAttr, field_validator
from datetime import datetime
import logging

from market_agents.orchestrators.group_chat.groupchat_api_utils import GroupChatAPIUtils
from market_agents.agents.cognitive_steps import ActionStep
from market_agents.environments.environment import (
    Mechanism, LocalAction, GlobalAction, LocalObservation, GlobalObservation,
    EnvironmentStep, ActionSpace, MultiAgentEnvironment, ObservationSpace, LocalEnvironmentStep
)
from market_agents.orchestrators.parallel_cognitive_steps import ParallelCognitiveProcessor

logger = logging.getLogger(__name__)

from typing import List, Dict, Any, Union, Type, Optional
from pydantic import BaseModel, Field
import logging

from market_agents.environments.environment import (
    Mechanism, LocalAction, GlobalAction, LocalObservation, GlobalObservation,
    EnvironmentStep, ActionSpace, ObservationSpace, LocalEnvironmentStep
)

logger = logging.getLogger(__name__)

class MessageType(str, Enum):
    CHAT = "chat_message"
    TOPIC_PROPOSAL = "propose_topic"

class GroupChatMessage(BaseModel):
    content: str
    message_type: MessageType = MessageType.CHAT

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        """Custom serialization to handle Enum type."""
        return {
            "content": self.content,
            "message_type": self.message_type.value
        }

    def model_dump(self, *args, **kwargs) -> Dict[str, Any]:
        """Ensure consistent serialization with newer Pydantic versions."""
        return self.dict(*args, **kwargs)

    @classmethod
    def set_topic_proposal_phase(cls, is_proposal_phase: bool):
        """Control when topic proposals are allowed and set message type."""
        cls._is_topic_proposal_phase = is_proposal_phase
        # Set the default message type based on the phase
        cls.model_fields['message_type'].default = MessageType.TOPIC_PROPOSAL if is_proposal_phase else MessageType.CHAT

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
    """Mechanism that manages group chat interactions with API integration."""
    sequential: bool = Field(
        default=False,
        description="Whether agents speak in sequential order"
    )
    current_round: int = Field(
        default=0,
        description="Current round number"
    )
    max_rounds: int = Field(
        default=3,
        description="Maximum number of rounds"
    )
    initial_topic: str = Field(
        default="",
        description="Initial topic for discussion"
    )
    form_cohorts: bool = Field(
        default=False,
        description="Whether to organize agents into cohorts"
    )
    group_size: Optional[int] = Field(
        default=None,
        description="Size of chat cohorts when form_cohorts is True"
    )
    api_url: str = Field(
        default="http://localhost:8002",
        description="URL for the GroupChat API"
    )
    cohorts: Dict[str, List[Any]] = Field(
        default_factory=dict,
        description="Mapping of cohort IDs to lists of agents"
    )
    round_messages: Dict[str, List[GroupChatMessage]] = Field(
        default_factory=lambda: {"default": []},
        description="Messages from current round by cohort"
    )
    api_utils: GroupChatAPIUtils = Field(
        default=None,
        description="API utilities for group chat",
        exclude=True
    )
    _cognitive_processor: Optional[ParallelCognitiveProcessor] = PrivateAttr(
        default=None
    )


    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context) -> None:
        """Initialize API utils after model initialization."""
        super().model_post_init(__context)
        if not self.api_utils:
            self.api_utils = GroupChatAPIUtils(self.api_url, logger)

    async def form_agent_cohorts(self, agents: List[Any]) -> None:
        """Form cohorts and initialize topics."""
        if not self.form_cohorts or not self.group_size:
            return

        # Register agents with API first
        await self.api_utils.register_agents(agents)
        
        # Get cohort assignments from API
        agent_ids = [agent.id for agent in agents]
        cohorts_info = await self.api_utils.form_cohorts(agent_ids, self.group_size)
        
        # Store cohort information locally
        self.cohorts.clear()
        self.round_messages.clear()
        
        for cohort in cohorts_info:
            cohort_id = cohort["cohort_id"]
            cohort_agent_ids = cohort["agent_ids"]
            cohort_agents = [agent for agent in agents if agent.id in cohort_agent_ids]
            self.cohorts[cohort_id] = cohort_agents
            self.round_messages[cohort_id] = []

        # Initialize topics if processor is available
        if self._cognitive_processor:
            logger.info("Starting topic proposals for cohorts...")
            await self._initialize_cohort_topics()
        else:
            logger.warning("No cognitive processor available for topic proposals")

    async def _initialize_cohort_topics(self) -> None:
        """Initialize topics for all cohorts using parallel processing."""
        if not self.cohorts:
            logger.warning("No cohorts available for topic initialization")
            return

        # Prepare proposer agents
        proposer_agents = []
        proposer_info = []

        for cohort_id, cohort_agents in self.cohorts.items():
            # First select a proposer through the API
            proposer_id = await self.api_utils.select_proposer(
                cohort_id=cohort_id,
                agent_ids=[agent.id for agent in cohort_agents]
            )
            if not proposer_id:
                logger.error(f"Failed to select proposer for cohort {cohort_id}")
                continue

            # Find the proposer agent
            proposer = next((agent for agent in cohort_agents if agent.id == proposer_id), None)
            if not proposer:
                logger.error(f"Selected proposer {proposer_id} not found in cohort {cohort_id}")
                continue

            logger.info(f"Setting up topic proposer {proposer.id} for cohort {cohort_id}")
            proposer.task = (
                f"You are the group chat topic proposer. "
                f"Propose a topic related to {self.initial_topic}, explaining why it matters."
            )
            proposer_agents.append(proposer)
            proposer_info.append((cohort_id, proposer.id))

        # Run parallel action for all proposers
        GroupChatMessage.set_topic_proposal_phase(True)
        if proposer_agents and self._cognitive_processor:
            try:
                logger.info(f"Running parallel topic proposals for {len(proposer_agents)} cohorts")
                outputs = await self._cognitive_processor.run_parallel_action(
                    agents=proposer_agents,
                    environment_name="group_chat"
                )
                
                # Process results and update topics
                for (cohort_id, proposer_id), output in zip(proposer_info, outputs):
                    topic = self._extract_topic_from_proposal(output)
                    if topic:
                        await self.api_utils.propose_topic(
                            agent_id=proposer_id,
                            cohort_id=cohort_id,
                            topic=topic,
                            round_num=0
                        )
                        # Use the logger utility instead of direct logging
                        from market_agents.orchestrators.logger_utils import log_topic_proposal
                        log_topic_proposal(logger, cohort_id, proposer_id.replace('agent_', ''), topic)
                    else:
                        logger.warning(f"Failed to extract topic from proposal for cohort {cohort_id}")
            except Exception as e:
                logger.error(f"Error during topic proposal: {str(e)}", exc_info=True)
                raise
            finally:
                GroupChatMessage.set_topic_proposal_phase(False)

    def _extract_topic_from_proposal(self, proposal) -> Optional[str]:
        """Extract topic from LLM response."""
        try:
            if proposal and proposal.json_object:
                root_obj = proposal.json_object.object
                if "action" in root_obj and "content" in root_obj["action"]:
                    return root_obj["action"]["content"]
            return proposal.str_content.strip() if proposal.str_content else None
        except Exception as e:
            logger.error(f"Error extracting topic: {e}")
            return None

    def step(
        self,
        action: Union[GroupChatAction, GroupChatGlobalAction, GlobalAction],
        cohort_id: Optional[str] = None
    ) -> Union[LocalEnvironmentStep, EnvironmentStep]:
        """Execute a step in the environment."""
        # Check if we're done before incrementing
        done = self.current_round >= self.max_rounds - 1

        # Use provided cohort_id or default
        effective_cohort = cohort_id if cohort_id else "default"
        
        # Initialize cohort's round_messages if needed
        if effective_cohort not in self.round_messages:
            self.round_messages[effective_cohort] = []

        # Handle single agent action (sequential mode)
        if isinstance(action, LocalAction):
            # Extract message from action
            if isinstance(action.action, dict):
                content = action.action.get('action', {}).get('content', '')
                message_type = action.action.get('action', {}).get('message_type', MessageType.CHAT)
            elif isinstance(action.action, GroupChatMessage):
                content = action.action.content
                message_type = action.action.message_type
            else:
                content = str(action.action)
                message_type = MessageType.CHAT

            # Create message
            action_content = GroupChatMessage(
                content=content,
                message_type=message_type
            )

            # Store the message in a format that matches what we're logging
            self.round_messages[effective_cohort].append({
                "content": content,
                "message_type": message_type,
                "agent_id": action.agent_id
            })

            # Post message to API if cohort_id provided
            if cohort_id:
                asyncio.create_task(
                    self.api_utils.post_message(
                        agent_id=action.agent_id,
                        cohort_id=cohort_id,
                        content=action_content.content,
                        round_num=self.current_round,
                        sub_round_num=0
                    )
                )

            # Create and return local step
            obs = GroupChatObservation(
                current_topic=self.initial_topic,
                agent_message=action_content
            )
            local_obs = GroupChatLocalObservation(
                agent_id=action.agent_id,
                observation=obs
            )
            local_step = LocalEnvironmentStep(
                observation=local_obs,
                done=done,
                info={
                    "round": self.current_round,
                    "cohort_id": effective_cohort
                }
            )
            
            # Increment round counter after processing if sequential
            if self.sequential:
                self.current_round += 1
            return local_step

        # Handle global actions (batch mode)
        else:
            # Convert action to GroupChatGlobalAction if needed
            if isinstance(action, GlobalAction):
                chat_actions = {}
                for agent_id, local_action in action.actions.items():
                    # Extract message from action
                    if isinstance(local_action.action, dict):
                        content = local_action.action.get('action', {}).get('content', '')
                        message_type = local_action.action.get('action', {}).get('message_type', MessageType.CHAT)
                    elif isinstance(local_action.action, GroupChatMessage):
                        content = local_action.action.content
                        message_type = local_action.action.message_type
                    else:
                        content = str(local_action.action)
                        message_type = MessageType.CHAT

                    chat_actions[agent_id] = GroupChatAction(
                        agent_id=agent_id,
                        action=GroupChatMessage(
                            content=content,
                            message_type=message_type
                        )
                    )
                action = GroupChatGlobalAction(actions=chat_actions)

            # Store messages for this round
            for agent_id, local_action in action.actions.items():
                self.round_messages[effective_cohort].append({
                    "content": local_action.action.content,
                    "message_type": local_action.action.message_type,
                    "agent_id": agent_id
                })

            # Post messages to API if cohort_id provided
            if cohort_id:
                for agent_id, local_action in action.actions.items():
                    asyncio.create_task(
                        self.api_utils.post_message(
                            agent_id=agent_id,
                            cohort_id=cohort_id,
                            content=local_action.action.content,
                            round_num=self.current_round,
                            sub_round_num=0
                        )
                    )

            # Create observations for each agent
            observations = {}
            for agent_id, local_action in action.actions.items():
                observations[agent_id] = GroupChatLocalObservation(
                    agent_id=agent_id,
                    observation=GroupChatObservation(
                        current_topic=self.initial_topic,
                        agent_message=local_action.action
                    )
                )

            # Create global observation
            global_obs = GroupChatGlobalObservation(
                observations=observations,
                round_messages=self.round_messages[effective_cohort],
                current_topic=self.initial_topic
            )

            # Create global step
            global_step = EnvironmentStep(
                global_observation=global_obs,
                done=done,
                info={
                    "round": self.current_round,
                    "cohort_id": effective_cohort
                }
            )

            # Increment round counter after processing batch
            self.current_round += 1
            return global_step

    def get_global_state(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get global state, filtered by agent's cohort if specified."""
        state = {
            "current_round": self.current_round,
            "current_topic": self.initial_topic,
            "max_rounds": self.max_rounds
        }

        if self.form_cohorts:
            if agent_id:
                # Find agent's cohort
                cohort_id = next(
                    (cid for cid, agents in self.cohorts.items() 
                    if any(a.id == agent_id for a in agents)),
                    None
                )
                if cohort_id:
                    state.update({
                        "cohort_id": cohort_id,
                        "cohort_agents": [a.id for a in self.cohorts[cohort_id]],
                        "round_messages": self.round_messages.get(cohort_id, [])
                    })
            else:
                # Return all cohorts' information
                state.update({
                    "cohorts": {
                        cid: [a.id for a in agents] 
                        for cid, agents in self.cohorts.items()
                    },
                    "round_messages": self.round_messages
                })
        else:
            # Not using cohorts, return all messages
            state["round_messages"] = self.round_messages.get("default", [])

        return state

    def reset(self) -> None:
        """Reset the chat state."""
        self.current_round = 0
        self.round_messages = {"default": []}
        self.cohorts.clear()
        logger.info("GroupChat mechanism has been reset")

class GroupChatEnvironment(MultiAgentEnvironment):
    """Multi-agent environment that orchestrates a group chat session."""
    name: str = Field(default="group_chat", description="Name of the environment")
    address: str = Field(default="group_chat", description="Address of the environment")
    max_steps: int = Field(default=3, description="Maximum number of steps")
    action_space: GroupChatActionSpace = Field(
        default_factory=GroupChatActionSpace,
        description="Defines the action space for chat messages"
    )
    observation_space: GroupChatObservationSpace = Field(
        default_factory=GroupChatObservationSpace,
        description="Observation space"
    )
    mechanism: GroupChat = Field(
        default_factory=lambda: GroupChat(max_rounds=3, sequential=False),
        description="The group chat mechanism"
    )

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **config):
        try:
            # Extract mechanism-specific config
            mechanism_config = {
                'sequential': config.get('sequential', False),
                'max_rounds': config.get('max_rounds', 3),
                'form_cohorts': config.get('form_cohorts', False),
                'group_size': config.get('group_size'),
                'api_url': config.get('api_url', 'http://localhost:8002'),
                'initial_topic': config.get('initial_topic', '')
            }

            # Initialize mechanism
            mechanism = GroupChat(**mechanism_config)

            # Initialize parent class with required fields
            parent_config = {
                'name': config.get('name', 'group_chat'),
                'address': config.get('address', 'group_chat'),
                'max_steps': config.get('max_steps', 3),
                'action_space': GroupChatActionSpace(),
                'observation_space': GroupChatObservationSpace(),
                'mechanism': mechanism
            }
            super().__init__(**parent_config)

            # Initialize cognitive processor if ai_utils provided
            if 'ai_utils' in config and 'storage_service' in config:
                self.mechanism._cognitive_processor = ParallelCognitiveProcessor(
                    ai_utils=config['ai_utils'],
                    storage_service=config['storage_service'],
                    logger=logging.getLogger(__name__),
                    tool_mode=config.get('tool_mode', False)
                )

        except Exception as e:
            raise ValueError(f"Failed to initialize GroupChatEnvironment: {e}")

    def step(self, action: GlobalAction, cohort_id: Optional[str] = None) -> EnvironmentStep:
        """Execute a step in the environment with optional cohort_id."""
        return self.mechanism.step(action, cohort_id=cohort_id)

    def reset(self) -> GlobalObservation:
        """Reset the environment."""
        self.mechanism.reset()
        if self.initial_topic:
            self.mechanism.initial_topic = self.initial_topic
        return GlobalObservation(observations={})

    def get_global_state(self, agent_id: str = None) -> Dict[str, Any]:
        """Return the environment's global state with filtered mechanism state."""
        mechanism_state = self.mechanism.get_global_state(agent_id) if agent_id else self.mechanism.get_global_state()
        
        return {
            **mechanism_state,
            "current_step": self.mechanism.current_round,
            "max_steps": self.mechanism.max_rounds
        }