# groupchat_orchestrator.py

import asyncio
import logging
import random
import json
from typing import List, Dict, Any, Optional
from pydantic import Field

from market_agents.orchestrators.base_orchestrator import BaseEnvironmentOrchestrator
from market_agents.agents.market_agent import MarketAgent
from market_agents.agents.personas.persona import Persona, generate_persona, save_persona_to_file
from market_agents.environments.environment import MultiAgentEnvironment, EnvironmentStep
from market_agents.environments.mechanisms.group_chat import (
    GroupChat,
    GroupChatAction,
    GroupChatActionSpace,
    GroupChatGlobalAction,
    GroupChatMessage,
    GroupChatObservationSpace,
    GroupChatGlobalObservation
)
from market_agents.inference.message_models import LLMOutput, LLMPromptContext
from market_agents.agents.protocols.acl_message import ACLMessage
from market_agents.orchestrators.config import GroupChatConfig
from market_agents.logger_utils import (
    log_section,
    log_environment_setup,
    log_agent_init,
    log_running,
    log_perception,
    log_action,
    log_reflection,
    log_round,
    log_completion,
    print_ascii_art
)
from market_agents.orchestrators.insert_simulation_data import SimulationDataInserter

class GroupChatTracker:
    def __init__(self):
        self.messages: List[GroupChatMessage] = []
        self.topics: List[str] = []

    def add_message(self, message: GroupChatMessage):
        self.messages.append(message)

    def add_topic(self, topic: str):
        self.topics.append(topic)

    def get_summary(self):
        return {
            "total_messages": len(self.messages),
            "total_topics": len(self.topics)
        }

class GroupChatOrchestrator(BaseEnvironmentOrchestrator):
    environment_name: str = Field(default='group_chat')
    environments: Dict[str, MultiAgentEnvironment] = Field(default_factory=dict)
    trackers: Dict[str, GroupChatTracker] = Field(default_factory=dict)
    group_size: int = Field(default=2) 
    sub_rounds_per_group_chat: int = Field(default=2)
    agent_batches: List[List[MarketAgent]] = Field(default_factory=list)
    current_topic: str = Field(default="")
    agent_dict: Dict[str, MarketAgent] = Field(default_factory=dict)
    topic_proposer: Optional[str] = Field(default=None)

    def __init__(
        self,
        config: GroupChatConfig,
        agents: List[MarketAgent],
        ai_utils,
        data_inserter: SimulationDataInserter,
        logger=None
    ):
        # Initialize with default values first
        super().__init__(
            config=config,
            agents=agents,
            ai_utils=ai_utils,
            data_inserter=data_inserter,
            logger=logger,
            group_size=0,
            sub_rounds_per_group_chat=0,
            current_topic=""
        )
        # Then set the actual values
        self.group_size = config.group_size
        self.sub_rounds_per_group_chat = config.sub_rounds
        self.current_topic = config.initial_topic
        self.agent_dict = {agent.id: agent for agent in agents}
        self.batch_agents()
        self.setup_environment()

    def batch_agents(self):
        # Divide agents into batches for group chats
        self.agent_batches = [
            self.agents[i:i + self.group_size]
            for i in range(0, len(self.agents), self.group_size)
        ]
        self.logger.info(f"Agents divided into {len(self.agent_batches)} batches of up to {self.group_size} agents each.")

    def setup_environment(self):
        log_section(self.logger, "CONFIGURING GROUP CHAT ENVIRONMENTS")
        # Create GroupChat Environments per Batch
        for batch_index, batch in enumerate(self.agent_batches):
            group_chat = GroupChat(
                max_rounds=self.config.max_rounds,
                current_topic=self.current_topic,
                speaker_order=[str(agent.id) for agent in batch],
                sequential=False,
                sub_rounds=self.sub_rounds_per_group_chat
            )
            group_chat_env_name = f"group_chat_batch_{batch_index}"
            group_chat_env = MultiAgentEnvironment(
                name=f"{self.config.name}_batch_{batch_index}",
                address=f"{self.config.address}_{batch_index}",
                max_steps=self.config.max_rounds,
                action_space=GroupChatActionSpace(),
                observation_space=GroupChatObservationSpace(),
                mechanism=group_chat
            )
            self.environments[group_chat_env_name] = group_chat_env
            self.trackers[group_chat_env_name] = GroupChatTracker()
            log_environment_setup(self.logger, group_chat_env_name)
        # Assign environments to agents
        for agent in self.agents:
            agent.environments = self.environments

    async def run_environment(self, round_num: int):
        # Run multiple sub-rounds within group chats
        for sub_round in range(1, self.sub_rounds_per_group_chat + 1):
            group_chat_tasks = []
            for batch_index, batch in enumerate(self.agent_batches):
                group_chat_env_name = f"group_chat_batch_{batch_index}"
                task = asyncio.create_task(
                    self.run_group_chat_sub_round(
                        env_name=group_chat_env_name,
                        round_num=round_num,
                        sub_round_num=sub_round,
                        batch=batch
                    )
                )
                group_chat_tasks.append(task)
            await asyncio.gather(*group_chat_tasks)

    async def run_group_chat_sub_round(self, env_name: str, round_num: int, sub_round_num: int, batch: List[MarketAgent]):
        env = self.environments[env_name]
        tracker = self.trackers[env_name]
        log_running(self.logger, f"{env_name} - Sub-round {sub_round_num}")

        # Set system messages for agents
        self.set_agent_system_messages(env_name, round_num, sub_round_num=sub_round_num)

        # Run perception and action generation
        perception_prompts = await self.run_parallel_perceive(env_name, batch=batch)
        for prompt in perception_prompts:
            print(prompt)
        perceptions = await self.ai_utils.run_parallel_ai_completion(perception_prompts, update_history=False)
        self.data_inserter.insert_ai_requests(self.ai_utils.get_all_requests())
        perceptions_map = {perception.source_id: perception for perception in perceptions}

        for agent in batch:
            perception = perceptions_map.get(agent.id)
            if perception:
                log_section(self.logger, f"Current Agent:\nAgent {agent.index} with persona:\n{agent.persona}")
                perception_content = perception.json_object.object if perception.json_object else perception.str_content
                log_perception(self.logger, agent.index, f"{perception_content}")
                agent.last_perception = perception_content
            else:
                self.logger.warning(f"No perception found for agent {agent.index} in {env_name}")
                agent.last_perception = None

        perception_contents = [agent.last_perception or "" for agent in batch]
        action_prompts = await self.run_parallel_generate_action(env_name, perception_contents, batch=batch)
        actions = await self.ai_utils.run_parallel_ai_completion(action_prompts, update_history=False)
        self.data_inserter.insert_ai_requests(self.ai_utils.get_all_requests())
        actions_map = {action.source_id: action for action in actions}

        # Process actions
        global_action = self.process_group_chat_actions(actions_map)
        env_state = env.step(global_action)
        self.process_environment_state(env_state, batch)

    async def run_parallel_perceive(self, env_name: str, batch: List[MarketAgent]) -> List[LLMPromptContext]:
        perceive_prompts = []
        for agent in batch:
            perceive_prompt = await agent.perceive(env_name, return_prompt=True)
            perceive_prompts.append(perceive_prompt)
        return perceive_prompts

    async def run_parallel_generate_action(self, env_name: str, perceptions: List[str], batch: List[MarketAgent]) -> List[LLMPromptContext]:
        action_prompts = []
        for agent, perception in zip(batch, perceptions):
            action_prompt = await agent.generate_action(env_name, perception, return_prompt=True)
            action_prompts.append(action_prompt)
        return action_prompts
    def set_agent_system_messages(self, env_name: str, round_num: int, sub_round_num: int = None):
        # Set system messages for agents in the batch
        batch_index = int(env_name.split('_')[-1])
        batch_agents = self.agent_batches[batch_index]
        for agent in batch_agents:
            agent.system = (
                f"You are participating in sub-round {sub_round_num} of round {round_num} "
                f"in a group chat about '{self.current_topic}'. "
                f"Your role is {'buyer' if agent.role == 'buyer' else 'seller'}. "
                f"Engage in an informative and fun discussion and share your insights. "
                f"Use emojis to maintain a friendly and playful tone."
            )

    def process_group_chat_actions(self, actions_map: Dict[str, LLMOutput]) -> GroupChatGlobalAction:
        agent_actions = {}
        for agent_id, action_output in actions_map.items():
            try:
                action_content = action_output.json_object.object if action_output.json_object else action_output.str_content
                action_data = json.loads(action_content) if isinstance(action_content, str) else action_content
                group_chat_message = GroupChatMessage(
                    content=action_data['action']['content'],
                    message_type=action_data.get('message_type', 'group_message'),
                    agent_id=agent_id
                )
                group_chat_action = GroupChatAction(agent_id=agent_id, action=group_chat_message)
                agent_actions[agent_id] = group_chat_action.model_dump()
                agent = self.agent_dict.get(agent_id)
                log_action(self.logger, agent.index if agent else "Unknown", f"Message: {group_chat_message.content}")
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                self.logger.error(f"Error creating GroupChatAction for agent {agent_id}: {str(e)}")
                continue
        return GroupChatGlobalAction(actions=agent_actions)

    def process_environment_state(self, env_state: EnvironmentStep, batch: List[MarketAgent]):
        # Process messages and update agent observations
        global_observation = env_state.global_observation
        env_name = env_state.env_name
        tracker = self.trackers[env_name]
        for message in global_observation.all_messages:
            tracker.add_message(message)
        if global_observation.current_topic != self.current_topic:
            self.current_topic = global_observation.current_topic
            tracker.add_topic(self.current_topic)
        self.logger.info(f"Processed {len(global_observation.all_messages)} messages in {env_name}")

        for agent in batch:
            agent_observation = global_observation.observations.get(agent.id)
            if agent_observation:
                agent.last_observation = agent_observation
                agent.last_step = env_state
            else:
                agent.last_perception = None

    def get_round_summary(self, round_num: int) -> dict:
        # Return a summary of the round
        summary = {
            'round': round_num,
            'agent_states': [{
                'id': agent.id,
                'role': agent.role,
                'memory': agent.memory[-1] if agent.memory else None
            } for agent in self.agents],
            'environment_states': {name: env.get_global_state() for name, env in self.environments.items()},
            'tracker_summary': {name: tracker.get_summary() for name, tracker in self.trackers.items()}
        }
        return summary

    async def run(self):
        for round_num in range(1, self.config.max_rounds + 1):
            log_round(self.logger, round_num)
            await self.run_environment(round_num)
            await self.process_round_results(round_num)
        # Print simulation summary after all rounds
        self.print_summary()

    async def process_round_results(self, round_num: int):
        # Save round data to the database
        try:
            self.data_inserter.insert_round_data(
                round_num,
                self.agents,
                self.environments,
                self.config,
                self.trackers
            )
            self.logger.info(f"Data for round {round_num} inserted successfully.")
        except Exception as e:
            self.logger.error(f"Error inserting data for round {round_num}: {str(e)}")

    def print_summary(self):
        log_section(self.logger, "GROUP CHAT SIMULATION SUMMARY")
        total_messages = sum(tracker.get_summary()['total_messages'] for tracker in self.trackers.values())
        total_topics = sum(tracker.get_summary()['total_topics'] for tracker in self.trackers.values())
        print(f"Total messages: {total_messages}")
        print(f"Total topics discussed: {total_topics}")
        print("\nFinal Agent States:")
        for agent in self.agents:
            print(f"Agent {agent.index} ({agent.role}):")
            if agent.memory:
                print(f"  Last Reflection: {agent.memory[-1]['content']}")
            print()