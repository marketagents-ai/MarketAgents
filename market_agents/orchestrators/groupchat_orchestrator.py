# groupchat_orchestrator.py

import asyncio
import logging
import random
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import Field

from market_agents.orchestrators.base_orchestrator import BaseEnvironmentOrchestrator
from market_agents.agents.market_agent import MarketAgent
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
from market_agents.orchestrators.config import GroupChatConfig
from market_agents.orchestrators.logger_utils import (
    log_section,
    log_environment_setup,
    log_persona,
    log_running,
    log_perception,
    log_action,
    log_reflection,
    log_round,
    log_completion,
    print_ascii_art
)
from market_agents.orchestrators.insert_simulation_data import SimulationDataInserter

class CohortManager:
    def __init__(self, agents: List[MarketAgent], cohort_size: int):
        self.agents = agents
        self.cohort_size = cohort_size
        self.cohorts: Dict[str, List[MarketAgent]] = {}
        self.topic_proposers: Dict[str, MarketAgent] = {}
        self.form_cohorts()

    def form_cohorts(self):
        # Randomly assign agents to cohorts once at initialization
        random.shuffle(self.agents)
        self.cohorts = {}
        for i in range(0, len(self.agents), self.cohort_size):
            cohort_agents = self.agents[i:i + self.cohort_size]
            cohort_id = f"cohort_{i // self.cohort_size}"
            self.cohorts[cohort_id] = cohort_agents

    def select_topic_proposers(self):
        # Rotate proposers within each cohort
        for cohort_id, agents in self.cohorts.items():
            if cohort_id in self.topic_proposers:
                current_proposer = self.topic_proposers[cohort_id]
                current_index = agents.index(current_proposer)
                next_index = (current_index + 1) % len(agents)
                self.topic_proposers[cohort_id] = agents[next_index]
            else:
                self.topic_proposers[cohort_id] = random.choice(agents)

class GroupChatTracker:
    def __init__(self):
        self.messages: List[GroupChatMessage] = []
        self.topics: Dict[str, str] = {}  # cohort_id -> topic

    def add_message(self, message: GroupChatMessage):
        self.messages.append(message)

    def add_topic(self, cohort_id: str, topic: str):
        self.topics[cohort_id] = topic

    def get_summary(self):
        return {
            "total_messages": len(self.messages),
            "total_topics": len(self.topics)
        }

class GroupChatOrchestrator(BaseEnvironmentOrchestrator):
    environment_name: str = Field(default='group_chat')
    environments: Dict[str, MultiAgentEnvironment] = Field(default_factory=dict)
    trackers: Dict[str, GroupChatTracker] = Field(default_factory=dict)
    cohort_manager: CohortManager = Field(default=None)
    sub_rounds_per_step: int = Field(default=2)
    agent_dict: Dict[str, MarketAgent] = Field(default_factory=dict)
    topics: Dict[str, str] = Field(default_factory=dict)  # cohort_id -> topic

    def __init__(
        self,
        config: GroupChatConfig,
        agents: List[MarketAgent],
        ai_utils,
        data_inserter: SimulationDataInserter,
        logger=None
    ):
        super().__init__(
            config=config,
            agents=agents,
            ai_utils=ai_utils,
            data_inserter=data_inserter,
            logger=logger,
        )

        
        self.cohort_manager = CohortManager(self.agents, self.config.group_size)
        self.sub_rounds_per_step = self.config.sub_rounds
        self.agent_dict = {agent.id: agent for agent in agents}
        self.setup_environment()

    def setup_environment(self):
        log_section(self.logger, "CONFIGURING GROUP CHAT ENVIRONMENTS")
        # Create GroupChat Environments per Cohort
        for cohort_id, cohort_agents in self.cohort_manager.cohorts.items():
            group_chat = GroupChat(
                max_rounds=self.config.max_rounds,
                sequential=False,
            )
            group_chat_env = MultiAgentEnvironment(
                name=f"{self.config.name}_{cohort_id}",
                address=f"{self.config.address}_{cohort_id}",
                max_steps=self.config.max_rounds,
                action_space=GroupChatActionSpace(),
                observation_space=GroupChatObservationSpace(),
                mechanism=group_chat
            )
            self.environments[cohort_id] = group_chat_env
            self.trackers[cohort_id] = GroupChatTracker()
            log_environment_setup(self.logger, cohort_id)
        # Assign environments to agents
        for agent in self.agents:
            agent.environments[self.environment_name] = self.environments[cohort_id]

    async def run_environment(self, round_num: int):
        # Keep cohorts the same across rounds; do not rotate cohorts
        # Rotate topic proposers within cohorts
        self.cohort_manager.select_topic_proposers()
        # Collect topics from proposers
        await self.collect_proposed_topics()
        # Run multiple sub-rounds within group chats
        for sub_round in range(1, self.sub_rounds_per_step + 1):
            group_chat_tasks = []
            for cohort_id, cohort_agents in self.cohort_manager.cohorts.items():
                task = asyncio.create_task(
                    self.run_group_chat_sub_round(
                        cohort_id=cohort_id,
                        round_num=round_num,
                        sub_round_num=sub_round,
                        cohort_agents=cohort_agents
                    )
                )
                group_chat_tasks.append(task)
            await asyncio.gather(*group_chat_tasks)
        # After all sub-rounds are completed
        # Run reflection step
        log_section(self.logger, "AGENT REFLECTIONS")
        await self.run_reflection(round_num)

    async def collect_proposed_topics(self):
        # Generate prompts for topic proposers
        proposer_prompts = []
        proposer_agents = []
        for cohort_id, proposer in self.cohort_manager.topic_proposers.items():
            self.set_proposer_system_message(proposer, cohort_id)
            perceive_prompt = await proposer.perceive(self.environment_name, return_prompt=True)
            proposer_prompts.append(perceive_prompt)
            proposer_agents.append(proposer)
        
        # Run AI completions
        proposals = await self.ai_utils.run_parallel_ai_completion(proposer_prompts, update_history=False)
        self.data_inserter.insert_ai_requests(self.ai_utils.get_all_requests())
        
        # Assign topics to cohorts
        for proposal, proposer in zip(proposals, proposer_agents):
            try:
                cohort_id = next(cid for cid, agent in self.cohort_manager.topic_proposers.items() if agent.id == proposer.id)
                
                # Extract topic from proposal using the same logic as the example
                if proposal.json_object:
                    action_content = proposal.json_object.object
                    if isinstance(action_content, dict):
                        if 'content' in action_content:
                            if isinstance(action_content['content'], dict) and 'action' in action_content['content']:
                                topic = action_content['content']['action']['content']
                            else:
                                topic = action_content['content']
                        elif 'action' in action_content:
                            topic = action_content['action']['content']
                        else:
                            topic = "Default topic: Recent market trends"
                else:
                    # If no JSON object, try to use str_content
                    topic = proposal.str_content.strip() if proposal.str_content else "Default topic: Recent market trends"
                
                self.topics[cohort_id] = topic
                self.trackers[cohort_id].add_topic(cohort_id, topic)
                self.logger.info(f"Cohort {cohort_id} topic proposed by Agent {proposer.index}: {topic}")
                
            except Exception as e:
                self.logger.error(f"Error processing topic proposal for cohort {cohort_id}: {str(e)}")
                default_topic = "Default topic: Recent market trends"
                self.topics[cohort_id] = default_topic
                self.trackers[cohort_id].add_topic(cohort_id, default_topic)
                self.logger.warning(f"Using default topic for cohort {cohort_id}: {default_topic}")

    async def run_group_chat_sub_round(self, cohort_id: str, round_num: int, sub_round_num: int, cohort_agents: List[MarketAgent]):
        env = self.environments[cohort_id]
        tracker = self.trackers[cohort_id]
        topic = self.topics.get(cohort_id, "No Topic")
        log_running(self.logger, f"{cohort_id} - Sub-round {sub_round_num}")
        # Set system messages for agents
        self.set_agent_system_messages(cohort_id, topic, round_num, sub_round_num=sub_round_num)
        # Run agents' perception in parallel
        perception_prompts = await self.run_parallel_perceive(cohort_agents, cohort_id)
        perceptions = await self.ai_utils.run_parallel_ai_completion(perception_prompts, update_history=False)
        self.data_inserter.insert_ai_requests(self.ai_utils.get_all_requests())
        # Map perceptions to agents
        perceptions_map = {perception.source_id: perception for perception in perceptions}

        for agent in cohort_agents:
            perception = perceptions_map.get(agent.id)
            if perception:
                log_persona(self.logger, agent.index, agent.persona)
                log_perception(self.logger, agent.index, f"{perception.str_content}")
                agent.last_perception = perception.str_content
            else:
                self.logger.warning(f"No perception found for agent {agent.index}")
                agent.last_perception = ""

        # Extract perception contents for action generation
        perception_contents = [agent.last_perception for agent in cohort_agents]

        # Run agents' action generation in parallel
        action_prompts = await self.run_parallel_generate_action(cohort_agents, perception_contents)
        actions = await self.ai_utils.run_parallel_ai_completion(action_prompts, update_history=False)
        self.data_inserter.insert_ai_requests(self.ai_utils.get_all_requests())
        actions_map = {action.source_id: action for action in actions}

        # Collect actions from agents
        agent_actions = {}
        for agent in cohort_agents:
            action = actions_map.get(agent.id)
            if action:
                try:
                    action_content = action.json_object.object if action.json_object else json.loads(action.str_content or '{}')
                    agent.last_action = action_content
                    if 'action' in action_content and 'content' in action_content['action']:
                        group_chat_message = GroupChatMessage(
                            content=action_content['action']['content'],
                            message_type=action_content['action'].get('message_type', 'group_message'),
                            agent_id=agent.id,
                            cohort_id=cohort_id,
                            sub_round=sub_round_num
                        )
                        group_chat_action = GroupChatAction(agent_id=agent.id, action=group_chat_message)
                        agent_actions[agent.id] = group_chat_action.model_dump()
                        log_action(self.logger, agent.index, f"Message: {group_chat_message.content}")
                    else:
                        raise ValueError(f"Invalid action content: {action_content}")
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    self.logger.error(f"Error creating GroupChatAction for agent {agent.index}: {str(e)}")
            else:
                self.logger.warning(f"No action found for agent {agent.index}")

        # Create global action and step the environment
        global_action = GroupChatGlobalAction(actions=agent_actions)
        try:
            env_state = env.step(global_action)
        except Exception as e:
            self.logger.error(f"Error in environment {self.environment_name}: {str(e)}")
            raise e

        self.logger.info(f"Completed {self.environment_name} step")

        # Process the environment state
        if isinstance(env_state.global_observation, GroupChatGlobalObservation):
            self.process_environment_state(env_state, cohort_agents, cohort_id)

        # Store the last environment state
        self.last_env_state = env_state

    async def run_parallel_perceive(self, cohort_agents: List[MarketAgent], cohort_id: str) -> List[Any]:
        perceive_prompts = []
        for agent in cohort_agents:
            perceive_prompt = await agent.perceive(self.environment_name, return_prompt=True)
            perceive_prompts.append(perceive_prompt)
        return perceive_prompts

    async def run_parallel_generate_action(self, cohort_agents: List[MarketAgent], perceptions: List[str]) -> List[Any]:
        action_prompts = []
        for agent, perception in zip(cohort_agents, perceptions):
            action_prompt = await agent.generate_action(self.environment_name, perception, return_prompt=True)
            action_prompts.append(action_prompt)
        return action_prompts

    async def run_reflection(self, round_num: int):
        # Run reflection for each cohort
        for cohort_id, cohort_agents in self.cohort_manager.cohorts.items():
            await self.run_reflection_for_cohort(cohort_id, cohort_agents)

    async def run_reflection_for_cohort(self, cohort_id: str, cohort_agents: List[MarketAgent]):
        reflect_prompts, agents_with_observations = await self.run_parallel_reflect(cohort_agents)
        if reflect_prompts:
            reflections = await self.ai_utils.run_parallel_ai_completion(reflect_prompts, update_history=False)
            self.data_inserter.insert_ai_requests(self.ai_utils.get_all_requests())
            for agent, reflection in zip(agents_with_observations, reflections):
                if reflection.json_object:
                    log_reflection(self.logger, agent.index, f"{reflection.json_object.object}")
                    # Append reflection to agent memory
                    agent.memory.append({
                        "type": "reflection",
                        "content": reflection.json_object.object.get("reflection", ""),
                        "strategy_update": reflection.json_object.object.get("strategy_update", ""),
                        "observation": agent.last_observation,
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    self.logger.warning(f"No reflection JSON object for agent {agent.index}")
        else:
            self.logger.info(f"No reflections generated for cohort {cohort_id} in this round.")

    async def run_parallel_reflect(self, cohort_agents: List[MarketAgent]) -> List[Any]:
        reflect_prompts = []
        agents_with_observations = []
        for agent in cohort_agents:
            if agent.last_observation:
                reflect_prompt = await agent.reflect(self.environment_name, return_prompt=True)
                reflect_prompts.append(reflect_prompt)
                agents_with_observations.append(agent)
            else:
                self.logger.info(f"Skipping reflection for agent {agent.index} due to no observation")
        return reflect_prompts, agents_with_observations

    def set_proposer_system_message(self, proposer: MarketAgent, cohort_id: str):
        # Set system message for the topic proposer
        proposer.system = (
            f"You are Agent {proposer.index}, selected as the topic proposer for your cohort. "
            f"Please propose an interesting topic for discussion related to the meme coin market. "
            f"Respond with a single sentence stating the topic."
        )

    def set_agent_system_messages(self, cohort_id: str, topic: str, round_num: int, sub_round_num: int = None):
        # Set system messages for agents in the cohort
        cohort_agents = self.cohort_manager.cohorts[cohort_id]
        proposer = self.cohort_manager.topic_proposers[cohort_id]
        for agent in cohort_agents:
            if agent.id == proposer.id and sub_round_num == 1:
                # Topic proposer has a different role in sub-round 1
                agent.system = (
                    f"You are Agent {agent.index}, selected as the topic proposer for your cohort in round {round_num}. "
                    f"The topic you proposed is '{topic}'. In this sub-round, initiate the discussion on this topic."
                )
            else:
                agent.system = (
                    f"You are Agent {agent.index} participating in sub-round {sub_round_num} of round {round_num} "
                    f"in a group chat about '{topic}'. Engage in the discussion with your cohort members."
                )

    def process_environment_state(self, env_state: EnvironmentStep, cohort_agents: List[MarketAgent], cohort_id: str):
        # Process messages and update agent observations
        global_observation = env_state.global_observation
        tracker = self.trackers[cohort_id]
        for message in global_observation.all_messages:
            if message.cohort_id == cohort_id:
                tracker.add_message(message)
        # Update agent observations
        for agent in cohort_agents:
            agent_observation = global_observation.observations.get(agent.id)
            if agent_observation:
                agent.last_observation = agent_observation
                agent.last_step = env_state
            else:
                agent.last_perception = ""

    def get_round_summary(self, round_num: int) -> dict:
        # Return a summary of the round
        summary = {
            'round': round_num,
            'agent_states': [{
                'id': agent.id,
                'role': agent.persona.role,
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

        group_chat_summaries = [tracker.get_summary() for tracker in self.trackers.values()]
        print("\nGroup Chat Summary:")
        for summary in group_chat_summaries:
            print(f"Total messages: {summary['total_messages']}")
            print(f"Total topics discussed: {summary['total_topics']}")

        print("\nFinal Agent States:")
        for agent in self.agents:
            print(f"Agent {agent.index}")
            print(f"  Last action: {agent.last_action}")
            if agent.memory:
                print(f"  Last reflection: {agent.memory[-1]['content']}")
            print()
