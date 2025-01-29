# groupchat_orchestrator.py

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import uuid

from market_agents.environments.environment import MultiAgentEnvironment
from market_agents.agents.market_agent import MarketAgent
from market_agents.environments.mechanisms.group_chat import GroupChat, GroupChatActionSpace, GroupChatObservationSpace
from market_agents.orchestrators.config import GroupChatConfig, OrchestratorConfig
from market_agents.orchestrators.logger_utils import (
    log_perception, log_persona, log_section, log_round,
    log_cohort_formation, log_topic_proposal, log_sub_round_start,
    log_group_message
)
from market_agents.memory.agent_storage.storage_service import StorageService
from market_agents.orchestrators.orchestration_data_inserter import OrchestrationDataInserter
from market_agents.orchestrators.group_chat.groupchat_api_utils import GroupChatAPIUtils
from market_agents.orchestrators.agent_cognitive import AgentCognitiveProcessor

class GroupChatOrchestrator:
    def __init__(
        self,
        config: GroupChatConfig,
        orchestrator_config: OrchestratorConfig,
        agents: List[MarketAgent],
        ai_utils,
        storage_service: StorageService,
        logger=None,
        **kwargs
    ):
        self.config = config
        self.orchestrator_config = orchestrator_config
        self.agents = agents
        self.ai_utils = ai_utils
        self.logger = logger or logging.getLogger(__name__)

        # Initialize storage components
        self.storage_service = storage_service
        self.data_inserter = OrchestrationDataInserter(storage_service=storage_service)
        
        # Initialize API utils with the correct URL from config
        self.api_utils = GroupChatAPIUtils(self.config.api_url, self.logger)

        # Initialize cognitive processor with storage service
        self.cognitive_processor = AgentCognitiveProcessor(
            ai_utils=ai_utils,
            storage_service=storage_service,
            logger=self.logger,
            tool_mode=self.orchestrator_config.tool_mode
        )

        # Agent dictionary for quick lookup
        self.agent_dict = {agent.id: agent for agent in agents}

        # Cohorts: cohort_id -> List[MarketAgent]
        self.cohorts: Dict[str, List[MarketAgent]] = {}

        # Topic proposers: cohort_id -> proposer_id
        self.topic_proposers: Dict[str, str] = {}

        # Round summaries
        self.round_summaries: List[Dict[str, Any]] = []

        # Sub-rounds per round
        self.sub_rounds_per_round = config.sub_rounds

        self.logger.info(f"Initialized GroupChatOrchestrator")

    async def setup_environment(self):
        """
        Sets up the environment by checking API health, registering agents,
        forming cohorts, and assigning agents to cohorts.
        """
        log_section(self.logger, "CONFIGURING GROUP CHAT ENVIRONMENT")

        # Check API health
        if not await self.api_utils.check_api_health():
            raise RuntimeError("GroupChat API is not available")

        # Register agents
        await self.api_utils.register_agents(self.agents)

        # Form cohorts
        agent_ids = [agent.id for agent in self.agents]
        cohorts_info = await self.api_utils.form_cohorts(agent_ids, self.config.group_size)

        # Create environments and assign cohorts
        for cohort in cohorts_info:
            cohort_id = cohort["cohort_id"]
            cohort_agent_ids = cohort["agent_ids"]
            cohort_agents = [self.agent_dict[agent_id] for agent_id in cohort_agent_ids]
            self.cohorts[cohort_id] = cohort_agents

            # Create environment for this cohort
            group_chat = GroupChat(
                max_rounds=self.config.max_rounds,
                sequential=False,
            )
            group_chat_env = MultiAgentEnvironment(
                name=f"group_chat_{cohort_id}",
                address=f"group_chat_{cohort_id}",
                max_steps=self.config.max_rounds,
                action_space=GroupChatActionSpace(),
                observation_space=GroupChatObservationSpace(),
                mechanism=group_chat
            )

            # Assign environment and cohort_id to agents
            for agent in cohort_agents:
                agent.cohort_id = cohort_id
                if not hasattr(agent, 'environments') or agent.environments is None:
                    agent.environments = {}
                agent.environments['group_chat'] = group_chat_env

            log_cohort_formation(self.logger, cohort_id, [agent.index for agent in cohort_agents])

        self.logger.info("Environment setup complete.")
        
        # Verify environments are properly set
        for agent in self.agents:
            if 'group_chat' not in agent.environments:
                self.logger.error(f"Agent {agent.index} missing group_chat environment!")
            else:
                self.logger.info(f"Agent {agent.index} environments: {list(agent.environments.keys())}")

    async def run_environment(self, round_num: int = None):
        """
        Runs the environment for the configured number of rounds.
        
        Args:
            round_num (int, optional): If provided, runs a specific round.
                                        If None, runs all rounds.
        """
        if round_num is not None:
            # Run specific round
            await self.run_round(round_num)
        else:
            # Run all rounds
            for round_num in range(1, self.config.max_rounds + 1):
                await self.run_round(round_num)

    async def run_round(self, round_num: int):
        """
        Runs a single round of the simulation.

        Args:
            round_num (int): The current round number.
        """
        log_round(self.logger, round_num)

        # Select topic proposers
        await self.select_topic_proposers()

        # Collect proposed topics
        await self.collect_proposed_topics(round_num)

        # Run sub-rounds
        for sub_round in range(1, self.sub_rounds_per_round + 1):
            log_sub_round_start(self.logger, 'All Cohorts', sub_round)
            # For each cohort, run the sub-round
            tasks = []
            for cohort_id, cohort_agents in self.cohorts.items():
                task = asyncio.create_task(
                    self.run_group_chat_sub_round(
                        cohort_id=cohort_id,
                        round_num=round_num,
                        sub_round_num=sub_round,
                        cohort_agents=cohort_agents
                    )
                )
                tasks.append(task)
            await asyncio.gather(*tasks)

        # Run reflection
        log_section(self.logger, "AGENT REFLECTIONS")
        await self.cognitive_processor.run_parallel_reflect(self.agents, self.config.name)

        await self.process_round_results(round_num)

        # Store round summary
        round_summary = await self.get_round_summary(round_num)
        self.round_summaries.append(round_summary)

    async def select_topic_proposers(self):
        """
        Selects topic proposers for each cohort using the API.
        """
        tasks = []
        for cohort_id, cohort_agents in self.cohorts.items():
            agent_ids = [agent.id for agent in cohort_agents]
            task = asyncio.create_task(
                self.api_utils.select_proposer(cohort_id, agent_ids)
            )
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        for cohort_id, proposer_id in zip(self.cohorts.keys(), results):
            if proposer_id:
                self.topic_proposers[cohort_id] = proposer_id
                self.logger.info(f"Selected proposer {proposer_id} for cohort {cohort_id}")
            else:
                self.logger.error(f"Failed to select proposer for cohort {cohort_id}")

    async def collect_proposed_topics(self, round_num: int):
        """
        Collects proposed topics from proposers and submits them via the API.

        Args:
            round_num (int): The current round number.
        """
        proposer_agents = []
        proposer_prompts = []

        # Collect prompts for proposers
        for cohort_id, proposer_id in self.topic_proposers.items():
            proposer_agent = self.agent_dict[proposer_id]
            # Set system message for proposer
            initial_topic = self.config.initial_topic
            proposer_agent_task = f"You are the group chat topic proposer agent. Your role is to propose interesting and relevant topics for group discussion about {initial_topic}.\n"
            #proposer_agent_task += f"Consider recent events, trends, or news related to {good_name}.\n" 
            proposer_agent_task += "Propose a specific topic for discussion that would be relevant to participants. Please describe the topic in detail."
            prompt = await proposer_agent.generate_action(
                self.config.name,
                proposer_agent_task,
                return_prompt=True,
                structured_tool=self.orchestrator_config.tool_mode
            )
            proposer_agents.append((cohort_id, proposer_agent))
            proposer_prompts.append(prompt)

        # Run prompts in parallel
        proposals = await self.ai_utils.run_parallel_ai_completion(proposer_prompts, update_history=False)
        self.storage_service.store_ai_requests(self.ai_utils.get_all_requests())

        tasks = []
        for (cohort_id, proposer_agent), proposal in zip(proposer_agents, proposals):
            topic = self.extract_topic_from_proposal(proposal)
            if topic:
                task = asyncio.create_task(
                    self.api_utils.propose_topic(
                        agent_id=proposer_agent.id,
                        cohort_id=cohort_id,
                        topic=topic,
                        round_num=round_num
                    )
                )
                tasks.append(task)
                log_topic_proposal(self.logger, cohort_id, proposer_agent.index, topic)
            else:
                self.logger.error(f"Failed to extract topic from proposer {proposer_agent.id} in cohort {cohort_id}")
        await asyncio.gather(*tasks)

    def extract_topic_from_proposal(self, proposal) -> Optional[str]:
        """
        Extracts the topic from the proposal.

        Args:
            proposal: The proposal response.

        Returns:
            Optional[str]: The extracted topic.
        """
        try:
            if proposal.json_object:
                action_content = proposal.json_object.object
                if 'action' in action_content and 'content' in action_content['action']:
                    topic = action_content['action']['content']
                else:
                    topic = None
            else:
                topic = proposal.str_content.strip() if proposal.str_content else None
            return topic
        except Exception as e:
            self.logger.error(f"Error extracting topic: {e}")
            return None

    async def run_group_chat_sub_round(
        self,
        cohort_id: str,
        round_num: int,
        sub_round_num: int,
        cohort_agents: List[MarketAgent]
    ):
        """
        Runs a single sub-round for a cohort.

        Args:
            cohort_id (str): The cohort ID.
            round_num (int): The current round number.
            sub_round_num (int): The current sub-round number.
            cohort_agents (List[MarketAgent]): The agents in the cohort.
        """
        # First try block for cognitive processes
        try:
            # Get topic and messages from API
            topic = await self.api_utils.get_topic(cohort_id)
            messages = await self.api_utils.get_messages(cohort_id)

            if not topic:
                self.logger.warning(f"No topic found for cohort {cohort_id}")
                return

            # Get the cohort's environment mechanism and update topic
            environment = cohort_agents[0].environments['group_chat']
            environment.mechanism._update_topic(topic, round_num)

            for agent in cohort_agents:
                # Filter messages to only include this agent's messages
                agent_messages = [msg for msg in messages if msg.get('agent_id') == agent.id]
                agent.last_observation = {
                    'messages': agent_messages[-1] if agent_messages else None
                }

            # Agents perceive the messages
            perceptions = await self.cognitive_processor.run_parallel_perceive(cohort_agents, self.config.name)
            # Log personas and perceptions
            for agent, perception in zip(cohort_agents, perceptions):
                log_persona(self.logger, agent.index, agent.persona)
                log_perception(
                    self.logger, 
                    agent.index, 
                    perception.json_object.object if perception and perception.json_object else None
                )
                agent.last_perception = perception.json_object.object if perception.json_object else perception.str_content

            # Agents generate actions (messages)
            actions = await self.cognitive_processor.run_parallel_action(cohort_agents, self.config.name)

        except Exception:
            return

        # Second try block for data insertion
        try:
            # Prepare messages for both API posting and database insertion
            messages_to_insert = []
            api_tasks = []

            for agent, action in zip(cohort_agents, actions):
                content = self.extract_message_content(action)
                if content:
                    # Prepare API task
                    api_task = asyncio.create_task(
                        self.api_utils.post_message(
                            agent_id=agent.id,
                            cohort_id=cohort_id,
                            content=content,
                            round_num=round_num,
                            sub_round_num=sub_round_num
                        )
                    )
                    api_tasks.append(api_task)

                    messages_to_insert.append({
                        'agent_id': agent.id,
                        'environment_name': self.config.name,
                        'round': round_num,
                        'sub_round': sub_round_num,
                        'cohort_id': cohort_id,
                        'content': content,
                        'timestamp': datetime.now(timezone.utc),
                        'topic': topic,
                        'message_id': str(uuid.uuid4()),
                        'type': 'group_chat_message'
                    })

                    # Update agent state and log
                    agent.last_action = content
                    log_group_message(self.logger, cohort_id, agent.index, content, sub_round_num)
                else:
                    self.logger.warning(f"Failed to extract message content for agent {agent.id}")

            # Execute API posts in parallel
            await asyncio.gather(*api_tasks)
            # Insert messages into database
            if messages_to_insert:
                await self.data_inserter.insert_actions(
                    actions_data=messages_to_insert,
                    agent_id_map={str(agent.id): agent.id for agent in cohort_agents}
                )

        except Exception as e:
            self.logger.warning(f"Error during data insertion in sub-round {sub_round_num} for cohort {cohort_id}: {e}")

    def extract_message_content(self, action) -> Optional[str]:
        """
        Extracts the message content from the action.

        Args:
            action: The action response.

        Returns:
            Optional[str]: The extracted message content.
        """
        try:
            if action.json_object:
                action_content = action.json_object.object
                if 'action' in action_content and 'content' in action_content['action']:
                    content = action_content['action']['content']
                else:
                    content = None
            else:
                content = action.str_content.strip() if action.str_content else None
            return content
        except Exception as e:
            self.logger.error(f"Error extracting message content: {e}")
            return None

    async def process_round_results(self, round_num: int):
        """Process and store the results of a round in the database."""
        try:
            # For each cohort, process its environment
            for cohort_id, cohort_agents in self.cohorts.items():
                # Get the cohort-specific environment name
                cohort_env_name = f"group_chat_{cohort_id}"
                environment = cohort_agents[0].environments['group_chat']
                
                # Insert round data using data inserter
                await self.data_inserter.insert_round_data(
                    round_num=round_num,
                    agents=cohort_agents,
                    environment=environment,
                    config=self.orchestrator_config,
                    environment_name=cohort_env_name
                )
                self.logger.info(f"Data for round {round_num}, cohort {cohort_id} inserted successfully.")

            # Store round summary
            round_summary = await self.get_round_summary(round_num)
            self.round_summaries.append(round_summary)

        except Exception as e:
            self.logger.error(f"Error processing round {round_num} results: {str(e)}")
            self.logger.exception("Exception details:")
            raise e

    async def get_round_summary(self, round_num: int) -> Dict[str, Any]:
        """Return a summary of the round"""
        summary = {
            'round': round_num,
            'agent_states': [{
                'id': agent.id,
                'index': agent.index,
                'last_action': agent.last_action,
                'last_observation': agent.last_observation,
                # Get the most recent memory from short-term memory
                'memory': (await agent.short_term_memory.retrieve_recent_memories(limit=1))[0] if agent.short_term_memory else None
            } for agent in self.agents],
            'cohorts': {cohort_id: [agent.id for agent in agents] 
                    for cohort_id, agents in self.cohorts.items()},
            'topics': self.topic_proposers,
        }
        return summary

    async def print_summary(self):
        """Print a summary of the simulation results"""
        log_section(self.logger, "GROUP CHAT SIMULATION SUMMARY")
        
        print("\nFinal Agent States:")
        for agent in self.agents:
            print(f"Agent {agent.index}")
            print(f"  Last action: {agent.last_action}")
            # Get the most recent reflection from short-term memory
            recent_reflections = await agent.short_term_memory.retrieve_recent_memories(cognitive_step='reflection', limit=1)
            if recent_reflections:
                print(f"  Last reflection: {recent_reflections[0].content}")
            print()
        
        # Print round summaries
        for summary in self.round_summaries:
            print(f"Round {summary['round']} summary:")
            for agent_state in summary['agent_states']:
                print(f"  Agent {agent_state['index']} last action: {agent_state['last_action']}")
            print()

