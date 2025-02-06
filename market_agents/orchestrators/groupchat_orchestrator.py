import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import uuid

from market_agents.agents.cognitive_steps import ActionStep
from market_agents.environments.environment import MultiAgentEnvironment
from market_agents.agents.market_agent import MarketAgent
from market_agents.environments.mechanisms.group_chat import (
    GroupChat,
    GroupChatAction,
    GroupChatActionSpace,
    GroupChatGlobalAction,
    GroupChatMessage,
    GroupChatObservationSpace
)
from market_agents.orchestrators.config import GroupChatConfig, OrchestratorConfig
from market_agents.orchestrators.logger_utils import (
    log_perception,
    log_persona,
    log_reflection,
    log_section,
    log_round,
    log_cohort_formation,
    log_topic_proposal,
    log_sub_round_start,
    log_group_message
)
from market_agents.memory.agent_storage.storage_service import StorageService
from market_agents.orchestrators.orchestration_data_inserter import OrchestrationDataInserter
from market_agents.orchestrators.group_chat.groupchat_api_utils import GroupChatAPIUtils
from market_agents.orchestrators.parallel_cognitive_steps import ParallelCognitiveProcessor


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

        self.storage_service = storage_service
        self.data_inserter = OrchestrationDataInserter(storage_service=storage_service)
        
        self.api_utils = GroupChatAPIUtils(self.config.api_url, self.logger)

        self.cognitive_processor = ParallelCognitiveProcessor(
            ai_utils=ai_utils,
            storage_service=storage_service,
            logger=self.logger,
            tool_mode=self.orchestrator_config.tool_mode
        )

        self.agent_dict = {agent.id: agent for agent in agents}
        self.cohorts: Dict[str, List[MarketAgent]] = {}
        self.agent_cohort_map: Dict[str, str] = {}
        self.topic_proposers: Dict[str, str] = {}
        self.round_summaries: List[Dict[str, Any]] = []
        self.sub_rounds_per_round = config.sub_rounds
        self.logger.info("Initialized GroupChatOrchestrator")

    async def setup_environment(self):
        """
        Sets up the environment by checking API health, registering agents,
        forming cohorts, and assigning agents to cohorts.
        """
        log_section(self.logger, "CONFIGURING GROUP CHAT ENVIRONMENT")

        if not await self.api_utils.check_api_health():
            raise RuntimeError("GroupChat API is not available")

        await self.api_utils.register_agents(self.agents)

        agent_ids = [agent.id for agent in self.agents]
        cohorts_info = await self.api_utils.form_cohorts(agent_ids, self.config.group_size)

        for cohort in cohorts_info:
            cohort_id = cohort["cohort_id"]
            cohort_agent_ids = cohort["agent_ids"]
            cohort_agents = [self.agent_dict[a_id] for a_id in cohort_agent_ids]
            self.cohorts[cohort_id] = cohort_agents

            for a in cohort_agents:
                self.agent_cohort_map[a.id] = cohort_id

            group_chat = GroupChat(
                max_rounds=self.config.sub_rounds,
                sequential=False,
            )
            group_chat_env = MultiAgentEnvironment(
                name=f"group_chat_{cohort_id}",
                address=f"group_chat_{cohort_id}",
                max_steps=self.config.sub_rounds,
                action_space=GroupChatActionSpace(),
                observation_space=GroupChatObservationSpace(),
                mechanism=group_chat
            )

            for agent in cohort_agents:
                if not hasattr(agent, "environments") or agent.environments is None:
                    agent.environments = {}
                agent.environments["group_chat"] = group_chat_env

            log_cohort_formation(
                self.logger,
                cohort_id,
                [getattr(agent, "index", agent.id) for agent in cohort_agents]
            )

        self.logger.info("Environment setup complete.")
        
        for agent in self.agents:
            if "group_chat" not in agent.environments:
                self.logger.error(
                    f"Agent {getattr(agent, 'index', agent.id)} "
                    f"(ID: {agent.id}) missing group_chat environment!"
                )
            else:
                self.logger.info(
                    f"Agent {getattr(agent, 'index', agent.id)} "
                    f"(ID: {agent.id}) has environments: {list(agent.environments.keys())}"
                )

    async def run_environment(self, round_num: Optional[int] = None):
        """
        Runs the environment for either a specific round or all rounds.
        
        Args:
            round_num (Optional[int]): If provided, runs that specific round.
                                       Otherwise, runs all from 1..max_rounds.
        """
        if round_num is not None:
            await self.run_round(round_num)
        else:
            for rn in range(1, self.orchestrator_config.max_rounds + 1):
                await self.run_round(rn)

    async def run_round(self, round_num: int):
        """
        Runs a single round of the simulation.

        Args:
            round_num (int): The current round number.
        """
        log_round(self.logger, round_num, self.config.name)

        await self.select_topic_proposers()

        await self.collect_proposed_topics(round_num)

        for sub_round in range(1, self.sub_rounds_per_round + 1):
            log_sub_round_start(self.logger, "All Cohorts", sub_round)
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

        await self.process_round_results(round_num)

        round_summary = await self.get_round_summary(round_num)
        self.round_summaries.append(round_summary)

    async def select_topic_proposers(self):
        """Selects topic proposers for each cohort via the GroupChat API."""
        tasks = []
        for cohort_id, cohort_agents in self.cohorts.items():
            agent_ids = [a.id for a in cohort_agents]
            tasks.append(self.api_utils.select_proposer(cohort_id, agent_ids))
        results = await asyncio.gather(*tasks)
        for c_id, proposer_id in zip(self.cohorts.keys(), results):
            if proposer_id:
                self.topic_proposers[c_id] = proposer_id
                self.logger.info(f"Selected proposer {proposer_id} for cohort {c_id}")
            else:
                self.logger.error(f"Failed to select proposer for cohort {c_id}")

    async def collect_proposed_topics(self, round_num: int):
        """
        Collects proposed topics from each cohort's proposer and submits them
        using an ActionStep, where the agent.task is dynamically set to a topic-
        generation prompt.
        """
        proposer_agents = []
        
        for cohort_id, proposer_id in self.topic_proposers.items():
            proposer_agent = self.agent_dict[proposer_id]

            initial_topic = self.config.initial_topic or "something interesting"
            proposer_agent_task = (
                f"You are the group chat topic proposer. "
                f"Propose a topic related to {initial_topic}, explaining why it matters."
            )

            action_step = ActionStep(
                step_name="action",
                agent_id=proposer_agent.id,
                environment_name=self.config.name,
                environment_info={},
                structured_tool=self.orchestrator_config.tool_mode,
                return_prompt=True
            )

            step_prompt = await proposer_agent.run_step(step=action_step)
            step_prompt.new_message += proposer_agent_task

            proposer_agents.append((cohort_id, proposer_agent, step_prompt))

        prompts = [p for (_, _, p) in proposer_agents]
        proposals = await self.ai_utils.run_parallel_ai_completion(prompts)
        await self.storage_service.store_ai_requests(self.cognitive_processor.get_all_requests())

        tasks = []
        for (cohort_id, agent, _), proposal in zip(proposer_agents, proposals):
            topic = self.extract_topic_from_proposal(proposal)
            if topic:
                tasks.append(
                    self.api_utils.propose_topic(
                        agent_id=agent.id,
                        cohort_id=cohort_id,
                        topic=topic,
                        round_num=round_num
                    )
                )
                log_topic_proposal(
                    self.logger,
                    cohort_id,
                    getattr(agent, "index", agent.id),
                    topic
                )
            else:
                self.logger.error(
                    f"Could not extract topic for cohort={cohort_id}, proposer={agent.id}"
                )

        await asyncio.gather(*tasks)

    def extract_topic_from_proposal(self, proposal) -> Optional[str]:
        """Extract the topic string from the proposal output."""
        try:
            if proposal and proposal.json_object:
                root_obj = proposal.json_object.object
                if "action" in root_obj and "content" in root_obj["action"]:
                    return root_obj["action"]["content"]
                return None
            else:
                return proposal.str_content.strip() if proposal.str_content else None
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
        """Runs a single sub-round of group chat for the given cohort."""
        try:
            # Get environment and topic
            environment = cohort_agents[0].environments["group_chat"]
            topic = await self.api_utils.get_topic(cohort_id)
            if not topic:
                self.logger.warning(f"No topic found for cohort {cohort_id}")
                return

            # Update topic in the GroupChat mechanism (not the environment wrapper)
            if hasattr(environment, 'mechanism'):
                environment.mechanism._update_topic(topic)
            else:
                self.logger.error("Environment missing GroupChat mechanism")
                return

            # Run perception step
            perceptions = await self.cognitive_processor.run_parallel_perception(
                cohort_agents, 
                environment_name="group_chat"
            )
            
            for agent, perception in zip(cohort_agents, perceptions):
                idx_str = getattr(agent, "index", agent.id)
                parsed_perception = perception.json_object.object if perception and perception.json_object else perception.str_content
                log_perception(self.logger, idx_str, parsed_perception)
                agent.last_perception = parsed_perception

            # Run action step
            actions = await self.cognitive_processor.run_parallel_action(
                cohort_agents, 
                environment_name="group_chat"
            )

            # Create global action from individual actions
            global_action = GroupChatGlobalAction(actions={})
            for agent, action in zip(cohort_agents, actions):
                if action:
                    content = self.extract_message_content(action)
                    if content:
                        global_action.actions[agent.id] = GroupChatAction(
                            agent_id=agent.id,
                            action=GroupChatMessage(
                                content=content,
                                message_type="chat_message"
                            )
                        )
                        agent.last_action = content

                        model_name = agent.llm_config.model.split('/')[-1] if agent.llm_config else None
                        log_group_message(
                            self.logger,
                            cohort_id,
                            getattr(agent, "index", agent.id),
                            content,
                            sub_round=sub_round_num,
                            model_name=model_name
                        )

            # Step environment and get observations
            step_result = environment.step(global_action)
            
            # Update agent observations
            if step_result and step_result.global_observation:
                for agent in cohort_agents:
                    if agent.id in step_result.global_observation.observations:
                        agent.last_observation = step_result.global_observation.observations[agent.id]
                        self.logger.debug(
                            f"Updated observation for agent {agent.id}: {agent.last_observation}"
                        )
                    else:
                        self.logger.warning(f"No observation for agent {agent.id}")

            # Store messages in database
            messages_to_insert = []
            for agent_id, action in global_action.actions.items():
                messages_to_insert.append({
                    "message_id": str(uuid.uuid4()),
                    "agent_id": agent_id,
                    "environment_name": "group_chat",
                    "round": round_num,
                    "sub_round": sub_round_num,
                    "cohort_id": cohort_id,
                    "topic": topic,
                    "content": action.action.content,
                    "type": "group_chat_message",
                    "timestamp": datetime.now(timezone.utc)
                })

            # Run reflections for agents with observations
            agents_with_obs = [
                agent for agent in cohort_agents 
                if agent.last_observation and agent.last_observation.observation
            ]
            
            if agents_with_obs:
                self.logger.info(f"Running reflections for {len(agents_with_obs)} agents")
                reflections = await self.cognitive_processor.run_parallel_reflection(
                    agents_with_obs,
                    environment_name="group_chat"
                )
            else:
                self.logger.info("No agents with observations to reflect on")

        except Exception as e:
            self.logger.error(f"Error in sub-round {round_num}.{sub_round_num}: {e}", exc_info=True)

    def extract_message_content(self, action) -> Optional[str]:
        """Extract the 'content' from the agent's action output."""
        try:
            if action and action.json_object:
                obj = action.json_object.object
                if "action" in obj and "content" in obj["action"]:
                    return obj["action"]["content"]
                return None
            else:
                return action.str_content.strip() if action and action.str_content else None
        except Exception as e:
            self.logger.error(f"Error extracting message content: {e}")
            return None

    async def process_round_results(self, round_num: int):
        """Process and store results for the round & insert data into DB."""
        try:
            for c_id, cohort_agents in self.cohorts.items():
                environment = cohort_agents[0].environments["group_chat"]
                env_name = f"group_chat_{c_id}"
                
                # Process actions
                actions_data = []
                for agent in cohort_agents:
                    if hasattr(agent, 'last_action') and agent.last_action:
                        actions_data.append({
                            'agent_id': agent.id,
                            'environment_name': env_name,
                            'round': round_num,
                            'action': agent.last_action,
                            'cohort_id': c_id,
                            'type': 'group_chat_message'
                        })
                
                if actions_data:
                    await self.data_inserter.insert_actions(actions_data)

                # Process environment state
                if hasattr(environment, 'get_global_state'):
                    env_state = environment.get_global_state()
                    config_dict = self.orchestrator_config.model_dump() if hasattr(self.orchestrator_config, 'model_dump') else vars(self.orchestrator_config)
                    metadata = {
                        'config': config_dict,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'num_agents': len(cohort_agents),
                        'cohort_id': c_id
                    }
                    await self.data_inserter.insert_environment_state(env_name, round_num, env_state, metadata)

                self.logger.info(f"Data for round {round_num}, cohort {c_id} inserted.")
        except Exception as e:
            self.logger.error(f"Error processing round {round_num} results: {e}")
            self.logger.exception("Details:")
            raise

    async def get_round_summary(self, round_num: int) -> Dict[str, Any]:
        """Return a summary of the round for later printing."""
        summary = {
            "round": round_num,
            "agent_states": [],
            "cohorts": {
                c_id: [a.id for a in ag_list] for c_id, ag_list in self.cohorts.items()
            },
            "topics": self.topic_proposers,
        }

        for agent in self.agents:
            recent_mem = None
            if agent.short_term_memory:
                try:
                    mems = await agent.short_term_memory.retrieve_recent_memories(limit=1)
                    if mems:
                        recent_mem = mems[0].content
                except Exception:
                    pass

            summary["agent_states"].append({
                "id": agent.id,
                "index": getattr(agent, "index", None),
                "last_action": agent.last_action,
                "last_observation": agent.last_observation,
                "memory": recent_mem
            })

        return summary

    async def print_summary(self):
        """Print a high-level summary of results after all rounds."""
        log_section(self.logger, "GROUP CHAT SIMULATION SUMMARY")
        
        print("\nFinal Agent States:")
        for agent in self.agents:
            idx_str = getattr(agent, "index", agent.id)
            print(f"Agent {idx_str}")
            print(f"  Last action: {agent.last_action}")
            if agent.short_term_memory:
                reflections = await agent.short_term_memory.retrieve_recent_memories(
                    cognitive_step="reflection",
                    limit=1
                )
                if reflections:
                    print(f"  Last reflection: {reflections[0].content}")
            print()
        
        for info in self.round_summaries:
            print(f"Round {info['round']} Summary:")
            for st in info["agent_states"]:
                print(f"  Agent {st['index']} => last_action: {st['last_action']}")
            print()

    async def run(self):
        """Entry point for orchestrating the entire multi-agent conversation"""
        await self.setup_environment()
        await self.run_environment()
        await self.print_summary()