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
    GroupChatActionSpace,
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
        """
        Runs a single sub-round of group chat for the given cohort.
        """
        try:
            agents_data = []
            for ag in cohort_agents:
                agents_data.append({
                    "id": ag.id,
                    "role": getattr(ag, "role", "Participant"),
                    "persona": getattr(ag, "persona", {}),
                    "is_llm": True,
                    "llm_config": {},
                    "economic_agent": {}
                })
            agent_id_map = await self.data_inserter.insert_agents(agents_data)

            topic = await self.api_utils.get_topic(cohort_id)
            messages = await self.api_utils.get_messages(cohort_id)

            if not topic:
                self.logger.warning(f"No topic found for cohort {cohort_id}")
                return

            environment = cohort_agents[0].environments["group_chat"]
            environment.mechanism._update_topic(topic)

            for agent in cohort_agents:
                agent_messages = [m for m in messages if m.get("agent_id") == agent.id]
                agent.last_observation = agent_messages[-1] if agent_messages else None

            perceptions = await self.cognitive_processor.run_parallel_perception(
                cohort_agents, environment_name="group_chat"
            )
            for agent, perc in zip(cohort_agents, perceptions):
                idx_str = getattr(agent, "index", agent.id)
                log_persona(self.logger, idx_str, agent.persona)
                text = perc.json_object.object if (perc and perc.json_object) else perc.str_content
                log_perception(self.logger, idx_str, text)
                agent.last_perception = text

            actions = await self.cognitive_processor.run_parallel_action(
                cohort_agents, environment_name="group_chat"
            )

            messages_to_insert = []
            api_tasks = []
            for agent, action in zip(cohort_agents, actions):
                content = self.extract_message_content(action)
                if content:
                    task = asyncio.create_task(
                        self.api_utils.post_message(
                            agent_id=agent.id,
                            cohort_id=cohort_id,
                            content=content,
                            round_num=round_num,
                            sub_round_num=sub_round_num
                        )
                    )
                    api_tasks.append(task)

                    messages_to_insert.append({
                        "message_id": str(uuid.uuid4()),
                        "agent_id": agent.id,
                        "environment_name": "group_chat",
                        "round": round_num,
                        "sub_round": sub_round_num,
                        "cohort_id": cohort_id,
                        "topic": topic,
                        "content": content,
                        "type": "group_chat_message",
                        "timestamp": datetime.now(timezone.utc)
                    })

                    agent.last_action = content
                    log_group_message(
                        self.logger,
                        cohort_id,
                        getattr(agent, "index", agent.id),
                        content,
                        sub_round=sub_round_num
                    )
                else:
                    self.logger.warning(f"No content for agent {agent.id}")

            await asyncio.gather(*api_tasks)

            #if messages_to_insert:
            #    for m in messages_to_insert:
            #        if isinstance(m["timestamp"], datetime):
            #            m["timestamp"] = m["timestamp"].isoformat()
#
            #    await self.data_inserter.insert_actions(messages_to_insert, agent_id_map)

        except Exception as e:
            self.logger.error(f"Error in sub-round {round_num}.{sub_round_num}: {e}")
            return

        try:
            agents_with_observation = [a for a in cohort_agents if a.last_observation]
            if agents_with_observation:
                reflections = await self.cognitive_processor.run_parallel_reflection(
                    agents_with_observation,
                    environment_name="group_chat"
                )
                if reflections:
                    for agent, reflection_output in zip(agents_with_observation, reflections):
                        reflection_content = None
                        if reflection_output and reflection_output.json_object:
                            reflection_content = reflection_output.json_object.object
                        elif reflection_output:
                            reflection_content = reflection_output.str_content

                        if reflection_content:
                            log_reflection(self.logger, getattr(agent, "index", agent.id), reflection_content)
                            agent.last_reflection = reflection_content
                        else:
                            self.logger.warning(f"No reflection content for agent {agent.id}")
            else:
                self.logger.info("No new observations to reflect on this sub-round.")
        except Exception as e:
            self.logger.error(f"Error in reflection step for sub-round {sub_round_num}: {e}", exc_info=True)

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
                await self.data_inserter.insert_round_data(
                    round_num=round_num,
                    agents=cohort_agents,
                    environment=environment,
                    config=self.orchestrator_config,
                    environment_name=env_name
                )
                self.logger.info(f"Data for round {round_num}, cohort {c_id} inserted.")
        except Exception as e:
            self.logger.error(f"Error processing round {round_num} results: {e}")
            self.logger.exception("Details:")
            raise e

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