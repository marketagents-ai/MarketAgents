import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
import random
from colorama import Fore, Style

from market_agents.agents.market_agent import MarketAgent
from market_agents.environments.environment import MultiAgentEnvironment, EnvironmentStep
from market_agents.environments.mechanisms.group_chat import (
    GroupChat, GroupChatAction, GroupChatActionSpace, GroupChatGlobalAction,
    GroupChatMessage, GroupChatObservationSpace, GroupChatGlobalObservation
)
from market_agents.inference.message_models import LLMConfig, LLMOutput, LLMPromptContext
from market_agents.agents.protocols.acl_message import ACLMessage
from market_agents.logger_utils import *
from market_agents.agents.personas.persona import generate_persona, save_persona_to_file, Persona
from market_agents.inference.parallel_inference import ParallelAIUtilities, RequestLimits

# Set up logging for this module
logger = logging.getLogger(__name__)
logger.handlers = []  # Clear any existing handlers
logger.addHandler(logging.NullHandler())  # Add a null handler to prevent logging to the root logger

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add a file handler to log to a file
file_handler = logging.FileHandler('orchestrator.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Add a stream handler to log to console
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Set the logger's level to INFO
logger.setLevel(logging.INFO)

# Prevent propagation to avoid double logging
logger.propagate = False

class AgentConfig(BaseModel):
    num_units: int
    base_value: float
    use_llm: bool
    buyer_initial_cash: float
    buyer_initial_goods: int
    seller_initial_cash: float
    seller_initial_goods: int
    good_name: str
    noise_factor: float
    max_relative_spread: float

class EnvironmentConfig(BaseModel):
    name: str
    address: str
    max_rounds: int
    good_name: str

class GroupChatConfig(BaseModel):
    name: str
    address: str
    max_rounds: int
    initial_topic: str

class OrchestratorConfig(BaseSettings):
    num_agents: int
    max_rounds: int
    agent_config: AgentConfig
    llm_configs: List[LLMConfig]
    environment_configs: Dict[str, Union[EnvironmentConfig, GroupChatConfig]]
    protocol: str
    database_config: Dict[str, str]

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

def load_config(config_path: Path = Path("./market_agents/orchestrator_config.yaml")) -> OrchestratorConfig:
    with open(config_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return OrchestratorConfig(**yaml_data)

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

class Orchestrator:
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.agents: List[MarketAgent] = []
        self.environments: Dict[str, MultiAgentEnvironment] = {}
        self.simulation_data: List[Dict[str, Any]] = []
        self.trackers: Dict[str, GroupChatTracker] = {}
        self.log_folder = Path("./outputs/interactions")
        oai_request_limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=200000)
        anthropic_request_limits = RequestLimits(max_requests_per_minute=40, max_tokens_per_minute=10000)
        self.ai_utils = ParallelAIUtilities(
            oai_request_limits=oai_request_limits,
            anthropic_request_limits=anthropic_request_limits
        )
        self.topic_proposer = None
        self.current_topic = None

    def load_or_generate_personas(self) -> List[Persona]:
        personas_dir = Path("./market_agents/agents/personas/generated_personas")
        existing_personas = []

        if personas_dir.exists():
            for filename in personas_dir.glob("*.yaml"):
                with open(filename, 'r') as file:
                    persona_data = yaml.safe_load(file)
                    existing_personas.append(Persona(**persona_data))

        while len(existing_personas) < self.config.num_agents:
            new_persona = generate_persona()
            existing_personas.append(new_persona)
            save_persona_to_file(new_persona, personas_dir)

        return existing_personas[:self.config.num_agents]

    def generate_agents(self):
        log_section(logger, "INITIALIZING MARKET AGENTS")
        personas = self.load_or_generate_personas()

        for i, persona in enumerate(personas):
            llm_config = random.choice(self.config.llm_configs).model_dump()

            agent = MarketAgent.create(
                agent_id=str(i),  # Ensure agent_id is a string
                is_buyer=persona.role.lower() == "buyer",
                num_units=self.config.agent_config.num_units,
                base_value=self.config.agent_config.base_value,
                use_llm=self.config.agent_config.use_llm,
                initial_cash=self.config.agent_config.buyer_initial_cash if persona.role.lower() == "buyer" else self.config.agent_config.seller_initial_cash,
                initial_goods=self.config.agent_config.buyer_initial_goods if persona.role.lower() == "buyer" else self.config.agent_config.seller_initial_goods,
                good_name=self.config.agent_config.good_name,
                noise_factor=self.config.agent_config.noise_factor,
                max_relative_spread=self.config.agent_config.max_relative_spread,
                llm_config=llm_config,
                protocol=ACLMessage,
                environments=self.environments,
                persona=persona
            )
            self.agents.append(agent)
            log_agent_init(logger, i, agent.is_buyer, persona)

        self.topic_proposer = random.choice(self.agents)
        self.topic_proposer.system = "You are the group chat topic proposer agent. Your role is to propose interesting and relevant topics for the group discussion."

    def setup_environments(self):
        log_section(logger, "CONFIGURING GROUP CHAT ENVIRONMENT")
        group_chat_config = self.config.environment_configs['group_chat']
        group_chat = GroupChat(
            max_rounds=group_chat_config.max_rounds,
            current_topic=group_chat_config.initial_topic,
            speaker_order=[str(agent.id) for agent in self.agents],
            sequential=False
        )
        env = MultiAgentEnvironment(
            name=group_chat_config.name,
            address=group_chat_config.address,
            max_steps=group_chat_config.max_rounds,
            action_space=GroupChatActionSpace(),
            observation_space=GroupChatObservationSpace(),
            mechanism=group_chat
        )
        self.environments['group_chat'] = env
        self.trackers['group_chat'] = GroupChatTracker()
        log_environment_setup(logger, "group_chat")

        for agent in self.agents:
            agent.environments = self.environments

    async def run_parallel_ai_completion(self, prompts: List[LLMPromptContext]) -> List[LLMOutput]:
        return await self.ai_utils.run_parallel_ai_completion(prompts, update_history=False)

    async def run_parallel_perceive(self, env_name: str) -> List[LLMPromptContext]:
        perceive_prompts = []
        for agent in self.agents:
            perceive_prompt = await agent.perceive(env_name, return_prompt=True)
            perceive_prompts.append(perceive_prompt)
        return perceive_prompts

    async def run_parallel_generate_action(self, env_name: str, perceptions: List[str]) -> List[LLMPromptContext]:
        action_prompts = []
        for agent, perception in zip(self.agents, perceptions):
            action_prompt = await agent.generate_action(env_name, perception, return_prompt=True)
            action_prompts.append(action_prompt)
        return action_prompts

    async def run_parallel_reflect(self, env_name: str) -> Tuple[List[LLMPromptContext], List[MarketAgent]]:
        reflect_prompts = []
        agents_with_observations = []
        for agent in self.agents:
            if agent.last_observation:
                reflect_prompt = await agent.reflect(env_name, return_prompt=True)
                reflect_prompts.append(reflect_prompt)
                agents_with_observations.append(agent)
            else:
                logger.info(f"Skipping reflection for agent {agent.id} due to no observation")
        return reflect_prompts, agents_with_observations
    async def generate_initial_topic(self) -> str:
        topic_action = await self.topic_proposer.generate_action(
            "group_chat",
            "Consider recent economic events, market trends, or financial news. For this round discuss how Fed rate cut of 50 bps is going to impact the market"
        )
        logger.debug(f"Topic proposer {self.topic_proposer.id} generated action: {topic_action}")

        try:
            if isinstance(topic_action, dict):
                if 'content' in topic_action:
                    if isinstance(topic_action['content'], dict) and 'action' in topic_action['content']:
                        content = topic_action['content']['action']['content']
                    else:
                        content = topic_action['content']
                elif 'action' in topic_action:
                    content = topic_action['action']['content']
                else:
                    raise ValueError("Unexpected topic_action structure")
            else:
                raise ValueError("topic_action is not a dictionary")

            logger.info(f"Proposed topic: {Fore.YELLOW}{content}{Style.RESET_ALL}")
            return content
        except Exception as e:
            logger.error(f"Invalid topic action structure: {e}")
            default_topic = "Default topic: Recent market trends"
            logger.info(f"Using default topic: {Fore.YELLOW}{default_topic}{Style.RESET_ALL}")
            return default_topic

    async def run_environment(self, env_name: str, round_num: int) -> EnvironmentStep:
        env = self.environments[env_name]
        tracker = self.trackers[env_name]

        log_running(logger, env_name)

        if round_num == 1:
            self.current_topic = await self.generate_initial_topic()
            env.mechanism.current_topic = self.current_topic
            #logger.info(f"Initial topic set: {self.current_topic}")

        perception_prompts = await self.run_parallel_perceive(env_name)
        perceptions = await self.run_parallel_ai_completion(perception_prompts)

        # Create a mapping of agent IDs to perceptions
        perceptions_map = {perception.source_id: perception for perception in perceptions}

        for agent in self.agents:
            perception = perceptions_map.get(str(agent.id))
            if perception:
                perception_content = perception.json_object.object if perception.json_object else perception.str_content
                log_section(logger, f"Current Agent:\nAgent {agent.id} with persona:\n{agent.persona}")
                log_perception(logger, int(agent.id), f"{Fore.CYAN}{perception_content}{Style.RESET_ALL}")
            else:
                logger.warning(f"No perception found for agent {agent.id}")

        self.set_agent_system_messages(round_num)

        # Extract perception content to pass to generate_action
        perception_contents = []
        for agent in self.agents:
            perception = perceptions_map.get(str(agent.id))
            if perception:
                perception_content = perception.json_object.object if perception.json_object else perception.str_content
                perception_contents.append(perception_content)
            else:
                perception_contents.append("")

        action_prompts = await self.run_parallel_generate_action(env_name, perception_contents)
        actions = await self.run_parallel_ai_completion(action_prompts)

        # Create a mapping of agent IDs to actions
        actions_map = {action.source_id: action for action in actions}

        global_action = self.process_group_chat_actions(actions_map)
        logger.debug(f"Global action before env step: {global_action}")
        env_step = env.step(global_action)
        logger.info(f"Completed {env_name} step")
        logger.debug(f"Environment step result: {env_step}")

        if isinstance(env_step.global_observation, GroupChatGlobalObservation):
            self.process_group_chat_messages(env_step.global_observation, tracker)
        else:
            logger.error(f"Unexpected global observation type: {type(env_step.global_observation)}")

        return env_step

    def set_agent_system_messages(self, round_num: int):
        for agent in self.agents:
            agent.system = f"You are participating in round {round_num} of a group chat about {self.current_topic}. Your role is {'buyer' if agent.is_buyer else 'seller'}. Engage in the discussion and share your insights."

    def process_group_chat_actions(self, actions_map: Dict[str, LLMOutput]) -> GroupChatGlobalAction:
        agent_actions = {}
        for agent_id, action_output in actions_map.items():
            try:
                action_content = action_output.json_object.object if action_output.json_object else json.loads(action_output.str_content)
                group_chat_message = GroupChatMessage(
                    content=action_content['action']['content'],
                    message_type=action_content.get('message_type', 'group_message'),
                    agent_id=agent_id
                )
                group_chat_action = GroupChatAction(agent_id=agent_id, action=group_chat_message)
                agent_actions[agent_id] = group_chat_action.model_dump()
                log_action(logger, int(agent_id), f"{Fore.BLUE}Message: {group_chat_message.content}{Style.RESET_ALL}")
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                logger.error(f"Error creating GroupChatAction for agent {agent_id}: {str(e)}")
                continue
        return GroupChatGlobalAction(actions=agent_actions)

    def process_group_chat_messages(self, global_observation: GroupChatGlobalObservation, tracker: GroupChatTracker):
        for message in global_observation.all_messages:
            tracker.add_message(message)
        if global_observation.current_topic != self.current_topic:
            self.current_topic = global_observation.current_topic
            tracker.add_topic(self.current_topic)
        logger.info(f"Processed {len(global_observation.all_messages)} messages in group chat")

    async def run_simulation(self):
        log_section(logger, "SIMULATION COMMENCING")

        try:
            for round_num in range(1, self.config.max_rounds + 1):
                log_round(logger, round_num)

                env_step = await self.run_environment("group_chat", round_num)
                self.update_simulation_state("group_chat", env_step)

                for agent in self.agents:
                    local_step = env_step.get_local_step(str(agent.id))
                    if local_step:
                        agent.last_observation = local_step.observation

                reflect_prompts, agents_with_observations = await self.run_parallel_reflect("group_chat")
                reflections = await self.run_parallel_ai_completion(reflect_prompts)
                reflections_map = {reflection.source_id: reflection for reflection in reflections}

                # Process reflections
                for agent in agents_with_observations:
                    reflection = reflections_map.get(str(agent.id))
                    if reflection and reflection.json_object:
                        log_reflection(logger, int(agent.id), f"{Fore.MAGENTA}{reflection.json_object.object}{Style.RESET_ALL}")
                        agent.memory.append({
                            "type": "reflection",
                            "content": reflection.json_object.object["reflection"],
                            "strategy_update": reflection.json_object.object["strategy_update"],
                            "observation": agent.last_observation,
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        logger.warning(f"No reflection found for agent {agent.id}")

                self.save_round_data(round_num)
                self.save_agent_interactions(round_num)

                if env_step.done:
                    logger.info("Environment signaled completion. Ending simulation.")
                    break
        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            logger.exception("Exception details:")
        finally:
            log_completion(logger, "SIMULATION COMPLETED")
            self.print_summary()

    def update_simulation_state(self, env_name: str, env_step: EnvironmentStep):
        if not self.simulation_data or 'state' not in self.simulation_data[-1]:
            self.simulation_data.append({'state': {}})

        self.simulation_data[-1]['state'][env_name] = env_step.info
        self.simulation_data[-1]['state']['messages'] = env_step.global_observation.all_messages

    def save_round_data(self, round_num):
        round_data = {
            'round': round_num,
            'agent_states': [{
                'id': agent.id,
                'role': 'buyer' if agent.is_buyer else 'seller',
                'last_action': agent.last_action,
                'memory': agent.memory[-1] if agent.memory else None
            } for agent in self.agents],
            'environment_states': {name: env.get_global_state() for name, env in self.environments.items()},
        }

        if self.simulation_data and 'state' in self.simulation_data[-1]:
            round_data['state'] = self.simulation_data[-1]['state']

        self.simulation_data.append(round_data)

    def save_agent_interactions(self, round_num):
        self.log_folder.mkdir(parents=True, exist_ok=True)

        for agent in self.agents:
            file_path = self.log_folder / f"agent_{agent.id}_interactions.jsonl"
            with open(file_path, 'a') as f:
                new_interactions = [interaction for interaction in agent.interactions if 'round' not in interaction]
                for interaction in new_interactions:
                    interaction_with_round = {
                        "round": round_num,
                        **interaction
                    }
                    json.dump(interaction_with_round, f)
                    f.write('\n')
                agent.interactions = [interaction for interaction in agent.interactions if 'round' in interaction]

        logger.info(f"Saved agent interactions for round {round_num} to {self.log_folder}")

    def print_summary(self):
        log_section(logger, "SIMULATION SUMMARY")

        group_chat_tracker = self.trackers['group_chat']
        group_chat_summary = group_chat_tracker.get_summary()
        print("\nGroup Chat Summary:")
        print(f"Total messages: {group_chat_summary['total_messages']}")
        print(f"Total topics discussed: {group_chat_summary['total_topics']}")

        print("\nFinal Agent States:")
        for agent in self.agents:
            print(f"Agent {agent.id}:")
            print(f"  Last action: {agent.last_action}")
            if agent.memory:
                print(f"  Last reflection: {agent.memory[-1]['content']}")

    async def start(self):
        log_section(logger, "GROUP CHAT SIMULATION INITIALIZING")
        self.generate_agents()
        self.setup_environments()

        await self.run_simulation()

if __name__ == "__main__":
    config = load_config()
    orchestrator = Orchestrator(config)
    asyncio.run(orchestrator.start())
