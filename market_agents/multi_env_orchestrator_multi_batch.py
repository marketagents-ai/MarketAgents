# joint_orchestrator.py

import asyncio
import json
import logging
import os
import random
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml
from colorama import Fore, Style
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from market_agents.agents.db.setup_database import create_database, create_tables
from market_agents.agents.market_agent import MarketAgent
from market_agents.agents.personas.persona import Persona, generate_persona, save_persona_to_file
from market_agents.agents.protocols.acl_message import ACLMessage
from market_agents.economics.econ_agent import EconomicAgent
from market_agents.economics.econ_models import Ask, Basket, Bid, BuyerPreferenceSchedule, Endowment, Good, SellerPreferenceSchedule, Trade
from market_agents.environments.environment import EnvironmentStep, MultiAgentEnvironment
from market_agents.environments.mechanisms.auction import (
    AuctionAction,
    AuctionActionSpace,
    AuctionGlobalObservation,
    AuctionObservationSpace,
    DoubleAuction,
    GlobalAuctionAction
)
from market_agents.environments.mechanisms.group_chat import (
    GroupChat,
    GroupChatAction,
    GroupChatActionSpace,
    GroupChatGlobalAction,
    GroupChatMessage,
    GroupChatObservationSpace,
    GroupChatGlobalObservation
)
from market_agents.inference.message_models import LLMConfig, LLMOutput, LLMPromptContext
from market_agents.inference.parallel_inference import ParallelAIUtilities, RequestLimits
from market_agents.insert_simulation_data import SimulationDataInserter
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

# Set up logging for this module
logger = logging.getLogger(__name__)
logger.handlers = []

# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add a file handler to log to a file
file_handler = logging.FileHandler('joint_orchestrator.log')
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
    buyer_base_value: float
    seller_base_value: float
    use_llm: bool
    buyer_initial_cash: float
    buyer_initial_goods: int
    seller_initial_cash: float
    seller_initial_goods: int
    good_name: str
    noise_factor: float
    max_relative_spread: float


class AuctionConfig(BaseModel):
    name: str
    address: str
    max_rounds: int
    good_name: str


class GroupChatConfig(BaseModel):
    name: str
    address: str
    max_rounds: int
    initial_topic: str
    sub_rounds: int = Field(default=3)
    group_size: int = Field(default=100)


class LLMConfigModel(BaseModel):
    name: str
    client: str
    model: str
    temperature: float
    max_tokens: int
    use_cache: bool


class DatabaseConfig(BaseSettings):
    db_type: str = "postgres"
    db_name: str = "market_simulation"
    db_user: str = Field(..., env='DB_USER')
    db_password: str = Field(..., env='DB_PASSWORD')
    db_host: str = Field('localhost', env='DB_HOST')
    db_port: str = Field('5433', env='DB_PORT')

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')


class OrchestratorConfig(BaseSettings):
    num_agents: int
    max_rounds: int
    agent_config: AgentConfig
    llm_configs: List[LLMConfigModel]
    environment_configs: Dict[str, Union[AuctionConfig, GroupChatConfig]]
    protocol: str
    database_config: DatabaseConfig = DatabaseConfig()

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')


def load_config(config_path: Path = Path("./market_agents/orchestrator_config.yaml")) -> OrchestratorConfig:
    with open(config_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return OrchestratorConfig(**yaml_data)


def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        raise TypeError(f"Type {type(obj)} not serializable")


def serialize_memory_data(memory_data):
    if isinstance(memory_data, dict):
        return {k: serialize_memory_data(v) for k, v in memory_data.items()}
    elif isinstance(memory_data, list):
        return [serialize_memory_data(v) for v in memory_data]
    elif isinstance(memory_data, datetime):
        return memory_data.isoformat()
    elif hasattr(memory_data, 'model_dump'):
        return serialize_memory_data(memory_data.model_dump())
    elif hasattr(memory_data, '__dict__'):
        return serialize_memory_data(vars(memory_data))
    elif isinstance(memory_data, (str, int, float, bool, type(None))):
        return memory_data
    else:
        return str(memory_data)


class AuctionTracker:
    def __init__(self):
        self.all_trades: List[Trade] = []
        self.per_round_surplus: List[float] = []
        self.per_round_quantities: List[int] = []

    def add_trade(self, trade: Trade):
        self.all_trades.append(trade)

    def add_round_data(self, surplus: float, quantity: int):
        self.per_round_surplus.append(surplus)
        self.per_round_quantities.append(quantity)

    def get_summary(self):
        return {
            "total_trades": len(self.all_trades),
            "total_surplus": sum(self.per_round_surplus),
            "total_quantity": sum(self.per_round_quantities)
        }


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
        self.dashboard = None
        self.database = None
        self.simulation_order = ['group_chat', 'auction']
        self.simulation_data: List[Dict[str, Any]] = []
        self.latest_data = None
        self.trackers: Dict[str, Union[AuctionTracker, GroupChatTracker]] = {}
        self.log_folder = Path("./outputs/interactions")
        # Initialize database parameters from config
        self.db_params = {
            'dbname': self.config.database_config.db_name,
            'user': self.config.database_config.db_user,
            'password': self.config.database_config.db_password,
            'host': self.config.database_config.db_host,
            'port': self.config.database_config.db_port
        }
        self.data_inserter = SimulationDataInserter(self.db_params)
        oai_request_limits = RequestLimits(max_requests_per_minute=20000, max_tokens_per_minute=2000000)
        anthropic_request_limits = RequestLimits(max_requests_per_minute=20000, max_tokens_per_minute=2000000)
        self.ai_utils = ParallelAIUtilities(
            oai_request_limits=oai_request_limits,
            anthropic_request_limits=anthropic_request_limits
        )
        self.group_size = config.environment_configs['group_chat'].group_size
        self.sub_rounds_per_group_chat = config.environment_configs['group_chat'].sub_rounds
        self.agent_batches: List[List[MarketAgent]] = []
        self.agent_surpluses = {}
        self.agent_dict: Dict[str, MarketAgent] = {}
        self.topic_proposer = None
        self.current_topic = None

    def load_or_generate_personas(self) -> List[Persona]:
        personas_dir = Path("./market_agents/agents/personas/generated_personas")
        existing_personas = []

        if os.path.exists(personas_dir):
            for filename in os.listdir(personas_dir):
                if filename.endswith(".yaml"):
                    with open(os.path.join(personas_dir, filename), 'r') as file:
                        persona_data = yaml.safe_load(file)
                        existing_personas.append(Persona(**persona_data))

        while len(existing_personas) < self.config.num_agents:
            new_persona = generate_persona()
            existing_personas.append(new_persona)
            save_persona_to_file(new_persona, personas_dir)

        return existing_personas[:self.config.num_agents]
    
    def batch_agents(self):
        self.agent_batches = [
            self.agents[i:i + self.group_size]
            for i in range(0, len(self.agents), self.group_size)
        ]
        logger.info(f"Agents divided into {len(self.agent_batches)} batches of up to {self.group_size} agents each.")

    def generate_agents(self):
        log_section(logger, "INITIALIZING MARKET AGENTS")
        personas = self.load_or_generate_personas()
        num_agents = len(personas)
        num_buyers = num_agents // 2
        num_sellers = num_agents - num_buyers

        for i, persona in enumerate(personas):
            agent_uuid = str(uuid.uuid4())
            # Randomly assign an LLM config if there are multiple configs
            llm_config = random.choice(self.config.llm_configs).dict() if len(self.config.llm_configs) > 1 else self.config.llm_configs[0].dict()
            # Assign roles explicitly based on index
            if i < num_buyers:
                is_buyer = True
                persona.role = "buyer"
            else:
                is_buyer = False
                persona.role = "seller"

            agent_config = self.config.agent_config.dict()
            if is_buyer:
                initial_cash = agent_config.get('buyer_initial_cash', 1000)
                initial_goods_quantity = agent_config.get('buyer_initial_goods', 0)
                base_value = agent_config.get('buyer_base_value', 120.0)
            else:
                initial_cash = agent_config.get('seller_initial_cash', 0)
                initial_goods_quantity = agent_config.get('seller_initial_goods', 10)
                base_value = agent_config.get('seller_base_value', 80.0)

            good_name = agent_config.get('good_name', 'apple')
            initial_goods = {good_name: initial_goods_quantity}

            # Create initial basket and endowment
            initial_basket = Basket(
                cash=initial_cash,
                goods=[Good(name=good_name, quantity=initial_goods_quantity)]
            )
            endowment = Endowment(
                initial_basket=initial_basket,
                agent_id=agent_uuid
            )

            # Create preference schedules
            if is_buyer:
                value_schedules = {
                    good_name: BuyerPreferenceSchedule(
                        num_units=agent_config.get('num_units', 10),
                        base_value=base_value,
                        noise_factor=agent_config.get('noise_factor', 0.05)
                    )
                }
                cost_schedules = {}
            else:
                value_schedules = {}
                cost_schedules = {
                    good_name: SellerPreferenceSchedule(
                        num_units=agent_config.get('num_units', 10),
                        base_value=base_value,
                        noise_factor=agent_config.get('noise_factor', 0.05)
                    )
                }

            economic_agent = EconomicAgent(
                id=agent_uuid,
                endowment=endowment,
                value_schedules=value_schedules,
                cost_schedules=cost_schedules,
                max_relative_spread=agent_config.get('max_relative_spread', 0.2)
            )

            agent = MarketAgent.create(
                agent_id=agent_uuid,
                use_llm=agent_config.get('use_llm', True),
                llm_config=llm_config,
                environments=self.environments,
                protocol=ACLMessage,
                persona=persona,
                econ_agent=economic_agent
            )

            # Initialize last_perception and last_observation
            agent.last_perception = None
            agent.last_observation = None
            agent.last_step = None
            agent.index = i
            self.agents.append(agent)
            self.agent_dict[agent.id] = agent
            log_agent_init(logger, agent.index, is_buyer, persona)
        
        # Initialize topic_proposer
        self.topic_proposer = random.choice(self.agents)
        self.topic_proposer.system = "You are the group chat topic proposer agent. Your role is to propose interesting and relevant topics for the group discussion."
        self.batch_agents()
        logger.info(f"Topic proposer initialized: Agent {self.topic_proposer.id}")

    def setup_environments(self):
        log_section(logger, "CONFIGURING ENVIRONMENTS")
        # Create Auction Environment
        auction_config = self.config.environment_configs.get('auction')
        if auction_config:
            double_auction = DoubleAuction(
                max_rounds=auction_config.max_rounds,
                good_name=auction_config.good_name
            )
            auction_env = MultiAgentEnvironment(
                name=auction_config.name,
                address=auction_config.address,
                max_steps=auction_config.max_rounds,
                action_space=AuctionActionSpace(),
                observation_space=AuctionObservationSpace(),
                mechanism=double_auction
            )
            self.environments['auction'] = auction_env
            self.trackers['auction'] = AuctionTracker()
            log_environment_setup(logger, 'auction')
        else:
            logger.error("Auction configuration not found in environment_configs.")
            raise ValueError("Auction configuration missing.")

        # Create GroupChat Environments per Batch
        group_chat_config = self.config.environment_configs.get('group_chat')
        if group_chat_config:
            for batch_index, batch in enumerate(self.agent_batches):
                group_chat = GroupChat(
                    max_rounds=group_chat_config.max_rounds,
                    current_topic=group_chat_config.initial_topic,  # Fixed here
                    speaker_order=[str(agent.id) for agent in batch],
                    sequential=False,
                    sub_rounds=self.sub_rounds_per_group_chat
                )
                group_chat_env_name = f"group_chat_batch_{batch_index}"
                group_chat_env = MultiAgentEnvironment(
                    name=f"{group_chat_config.name}_batch_{batch_index}",
                    address=f"{group_chat_config.address}_{batch_index}",
                    max_steps=group_chat_config.max_rounds,
                    action_space=GroupChatActionSpace(),
                    observation_space=GroupChatObservationSpace(),
                    mechanism=group_chat
                )
                self.environments[group_chat_env_name] = group_chat_env
                self.trackers[group_chat_env_name] = GroupChatTracker()
                log_environment_setup(logger, group_chat_env_name)
        else:
            logger.error("GroupChat configuration not found in environment_configs.")
            raise ValueError("GroupChat configuration missing.")

        logger.info(f"Set up {len(self.environments)} environments")

        for agent in self.agents:
            agent.environments = self.environments

    def setup_database(self):
        log_section(logger, "CONFIGURING SIMULATION DATABASE")
        # Create the database if it doesn't exist
        create_database(db_params=self.db_params)
        # Check if required tables exist
        if not self.data_inserter.check_tables_exist():
            create_tables(db_params=self.db_params)
        else:
            logger.info("Required tables already exist.")
        logger.info("Database setup completed")

    async def run_parallel_ai_completion(self, prompts: List[LLMPromptContext]) -> List[LLMOutput]:
        # Implement batching within AI completions if necessary
        results = await self.ai_utils.run_parallel_ai_completion(prompts, update_history=False)
        # Insert AI requests into the database
        ai_requests = self.ai_utils.get_all_requests()
        self.data_inserter.insert_ai_requests(ai_requests)
        return results

    async def run_parallel_perceive(self, env_name: str, batch: List[MarketAgent] = None) -> List[LLMPromptContext]:
        perceive_prompts = []
        if env_name.startswith('group_chat_batch_'):
            # Per batch perceive
            for agent in batch:
                perceive_prompt = await agent.perceive(env_name, return_prompt=True)
                perceive_prompts.append(perceive_prompt)
        else:
            # For auction perceive all agents
            for agent in self.agents:
                perceive_prompt = await agent.perceive(env_name, return_prompt=True)
                perceive_prompts.append(perceive_prompt)
        return perceive_prompts

    async def run_parallel_generate_action(self, env_name: str, perceptions: List[str], batch: List[MarketAgent] = None) -> List[LLMPromptContext]:
        action_prompts = []
        if env_name.startswith('group_chat_batch_'):
            # Per batch generate_action
            for agent, perception in zip(batch, perceptions):
                action_prompt = await agent.generate_action(env_name, perception, return_prompt=True)
                action_prompts.append(action_prompt)
        else:
            # For auction generate_action for all agents
            for agent, perception in zip(self.agents, perceptions):
                action_prompt = await agent.generate_action(env_name, perception, return_prompt=True)
                action_prompts.append(action_prompt)
        return action_prompts

    async def run_parallel_reflect(self, env_name: str) -> List[LLMPromptContext]:
        reflect_prompts = []
        for agent in self.agents:
            if agent.last_observation:
                reflect_prompt = await agent.reflect(env_name, return_prompt=True)
                reflect_prompts.append(reflect_prompt)
            else:
                logger.info(f"Skipping reflection for agent {agent.index} due to no observation")
        return reflect_prompts

    def set_agent_system_messages(self, env_name: str, round_num: int, sub_round_num: int = None, **kwargs):
        if env_name.startswith('group_chat_batch_'):
            # Extract batch index
            batch_index = env_name.split('_')[-1]
            for agent in self.agent_batches[int(batch_index)]:
                if sub_round_num:
                    agent.system = (
                        f"You are participating in sub-round {sub_round_num} of round {round_num} "
                        f"in a group chat about '{kwargs.get('current_topic', 'various topics')}'. "
                        f"Your role is {'buyer' if agent.role == 'buyer' else 'seller'}. "
                        "Engage in a informative and fun discussion and share your insights."
                        "Use emojis to maintain a friendly and playful tone."
                    )
                else:
                    agent.system = (
                        f"You are participating in round {round_num} "
                        f"in a group chat about '{kwargs.get('current_topic', 'various topics')}'. "
                        f"Your role is {'buyer' if agent.role == 'buyer' else 'seller'}. "
                        "Engage in a informative and fun discussion and share your insights."
                        "Use emojis to maintain a friendly and playful tone."
                    )
        elif env_name == 'auction':
            for agent in self.agents:
                good_name = kwargs.get('good_name', 'apple')
                current_cash = agent.economic_agent.endowment.current_basket.cash
                current_goods = agent.economic_agent.endowment.current_basket.get_good_quantity(good_name)
                if round_num == 1:
                    if agent.role == "buyer":
                        current_value = agent.economic_agent.get_current_value(good_name)
                        if current_value is not None:
                            suggested_price = current_value * 0.99
                            agent.system = (
                                f"This is the first round of the market so there are no bids or asks yet. "
                                f"You have {current_cash:.2f} cash and {current_goods} units of {good_name}. "
                                f"You can make a profit by buying at {suggested_price:.2f} or lower."
                            )
                        else:
                            agent.system = f"You have reached your maximum quantity for {good_name}."
                    elif agent.role == "seller":
                        if current_goods <= 0:
                            agent.system = f"You have no {good_name} to sell."
                        else:
                            current_cost = agent.economic_agent.get_current_cost(good_name)
                            if current_cost is not None:
                                suggested_price = current_cost * 1.01
                                agent.system = (
                                    f"This is the first round of the market so there are no bids or asks yet. "
                                    f"You have {current_cash:.2f} cash and {current_goods} units of {good_name}. "
                                    f"You can make a profit by selling at {suggested_price:.2f} or higher."
                                )
                            else:
                                agent.system = f"You have no more {good_name} to sell."
                else:
                    if agent.role == "buyer":
                        marginal_value = agent.economic_agent.get_current_value(good_name)
                        if marginal_value is not None:
                            suggested_price = marginal_value * 0.99
                            agent.system = (
                                f"Your current cash: {current_cash:.2f}, goods: {current_goods}. "
                                f"Your marginal value for the next unit of {good_name} is {marginal_value:.2f}. "
                                f"You can make a profit by buying at {suggested_price:.2f} or lower."
                            )
                        else:
                            agent.system = f"You have reached your maximum quantity for {good_name}."
                    elif agent.role == "seller":
                        if current_goods <= 0:
                            agent.system = f"You have no {good_name} to sell."
                        else:
                            marginal_cost = agent.economic_agent.get_current_cost(good_name)
                            if marginal_cost is not None:
                                suggested_price = marginal_cost * 1.01
                                agent.system = (
                                    f"Your current cash: {current_cash:.2f}, goods: {current_goods}. "
                                    f"Your marginal cost for selling the next unit of {good_name} is {marginal_cost:.2f}. "
                                    f"You can make a profit by selling at {suggested_price:.2f} or higher."
                                )
                            else:
                                agent.system = f"You have no more {good_name} to sell."

    async def run_environment(self, env_name: str, round_num: int, sub_round_num: int = None, batch: List[MarketAgent] = None) -> EnvironmentStep:
        env = self.environments[env_name]
        tracker = self.trackers[env_name]

        log_running(logger, env_name)

        if env_name == 'auction':
            # Reset pending orders at the beginning of the auction round
            for agent in self.agents:
                agent.economic_agent.reset_all_pending_orders()

        # Set system messages for agents based on environment
        if env_name.startswith('group_chat_batch_'):
            self.set_agent_system_messages(env_name, round_num, sub_round_num=sub_round_num, current_topic=self.current_topic)
        elif env_name == 'auction':
            good_name = env.mechanism.good_name
            self.set_agent_system_messages(env_name, round_num, good_name=good_name)

        if env_name.startswith('group_chat_batch_'):
            batch_index = int(env_name.split('_')[-1])
            batch_agents = self.agent_batches[batch_index]

            if sub_round_num == 1:
                # First sub-round: perception and generate_action
                perception_prompts = await self.run_parallel_perceive(env_name, batch=batch_agents)
                perceptions = await self.run_parallel_ai_completion(perception_prompts)
                perceptions_map = {perception.source_id: perception for perception in perceptions}

                for agent in batch_agents:
                    perception = perceptions_map.get(agent.id)
                    if perception:
                        log_section(logger, f"Current Agent:\nAgent {agent.index} with persona:\n{agent.persona}")
                        perception_content = perception.json_object.object if perception.json_object else perception.str_content
                        log_perception(logger, agent.index, f"{Fore.CYAN}{perception_content}{Style.RESET_ALL}")
                        agent.last_perception = perception_content
                    else:
                        logger.warning(f"No perception found for agent {agent.index} in {env_name}")
                        agent.last_perception = None

                perception_contents = [agent.last_perception or "" for agent in batch_agents]
                action_prompts = await self.run_parallel_generate_action(env_name, perception_contents, batch=batch_agents)
            elif sub_round_num == self.sub_rounds_per_group_chat:
                # Final sub-round: generate_action and reflection
                perception_contents = [agent.last_perception or "" for agent in batch_agents]
                action_prompts = await self.run_parallel_generate_action(env_name, perception_contents, batch=batch_agents)
                reflection_prompts = await self.run_parallel_reflect(env_name)
                reflection_results = await self.run_parallel_ai_completion(reflection_prompts)
                # Process reflection results as needed
            else:
                # Middle sub-rounds: only generate_action
                perception_contents = [agent.last_perception or "" for agent in batch_agents]
                action_prompts = await self.run_parallel_generate_action(env_name, perception_contents, batch=batch_agents)

            actions = await self.run_parallel_ai_completion(action_prompts)
            actions_map = {action.source_id: action for action in actions}
            global_action = self.process_group_chat_actions(actions_map)

            # Add last_observation to agents after each sub-round
            env_state = env.step(global_action)
            for agent in batch_agents:
                agent_observation = env_state.global_observation.observations.get(agent.id)
                if agent_observation:
                    agent.last_observation = agent_observation
                    agent.last_step = env_state

                    if hasattr(agent_observation.observation, 'model_dump'):
                        observation_dict = agent_observation.observation.model_dump()
                    else:
                        observation_dict = agent_observation.observation

                    # You can process the group chat observation here if needed
                else:
                    agent.last_perception = None

        else:
            # Auction environment logic (unchanged)
            perception_prompts = await self.run_parallel_perceive(env_name)
            perceptions = await self.run_parallel_ai_completion(perception_prompts)
            perceptions_map = {perception.source_id: perception for perception in perceptions}

            for agent in self.agents:
                perception = perceptions_map.get(agent.id)
                if perception:
                    log_section(logger, f"Current Agent:\nAgent {agent.index} with persona:\n{agent.persona}")
                    perception_content = perception.json_object.object if perception.json_object else None
                    log_perception(logger, agent.index, f"{Fore.CYAN}{json.dumps(perception_content)}{Style.RESET_ALL}")
                    agent.last_perception = perception_content
                else:
                    logger.warning(f"No perception found for agent {agent.index} in {env_name}")
                    agent.last_perception = None

            perception_contents = [agent.last_perception or "" for agent in self.agents]
            action_prompts = await self.run_parallel_generate_action(env_name, perception_contents)
            actions = await self.run_parallel_ai_completion(action_prompts)
            actions_map = {action.source_id: action for action in actions}
            global_action = self.process_auction_actions(actions_map, env)

        try:
            env_state = env.step(global_action)
        except Exception as e:
            logger.error(f"Error in environment {env_name}: {str(e)}")
            raise e  # Re-raise the exception to be caught in run_simulation

        logger.info(f"Completed {env_name} step")

        if env_name.startswith('group_chat_batch_') and isinstance(env_state.global_observation, GroupChatGlobalObservation):
            self.process_group_chat_messages(env_state.global_observation, tracker)
        elif env_name == 'auction' and isinstance(env_state.global_observation, AuctionGlobalObservation):
            self.process_trades(env_state.global_observation, tracker)

        return env_state

    def process_group_chat_actions(self, actions_map: Dict[str, LLMOutput]) -> GroupChatGlobalAction:
        agent_actions = {}
        for agent_id, action_output in actions_map.items():
            try:
                action_content = action_output.json_object.object if action_output.json_object else json.loads(action_output.str_content or '{}')
                group_chat_message = GroupChatMessage(
                    content=action_content['action']['content'],
                    message_type=action_content.get('message_type', 'group_message'),
                    agent_id=agent_id
                )
                group_chat_action = GroupChatAction(agent_id=agent_id, action=group_chat_message)
                agent_actions[agent_id] = group_chat_action.model_dump()
                # Use the agent's index for logging instead of trying to convert the UUID to an int
                agent = next((a for a in self.agents if a.id == agent_id), None)
                log_action(logger, agent.index if agent else "Unknown", f"{Fore.BLUE}Message: {group_chat_message.content}{Style.RESET_ALL}")
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                logger.error(f"Error creating GroupChatAction for agent {agent_id}: {str(e)}")
                continue
        return GroupChatGlobalAction(actions=agent_actions)

    def process_auction_actions(self, actions_map: Dict[str, LLMOutput], env: MultiAgentEnvironment) -> GlobalAuctionAction:
        agent_actions = {}
        good_name = env.mechanism.good_name
        for agent_id, action_output in actions_map.items():
            try:
                action_content = action_output.json_object.object if action_output.json_object else json.loads(action_output.str_content or '{}')
                if 'price' in action_content and 'quantity' in action_content:
                    # Optionally, you can modify quantity or other parameters here
                    action_content['quantity'] = 1  # Ensuring quantity is 1 as per original code

                    # Determine if the agent is a buyer or seller
                    agent = self.agent_dict.get(agent_id)
                    if not agent:
                        logger.warning(f"Agent with ID {agent_id} not found.")
                        continue

                    if agent.role == "buyer":
                        auction_action = Bid(**action_content)
                    elif agent.role == "seller":
                        auction_action = Ask(**action_content)
                    else:
                        raise ValueError(f"Invalid agent role: {agent.role}")

                    agent_actions[agent_id] = AuctionAction(agent_id=agent_id, action=auction_action)

                    # Update agent's pending orders
                    agent.economic_agent.pending_orders.setdefault(good_name, []).append(auction_action)

                    action_type = "Bid" if isinstance(auction_action, Bid) else "Ask"
                    color = Fore.BLUE if action_type == "Bid" else Fore.GREEN
                    log_action(logger, agent.index, f"{color}{action_type}: {auction_action}{Style.RESET_ALL}")
                else:
                    raise ValueError(f"Invalid action content: {action_content}")
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Error creating AuctionAction for agent {agent_id}: {str(e)}")
        return GlobalAuctionAction(actions=agent_actions)

    def process_trades(self, global_observation: AuctionGlobalObservation, tracker: AuctionTracker):
        round_surplus = 0
        round_quantity = 0
        agent_surpluses = {}
        logger.info(f"Processing {len(global_observation.all_trades)} trades")
        for trade in global_observation.all_trades:
            buyer = self.agent_dict.get(trade.buyer_id)
            seller = self.agent_dict.get(trade.seller_id)

            if not buyer or not seller:
                logger.warning(f"Skipping trade: buyer or seller not found. Buyer ID: {trade.buyer_id}, Seller ID: {trade.seller_id}")
                continue

            try:
                buyer_value = buyer.economic_agent.value_schedules[self.config.agent_config.good_name].get_value(trade.quantity)
                seller_cost = seller.economic_agent.cost_schedules[self.config.agent_config.good_name].get_value(trade.quantity)

                logger.info(f"Buyer value: {buyer_value}, Seller cost: {seller_cost}, Trade price: {trade.price}")

                if trade.price > buyer_value or trade.price < seller_cost:
                    logger.warning(f"Skipping invalid trade: price={trade.price}, buyer_value={buyer_value}, seller_cost={seller_cost}")
                    continue

                logger.info(f"Processing trade between buyer {buyer.index} and seller {seller.index}")

                buyer_utility_before = buyer.economic_agent.calculate_utility(buyer.economic_agent.endowment.current_basket)
                seller_utility_before = seller.economic_agent.calculate_utility(seller.economic_agent.endowment.current_basket)

                logger.info(f"Buyer utility before: {buyer_utility_before}, Seller utility before: {seller_utility_before}")

                try:
                    buyer.economic_agent.process_trade(trade)
                except ValueError as e:
                    logger.warning(f"Error processing trade for buyer: {e}")
                    continue

                try:
                    seller.economic_agent.process_trade(trade)
                except ValueError as e:
                    logger.warning(f"Error processing trade for seller: {e}")
                    # Revert buyer's trade processing if seller fails
                    buyer.economic_agent.revert_trade(trade)
                    continue

                buyer_utility_after = buyer.economic_agent.calculate_utility(buyer.economic_agent.endowment.current_basket)
                seller_utility_after = seller.economic_agent.calculate_utility(seller.economic_agent.endowment.current_basket)

                logger.info(f"Buyer utility after: {buyer_utility_after}, Seller utility after: {seller_utility_after}")

                buyer_surplus = buyer_utility_after - buyer_utility_before
                seller_surplus = seller_utility_after - seller_utility_before

                logger.info(f"Buyer surplus: {buyer_surplus}, Seller surplus: {seller_surplus}")

                agent_surpluses[buyer.id] = agent_surpluses.get(buyer.id, 0) + buyer_surplus
                agent_surpluses[seller.id] = agent_surpluses.get(seller.id, 0) + seller_surplus

                trade_surplus = buyer_surplus + seller_surplus

                tracker.add_trade(trade)
                round_surplus += trade_surplus
                round_quantity += trade.quantity
                logger.info(f"Executed trade: {trade}")
                logger.info(f"Trade surplus: {trade_surplus}")

            except Exception as e:
                logger.error(f"Error processing trade: {e}")
                logger.exception("Exception details:")
                continue

        tracker.add_round_data(round_surplus, round_quantity)
        logger.info(f"Round summary - Surplus: {round_surplus}, Quantity: {round_quantity}")

        # Store agent_surpluses for use in update_simulation_state
        self.agent_surpluses = agent_surpluses

    def process_group_chat_messages(self, global_observation: GroupChatGlobalObservation, tracker: GroupChatTracker):
        for message in global_observation.all_messages:
            tracker.add_message(message)
        if global_observation.current_topic != self.current_topic:
            self.current_topic = global_observation.current_topic
            tracker.add_topic(self.current_topic)
        logger.info(f"Processed {len(global_observation.all_messages)} messages in group chat")

    def update_simulation_state(self, env_name: str, env_state: EnvironmentStep):
        if env_name.startswith('group_chat_batch_'):
            batch_index = int(env_name.split('_')[-1])
            batch_agents = self.agent_batches[batch_index]
            for agent in batch_agents:
                agent_observation = env_state.global_observation.observations.get(agent.id)
                if agent_observation:
                    agent.last_observation = agent_observation
                    agent.last_step = env_state

                    # Assuming group chat observation contains relevant state information
                    if hasattr(agent_observation.observation, 'model_dump'):
                        observation_dict = agent_observation.observation.model_dump()
                    else:
                        observation_dict = agent_observation.observation

                    # You can process the group chat observation here if needed
                else:
                    agent.last_perception = None  # Ensure it's set even if None
        elif env_name == 'auction':
            for agent in self.agents:
                agent_observation = env_state.global_observation.observations.get(agent.id)
                if agent_observation:
                    agent.last_observation = agent_observation
                    agent.last_step = env_state

                    # Convert agent_observation.observation to dict if necessary
                    if hasattr(agent_observation.observation, 'model_dump'):
                        observation_dict = agent_observation.observation.model_dump()
                    else:
                        observation_dict = agent_observation.observation

                    new_cash = observation_dict.get('endowment', {}).get('cash')
                    new_goods = observation_dict.get('endowment', {}).get('goods', {})

                    if new_cash is not None:
                        agent.economic_agent.endowment.current_basket.cash = new_cash

                    for good, quantity in new_goods.items():
                        agent.economic_agent.endowment.current_basket.update_good_quantity(good, quantity)
                else:
                    agent.last_perception = None  # Ensure it's set even if None

        if not self.simulation_data or 'state' not in self.simulation_data[-1]:
            self.simulation_data.append({'state': {}})

        self.simulation_data[-1]['state'][env_name] = env_state.info

    def insert_ai_requests(self, ai_requests):
        self.data_inserter.insert_ai_requests(ai_requests)

    async def generate_initial_topic(self) -> str:
        if self.topic_proposer is None:
            raise ValueError("topic_proposer is not initialized")
        
        # Reference a valid group_chat_batch environment, e.g., group_chat_batch_0
        if not self.agent_batches:
            raise ValueError("No agent batches available for group chat.")

        first_batch_env_name = "group_chat_batch_0" if len(self.agent_batches) > 0 else None
        if not first_batch_env_name or first_batch_env_name not in self.environments:
            raise ValueError("No valid group_chat_batch_x environment found for topic generation.")

        topic_action = await self.topic_proposer.generate_action(
            first_batch_env_name,  # Updated environment name
            "Consider recent economic events, market trends, or financial news."
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

    async def generate_new_topic_based_on_market(self) -> str:
        auction_env = self.environments.get('auction')
        if not auction_env:
            logger.error("Auction environment not found.")
            return "Default topic: Market Overview"

        # Get the latest global state
        latest_observation = auction_env.get_global_state()
        if not latest_observation:
            logger.warning("No global state available.")
            return "Initial market discussion"

        # Select a random batch
        batch_index = random.randint(0, len(self.agent_batches) - 1)
        group_chat_env_name = f"group_chat_batch_{batch_index}"

        if group_chat_env_name not in self.environments:
            logger.error(f"Group chat environment {group_chat_env_name} not found.")
            return "General market discussion"

        # Select a random agent from the chosen batch as the topic proposer
        topic_proposer = random.choice(self.agent_batches[batch_index])

        # Prepare the prompt with the latest market data
        prompt = (
            f"Consider the following market data:\n{latest_observation}\n\n"
            f"Based on this information and the current topic '{self.current_topic}', "
            "propose an interesting and relevant topic for the next round of market discussion. "
            "The topic should be related to the current market conditions, trends, or potential strategies. "
            "Suggest a single, concise topic (1-2 sentences) that captures an important "
            "aspect of the current market situation or a relevant economic concept."
        )

        # Use the topic proposer's generate_action method
        topic_action = await topic_proposer.generate_action(
            group_chat_env_name,
            prompt
        )

        logger.debug(f"Topic proposer {topic_proposer.id} generated action: {topic_action}")

        try:
            if isinstance(topic_action, dict):
                if 'content' in topic_action:
                    if isinstance(topic_action['content'], dict) and 'action' in topic_action['content']:
                        new_topic = topic_action['content']['action']['content']
                    else:
                        new_topic = topic_action['content']
                elif 'action' in topic_action:
                    new_topic = topic_action['action']['content']
                else:
                    raise ValueError("Unexpected topic_action structure")
            else:
                raise ValueError("topic_action is not a dictionary")
        except Exception as e:
            logger.error(f"Invalid topic action structure: {e}")
            new_topic = "General market trends and strategies"

        logger.info(f"Generated new topic based on market: {new_topic}")
        return new_topic

    async def run_simulation(self):
        log_section(logger, "SIMULATION COMMENCING")

        # Initialize the first topic
        initial_topic = await self.generate_initial_topic()
        self.current_topic = initial_topic

        # Limit the number of concurrent group chat tasks to prevent resource exhaustion
        semaphore = asyncio.Semaphore(100)  # Adjust based on system capabilities

        async def run_group_chat(env_name: str, round_num: int, sub_round_num: int, batch: List[MarketAgent]):
            async with semaphore:
                await self.run_environment(env_name, round_num, sub_round_num=sub_round_num, batch=batch)

        try:
            for round_num in range(1, self.config.max_rounds + 1):
                log_round(logger, round_num)

                # Run multiple sub-rounds within group chats
                for sub_round in range(1, self.sub_rounds_per_group_chat + 1):
                    group_chat_tasks = []
                    for batch_index, batch in enumerate(self.agent_batches):
                        group_chat_env_name = f"group_chat_batch_{batch_index}"
                        task = asyncio.create_task(
                            run_group_chat(
                                env_name=group_chat_env_name,
                                round_num=round_num,
                                sub_round_num=sub_round,
                                batch=batch
                            )
                        )
                        group_chat_tasks.append(task)
                    await asyncio.gather(*group_chat_tasks)

                # After group chats, run auction
                env_name = 'auction'
                env_state = await self.run_environment(env_name, round_num)
                self.update_simulation_state(env_name, env_state)

                # Generate a new topic based on auction results
                new_topic = await self.generate_new_topic_based_on_market()
                self.simulation_data[-1]['new_topic'] = new_topic
                logger.info(f"New topic for next round: {new_topic}")
                self.current_topic = new_topic

                # Update all group_chat_batch_x environments with the new topic
                for batch_index, batch in enumerate(self.agent_batches):
                    group_chat_env_name = f"group_chat_batch_{batch_index}"
                    self.environments[group_chat_env_name].mechanism.current_topic = self.current_topic
                    logger.info(f"Updated topic for {group_chat_env_name}: {self.current_topic}")

                # Run reflections for all environments (only auction in this case)
                reflect_prompts = await self.run_parallel_reflect('auction')
                if reflect_prompts:
                    reflections = await self.run_parallel_ai_completion(reflect_prompts)
                    agents_with_observations = [a for a in self.agents if a.last_observation]
                    for agent, reflection in zip(agents_with_observations, reflections):
                        if reflection.json_object:
                            log_reflection(logger, agent.index, f"{Fore.MAGENTA}{reflection.json_object.object}{Style.RESET_ALL}")
                            # Handle reflection content based on environment
                            if 'auction' in reflection.json_object.object:
                                environment_reward = self.agent_surpluses.get(agent.id, 0.0)
                                self_reward = reflection.json_object.object.get("self_reward", 0.0)

                                # Normalize environment_reward
                                normalized_environment_reward = environment_reward / (1 + abs(environment_reward))
                                normalized_environment_reward = max(0.0, min(normalized_environment_reward, 1.0))

                                # Weighted average of normalized_environment_reward and self_reward
                                total_reward = normalized_environment_reward * 0.5 + self_reward * 0.5

                                # Add logging for rewards
                                logger.info(
                                    f"Agent {agent.index} rewards - Environment Reward: {environment_reward}, "
                                    f"Normalized Environment Reward: {normalized_environment_reward}, "
                                    f"Self Reward: {self_reward}, Total Reward: {total_reward}"
                                )
                                agent.memory.append({
                                    "type": "reflection",
                                    "content": reflection.json_object.object.get("reflection", ""),
                                    "strategy_update": reflection.json_object.object.get("strategy_update", ""),
                                    "observation": agent.last_observation,
                                    "environment_reward": round(environment_reward, 4),
                                    "self_reward": round(self_reward, 4),
                                    "total_reward": round(total_reward, 4),
                                    "timestamp": datetime.now().isoformat()
                                })
                            elif 'group_chat' in reflection.json_object.object:
                                agent.memory.append({
                                    "type": "reflection",
                                    "content": reflection.json_object.object.get("reflection", ""),
                                    "strategy_update": reflection.json_object.object.get("strategy_update", ""),
                                    "observation": agent.last_observation,
                                    "timestamp": datetime.now().isoformat()
                                })
                        else:
                            logger.warning(f"No reflection JSON object for agent {agent.index} in 'auction'")
                else:
                    logger.info("No reflections generated for 'auction' this round.")

                self.save_round_data(round_num)
                self.save_agent_interactions(round_num)

                # Insert data after each round
                try:
                    self.data_inserter.insert_round_data(
                        round_num, 
                        self.agents, 
                        self.environments, 
                        self.config,
                        self.trackers
                    )
                    logger.info(f"Data for round {round_num} inserted successfully.")
                except Exception as e:
                    logger.error(f"Error inserting data for round {round_num}: {str(e)}")

        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            logger.exception("Exception details:")
        finally:
            log_completion(logger, "SIMULATION COMPLETED")
            self.print_summary()

    def save_round_data(self, round_num):
        round_data = {
            'round': round_num,
            'agent_states': [{
                'id': agent.id,
                'is_buyer': agent.role == "buyer",
                'cash': agent.economic_agent.endowment.current_basket.cash,
                'goods': agent.economic_agent.endowment.current_basket.goods_dict,
                'last_action': agent.last_action,
                'memory': agent.memory[-1] if agent.memory else None
            } for agent in self.agents],
            'environment_states': {name: env.get_global_state() for name, env in self.environments.items()},
        }

        if self.simulation_data and 'state' in self.simulation_data[-1]:
            round_data['state'] = self.simulation_data[-1]['state']

        self.simulation_data.append(round_data)

    def save_agent_interactions(self, round_num):
        """Save interactions for all agents for the current round."""
        self.log_folder.mkdir(parents=True, exist_ok=True)

        for agent in self.agents:
            file_path = self.log_folder / f"agent_{agent.index}_interactions.jsonl"
            with open(file_path, 'a') as f:
                # Process all interactions
                for interaction in agent.interactions:
                    interaction_with_round = {
                        "round": round_num,
                        **interaction
                    }
                    json.dump(interaction_with_round, f, default=str)
                    f.write('\n')

            # Clear all interactions after saving
            agent.interactions.clear()

        logger.info(f"Saved agent interactions for round {round_num} to {self.log_folder}")

    def print_summary(self):
        log_section(logger, "SIMULATION SUMMARY")

        # Auction Summary
        if 'auction' in self.trackers:
            auction_tracker: AuctionTracker = self.trackers['auction']
            total_buyer_surplus = sum(agent.economic_agent.calculate_individual_surplus() for agent in self.agents if agent.role == "buyer")
            total_seller_surplus = sum(agent.economic_agent.calculate_individual_surplus() for agent in self.agents if agent.role == "seller")
            total_empirical_surplus = total_buyer_surplus + total_seller_surplus

            print(f"Total Empirical Buyer Surplus: {total_buyer_surplus:.2f}")
            print(f"Total Empirical Seller Surplus: {total_seller_surplus:.2f}")
            print(f"Total Empirical Surplus: {total_empirical_surplus:.2f}")

            env = self.environments['auction']
            global_state = env.get_global_state()
            equilibria = global_state.get('equilibria', {})

            if equilibria:
                theoretical_total_surplus = sum(data['total_surplus'] for data in equilibria.values())
                print(f"\nTheoretical Total Surplus: {theoretical_total_surplus:.2f}")

                efficiency = (total_empirical_surplus / theoretical_total_surplus) * 100 if theoretical_total_surplus > 0 else 0
                print(f"\nEmpirical Efficiency: {efficiency:.2f}%")
            else:
                print("\nTheoretical equilibrium data not available.")

            summary = auction_tracker.get_summary()
            print(f"\nAuction Environment:")
            print(f"Total number of trades: {summary['total_trades']}")
            print(f"Total surplus: {summary['total_surplus']:.2f}")
            print(f"Total quantity traded: {summary['total_quantity']}")

        # Group Chat Summary
        group_chat_total_messages = 0
        group_chat_total_topics = 0
        for env_name, tracker in self.trackers.items():
            if env_name.startswith('group_chat_batch_'):
                group_chat_total_messages += tracker.get_summary()['total_messages']
                group_chat_total_topics += tracker.get_summary()['total_topics']

        if group_chat_total_messages > 0 or group_chat_total_topics > 0:
            print("\nGroup Chat Summary:")
            print(f"Total messages: {group_chat_total_messages}")
            print(f"Total topics discussed: {group_chat_total_topics}")

        print("\nFinal Agent States:")
        for agent in self.agents:
            print(f"Agent {agent.index} ({agent.role}):")
            print(f"  Cash: {agent.economic_agent.endowment.current_basket.cash:.2f}")
            print(f"  Goods: {agent.economic_agent.endowment.current_basket.goods_dict}")
            surplus = agent.economic_agent.calculate_individual_surplus()
            print(f"  Individual Surplus: {surplus:.2f}")
            if agent.memory:
                print(f"  Last Reflection: {agent.memory[-1]['content']}")
            print()

    async def start(self):
        print_ascii_art()
        log_section(logger, "JOINT ORCHESTRATOR INITIALIZING")
        self.generate_agents()
        self.setup_environments()
        self.setup_database()

        if self.dashboard:
            dashboard_thread = threading.Thread(target=self.run_dashboard)
            dashboard_thread.start()

        await self.run_simulation()

        if self.dashboard:
            dashboard_thread.join()


if __name__ == "__main__":
    config = load_config()
    orchestrator = Orchestrator(config)
    asyncio.run(orchestrator.start())