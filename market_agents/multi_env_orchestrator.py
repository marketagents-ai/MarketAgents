# multi_env_orchestrator.py

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
    setup_logger, print_ascii_art, log_section, log_agent_init, log_perception, log_action, log_reflection,
    log_round, log_environment_setup, log_completion, log_trade
)

import warnings
warnings.filterwarnings("ignore", module="pydantic")

# Set up logging using logger_utils
logger = setup_logger(__name__)

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
    # Removed group_size as we are using a single batch


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
        # Removed self.simulation_order as it's no longer needed for multiple environments
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
        # Removed batching attributes
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
    
    # Removed batch_agents method as batching is no longer needed

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
            good_name = agent_config.get('good_name', 'apple')
            
            if is_buyer:
                initial_cash = agent_config.get('buyer_initial_cash', 1000)
                initial_goods_quantity = agent_config.get('buyer_initial_goods', 0)
                base_value = agent_config.get('buyer_base_value', 120.0)
                value_schedule = BuyerPreferenceSchedule(
                    num_units=agent_config.get('num_units', 10),
                    base_value=base_value,
                    noise_factor=agent_config.get('noise_factor', 0.05)
                )
                value_schedules = {good_name: value_schedule}
                cost_schedules = {}
            else:
                initial_cash = agent_config.get('seller_initial_cash', 0)
                initial_goods_quantity = agent_config.get('seller_initial_goods', 10)
                base_value = agent_config.get('seller_base_value', 80.0)
                cost_schedule = SellerPreferenceSchedule(
                    num_units=agent_config.get('num_units', 10),
                    base_value=base_value,
                    noise_factor=agent_config.get('noise_factor', 0.05)
                )
                value_schedules = {}
                cost_schedules = {good_name: cost_schedule}

            # Create initial basket and endowment
            initial_basket = Basket(
                cash=initial_cash,
                goods=[Good(name=good_name, quantity=initial_goods_quantity)]
            )
            endowment = Endowment(
                initial_basket=initial_basket,
                agent_id=agent_uuid
            )

            # Create economic agent
            economic_agent = EconomicAgent(
                id=agent_uuid,
                endowment=endowment,
                value_schedules=value_schedules,
                cost_schedules=cost_schedules,
                max_relative_spread=agent_config.get('max_relative_spread', 0.2)
            )

            # Create MarketAgent
            agent = MarketAgent.create(
                agent_id=agent_uuid,
                use_llm=agent_config.get('use_llm', True),
                llm_config=llm_config,
                environments=self.environments,
                protocol=ACLMessage,
                persona=persona,
                econ_agent=economic_agent
            )

            # Initialize agent attributes
            agent.last_perception = None
            agent.last_observation = None
            agent.last_step = None
            agent.index = i

            # Append to agents list and dictionary
            self.agents.append(agent)
            self.agent_dict[agent.id] = agent

            # Log agent initialization using log_agent_init
            log_agent_init(logger, agent.index, is_buyer, persona)

        # Initialize topic_proposer
        self.topic_proposer = random.choice(self.agents)
        self.topic_proposer.system = "You are the group chat topic proposer agent. Your role is to propose interesting and relevant topics for the group discussion."
        logger.info(f"Topic proposer initialized: Agent {self.topic_proposer.id} with role: {'Buyer' if self.topic_proposer.role == 'buyer' else 'Seller'}")

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

        # Create a single GroupChat Environment with all agents
        group_chat_config = self.config.environment_configs.get('group_chat')
        if group_chat_config:
            group_chat = GroupChat(
                max_rounds=group_chat_config.max_rounds,
                current_topic=group_chat_config.initial_topic,  # Fixed here
                speaker_order=[str(agent.id) for agent in self.agents],
                sequential=False,
                sub_rounds=group_chat_config.sub_rounds
            )
            group_chat_env_name = "group_chat_single_batch"
            group_chat_env = MultiAgentEnvironment(
                name=group_chat_config.name,
                address=group_chat_config.address,
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
            logger.info("Database tables created.")
        else:
            logger.info("Required tables already exist.")
        log_completion(logger, "Database setup completed")

    async def run_parallel_ai_completion(self, prompts: List[LLMPromptContext]) -> List[LLMOutput]:
        # Implement batching within AI completions if necessary
        results = await self.ai_utils.run_parallel_ai_completion(prompts, update_history=False)
        # Insert AI requests into the database
        ai_requests = self.ai_utils.get_all_requests()
        self.data_inserter.insert_ai_requests(ai_requests)
        return results

    async def run_parallel_perceive(self, env_name: str, agents_subset: List[MarketAgent] = None) -> List[LLMPromptContext]:
        perceive_prompts = []
        if env_name == 'group_chat_single_batch':
            # Perceive for all agents in the single group chat
            for agent in self.agents:
                perceive_prompt = await agent.perceive(env_name, return_prompt=True)
                perceive_prompts.append(perceive_prompt)
        elif env_name == 'auction':
            # Perceive for all agents in the auction
            for agent in self.agents:
                perceive_prompt = await agent.perceive(env_name, return_prompt=True)
                perceive_prompts.append(perceive_prompt)
        else:
            logger.error(f"Unknown environment: {env_name}")
            raise ValueError(f"Unknown environment: {env_name}")
        return perceive_prompts

    async def run_parallel_generate_action(self, env_name: str, perceptions: List[str]) -> List[LLMPromptContext]:
        action_prompts = []
        if env_name in ['group_chat_single_batch', 'auction']:
            for agent, perception in zip(self.agents, perceptions):
                action_prompt = await agent.generate_action(env_name, perception, return_prompt=True)
                action_prompts.append(action_prompt)
        else:
            logger.error(f"Unknown environment: {env_name}")
            raise ValueError(f"Unknown environment: {env_name}")
        return action_prompts

    async def run_parallel_reflect(self, env_name: str) -> List[LLMPromptContext]:
        reflect_prompts = []
        for agent in self.agents:
            if agent.last_observation:
                reflect_prompt = await agent.reflect(env_name, return_prompt=True)
                reflect_prompts.append(reflect_prompt)
            else:
                log_skipped(logger, f"Skipping reflection for agent {agent.index} due to no observation")
        return reflect_prompts

    def set_agent_system_messages(self, env_name: str, round_num: int, sub_round_num: int = None, **kwargs):
        if env_name == 'group_chat_single_batch':
            if sub_round_num:
                for agent in self.agents:
                    agent.system = (
                        f"You are participating in sub-round {sub_round_num} of round {round_num} "
                        f"in a group chat about '{kwargs.get('current_topic', 'various topics')}'. "
                        f"Your role is {'buyer' if agent.role == 'buyer' else 'seller'}. "
                        "Engage in the discussion and share your insights."
                    )
            else:
                for agent in self.agents:
                    agent.system = (
                        f"You are participating in round {round_num} "
                        f"in a group chat about '{kwargs.get('current_topic', 'various topics')}'. "
                        f"Your role is {'buyer' if agent.role == 'buyer' else 'seller'}. "
                        "Engage in the discussion and share your insights."
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

    async def run_environment(self, env_name: str, round_num: int, sub_round_num: int = None) -> EnvironmentStep:
        env = self.environments[env_name]
        tracker = self.trackers[env_name]

        log_environment_setup(logger, env_name)
        if sub_round_num:
            logger.info(f"Sub-round {sub_round_num}")
        
        if env_name == 'auction':
            # Reset pending orders at the beginning of the auction round
            for agent in self.agents:
                agent.economic_agent.reset_all_pending_orders()

        # Set system messages for agents based on environment
        if env_name == 'group_chat_single_batch':
            self.set_agent_system_messages(env_name, round_num, sub_round_num=sub_round_num, current_topic=self.current_topic)
        elif env_name == 'auction':
            good_name = env.mechanism.good_name
            self.set_agent_system_messages(env_name, round_num, good_name=good_name)

        # Run parallel perceive
        perception_prompts = await self.run_parallel_perceive(env_name)
        perceptions = await self.run_parallel_ai_completion(perception_prompts)
        perceptions_map = {perception.source_id: perception for perception in perceptions}

        if env_name == 'group_chat_single_batch':
            for agent in self.agents:
                perception = perceptions_map.get(agent.id)
                if perception:
                    log_section(logger, agent.persona)
                    perception_content = perception.json_object.object if perception.json_object else perception.str_content
                    log_perception(logger, agent.index, json.dumps(perception_content, indent=2))
                    agent.last_perception = perception_content
                else:
                    log_skipped(logger, f"No perception found for agent {agent.index} in {env_name}")
                    agent.last_perception = None
        elif env_name == 'auction':
            for agent in self.agents:
                perception = perceptions_map.get(agent.id)
                if perception:
                    log_section(logger, agent.persona)
                    if env_name == 'auction':
                        perception_content = perception.json_object.object if perception.json_object else perception.str_content
                        log_perception(logger, agent.index, json.dumps(perception_content, indent=2))
                        agent.last_perception = perception_content
                else:
                    log_skipped(logger, f"No perception found for agent {agent.index} in {env_name}")
                    agent.last_perception = None

        # Extract perception content to pass to generate_action
        perception_contents = [agent.last_perception or "" for agent in self.agents]

        # Run parallel generate_action
        action_prompts = await self.run_parallel_generate_action(env_name, perception_contents)

        actions = await self.run_parallel_ai_completion(action_prompts)
        actions_map = {action.source_id: action for action in actions}

        if env_name == 'group_chat_single_batch':
            global_action = self.process_group_chat_actions(actions_map)
        elif env_name == 'auction':
            global_action = self.process_auction_actions(actions_map, env)

        try:
            env_state = env.step(global_action)
        except Exception as e:
            log_section(logger, f"Error in environment {env_name}: {str(e)}")
            raise e  # Re-raise the exception to be caught in run_simulation

        logger.info(f"Completed {env_name} step")

        if env_name == 'group_chat_single_batch' and isinstance(env_state.global_observation, GroupChatGlobalObservation):
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
                agent = next((a for a in self.agents if a.id == agent_id), None)
                if agent:
                    log_action(logger, agent.index, f"Message: {group_chat_message.content}")
                else:
                    log_skipped(logger, f"Unknown agent {agent_id} attempted to send a message.")
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
                    action_content['quantity'] = 1  # Ensuring quantity is 1 as per original code

                    agent = self.agent_dict.get(agent_id)
                    if not agent:
                        log_skipped(logger, f"Agent with ID {agent_id} not found.")
                        continue

                    if agent.role == "buyer":
                        auction_action = Bid(**action_content)
                    elif agent.role == "seller":
                        auction_action = Ask(**action_content)
                    else:
                        raise ValueError(f"Invalid agent role: {agent.role}")

                    agent_actions[agent_id] = AuctionAction(agent_id=agent_id, action=auction_action)
                    agent.economic_agent.pending_orders.setdefault(good_name, []).append(auction_action)

                    action_type = "Bid" if isinstance(auction_action, Bid) else "Ask"
                    log_action(logger, agent.index, f"{action_type}: {auction_action}")
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
                log_skipped(logger, f"Skipping trade: buyer or seller not found. Buyer ID: {trade.buyer_id}, Seller ID: {trade.seller_id}")
                continue

            # Log roles of buyer and seller
            logger.info(f"Trade Details: Trade ID: {trade.trade_id}, Buyer ID: {buyer.id}, Role: {buyer.role}, Seller ID: {seller.id}, Role: {seller.role}")

            if buyer.role != "buyer" or seller.role != "seller":
                logger.error(f"Agent roles mismatch: Buyer ID: {buyer.id} is '{buyer.role}', Seller ID: {seller.id} is '{seller.role}'")
                raise ValueError(f"Agent is neither a buyer nor a seller for trade {trade}")

            try:
                buyer_value = buyer.economic_agent.value_schedules[self.config.agent_config.good_name].get_value(trade.quantity)
                seller_cost = seller.economic_agent.cost_schedules[self.config.agent_config.good_name].get_value(trade.quantity)

                logger.info(f"Buyer value: {buyer_value}, Seller cost: {seller_cost}, Trade price: {trade.price}")

                if trade.price > buyer_value or trade.price < seller_cost:
                    log_skipped(logger, f"Skipping invalid trade: price={trade.price}, buyer_value={buyer_value}, seller_cost={seller_cost}")
                    continue

                logger.info(f"Processing trade between Buyer {buyer.index} and Seller {seller.index}")

                buyer_utility_before = buyer.economic_agent.calculate_utility(buyer.economic_agent.endowment.current_basket)
                seller_utility_before = seller.economic_agent.calculate_utility(seller.economic_agent.endowment.current_basket)

                logger.info(f"Buyer utility before: {buyer_utility_before}, Seller utility before: {seller_utility_before}")

                buyer.economic_agent.process_trade(trade)
                seller.economic_agent.process_trade(trade)

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
                log_trade(logger, buyer.index, seller.index, self.config.agent_config.good_name, trade.price)

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
            agent = next((a for a in self.agents if a.id == message.agent_id), None)
            if agent:
                log_action(logger, agent.index, f"Group Chat Message: {message.content}")
            else:
                log_skipped(logger, f"Unknown agent {message.agent_id} sent a message.")
        if global_observation.current_topic != self.current_topic:
            self.current_topic = global_observation.current_topic
            tracker.add_topic(self.current_topic)
            logger.info(f"New topic introduced: {self.current_topic}")
        logger.info(f"Processed {len(global_observation.all_messages)} messages in group chat")

    def update_simulation_state(self, env_name: str, env_state: EnvironmentStep):
        if env_name == 'group_chat_single_batch':
            for agent in self.agents:
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

    async def generate_topic_for_group_chat(self, topic_proposer: MarketAgent, round_num: int) -> str:
        group_chat_env_name = "group_chat_single_batch"
        
        # Get the good name from the auction config
        auction_config = self.config.environment_configs.get('auction')
        if not auction_config:
            logger.error("Auction configuration not found.")
            good_name = "unknown good"
        else:
            good_name = auction_config.good_name
        
        # Check if market data is available (i.e., after the first round)
        auction_env = self.environments.get('auction')
        if auction_env and round_num > 1:
            latest_observation = auction_env.get_global_state()
            if not latest_observation:
                log_skipped(logger, "No global state available for auction; reverting to a general topic prompt.")
                latest_observation = f"General market trends and strategies for the {good_name} market"
            
            # Market-based prompt for subsequent rounds
            prompt = (
                f"Consider the following market data for the {good_name} market:\n{latest_observation}\n\n"
                f"Based on this information and the current topic '{self.current_topic}', "
                f"propose an interesting and relevant topic for the next round of {good_name} market discussion. "
                "The topic should be related to the current market conditions, trends, or potential strategies. "
                "Suggest a single, concise topic (1-2 sentences) that captures an important aspect "
                f"of the current {good_name} market situation or a relevant economic concept."
            )
        else:
            # General prompt for the first round (or if no market data is available)
            prompt = (
                f"Consider recent economic events, market trends, or financial news related to the {good_name} market "
                "to propose a relevant discussion topic. Focus on aspects that might affect supply, demand, "
                f"or pricing of {good_name}s in the upcoming auction rounds."
            )
        
        # Generate the topic using the topic proposer
        topic_action = await topic_proposer.generate_action(group_chat_env_name, prompt)
    
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
            new_topic = f"Default topic for discussion: General {good_name} market discussion"

        # Colored print for the new topic
        print(f"{Fore.CYAN}Generated new topic:{Fore.YELLOW} {new_topic}{Style.RESET_ALL}")
        logger.info(f"Generated new topic: {new_topic}")
        return new_topic

    async def run_simulation(self):
        log_section(logger, "SIMULATION COMMENCING")

        # Limit the number of concurrent group chat tasks to prevent resource exhaustion
        semaphore = asyncio.Semaphore(100)  # Adjust based on system capabilities

        async def run_group_chat(env_name: str, round_num: int, sub_round_num: int):
            async with semaphore:
                await self.run_environment(env_name, round_num, sub_round_num=sub_round_num)

        try:
            for round_num in range(1, self.config.max_rounds + 1):
                log_round(logger, round_num)
            
                # Generate a topic for the group chat
                new_topic = await self.generate_topic_for_group_chat(self.topic_proposer, round_num)
                
                # Update the environment with the new topic
                self.current_topic = new_topic
                self.environments["group_chat_single_batch"].mechanism.current_topic = new_topic
                logger.info(f"Updated topic for group_chat_single_batch: {new_topic}")

                # Run multiple sub-rounds within the group chat
                for sub_round in range(1, self.config.environment_configs['group_chat'].sub_rounds + 1):
                    task = asyncio.create_task(
                        run_group_chat(
                            env_name="group_chat_single_batch",
                            round_num=round_num,
                            sub_round_num=sub_round
                        )
                    )
                    await task  # Sequential execution; use gather if parallelism is desired

                # After group chats, run auction
                env_name = 'auction'
                env_state = await self.run_environment(env_name, round_num)
                self.update_simulation_state(env_name, env_state)

               # Run reflections for all environments (only auction in this case)
                reflect_prompts = await self.run_parallel_reflect('auction')
                if reflect_prompts:
                    reflections = await self.run_parallel_ai_completion(reflect_prompts)
                    agents_with_observations = [a for a in self.agents if a.last_observation]
                    for agent, reflection in zip(agents_with_observations, reflections):
                        if reflection.json_object:
                            log_reflection(logger, agent.index, json.dumps(reflection.json_object.object))
                            agent.memory.append({
                                "type": "reflection",
                                "content": reflection.json_object.object.get("reflection", ""),
                                "strategy_update": reflection.json_object.object.get("strategy_update", ""),
                                "observation": agent.last_observation,
                                "timestamp": datetime.now().isoformat()
                            })
                        else:
                            log_skipped(logger, f"No reflection JSON object for agent {agent.index} in 'auction'")
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
                        self.config
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
            if env_name == 'group_chat_single_batch':
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
