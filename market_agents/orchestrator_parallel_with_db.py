# orchestrator_parallel_with_db.py

import asyncio
import json
import logging
import os
import random
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml
from colorama import Fore, Style
from market_agents.agents.db.setup_database import create_database, create_tables
from market_agents.agents.market_agent import MarketAgent
from market_agents.agents.personas.persona import Persona, generate_persona, save_persona_to_file
from market_agents.agents.protocols.acl_message import ACLMessage
from market_agents.economics.econ_agent import EconomicAgent
from market_agents.economics.econ_models import Ask, Basket, Bid, BuyerPreferenceSchedule, Endowment, Good, SellerPreferenceSchedule, Trade
from market_agents.environments.environment import EnvironmentStep, MultiAgentEnvironment
from market_agents.environments.mechanisms.auction import (AuctionAction,
                                                           AuctionActionSpace,
                                                           AuctionGlobalObservation,
                                                           AuctionObservationSpace,
                                                           DoubleAuction,
                                                           GlobalAuctionAction)
from market_agents.inference.message_models import LLMConfig, LLMOutput, LLMPromptContext
from market_agents.inference.parallel_inference import (ParallelAIUtilities,
                                                        RequestLimits)
from market_agents.insert_simulation_data import SimulationDataInserter
from market_agents.logger_utils import *
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Set up logging for this module
logger = logging.getLogger(__name__)
logger.handlers = []
logger.addHandler(logging.NullHandler())

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


class LLMConfig(BaseModel):
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
    llm_configs: List[LLMConfig]
    environment_configs: Dict[str, AuctionConfig]
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


class Orchestrator:
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.agents: List[MarketAgent] = []
        self.environments: Dict[str, MultiAgentEnvironment] = {}
        self.dashboard = None
        self.database = None
        self.simulation_order = ['auction']
        self.simulation_data: List[Dict[str, Any]] = []
        self.latest_data = None
        self.trackers: Dict[str, AuctionTracker] = {}
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
        self.agent_surpluses = {}
        self.agent_dict: Dict[str, MarketAgent] = {}  # Mapping from agent IDs to agents

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

    def generate_agents(self):
        log_section(logger, "INITIALIZING MARKET AGENTS")
        personas = self.load_or_generate_personas()
        num_agents = len(personas)
        num_buyers = num_agents // 2
        num_sellers = num_agents - num_buyers

        for i, persona in enumerate(personas):
            agent_uuid = str(uuid.uuid4())
            llm_config = random.choice(self.config.llm_configs).dict()
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

            # Set additional attributes after creating the agent
            agent.task = None
            agent.tools = {}
            agent.output_format = {}
            agent.metadata = {}
            agent.interactions = []
            agent.system = None  # Will be set later
            agent.last_perception = None
            agent.last_observation = None
            agent.last_step = None
            agent.index = i

            self.agents.append(agent)
            self.agent_dict[agent.id] = agent
            log_agent_init(logger, agent.index, is_buyer, persona)

        # After creating all agents, set their system messages
        self.set_agent_system_messages(1, good_name)

    def setup_environments(self):
        log_section(logger, "CONFIGURING MARKET ENVIRONMENTS")
        for env_name, env_config in self.config.environment_configs.items():
            if env_name == 'auction':
                double_auction = DoubleAuction(
                    max_rounds=env_config.max_rounds,
                    good_name=env_config.good_name
                )
                env = MultiAgentEnvironment(
                    name=env_config.name,
                    address=env_config.address,
                    max_steps=env_config.max_rounds,
                    action_space=AuctionActionSpace(),
                    observation_space=AuctionObservationSpace(),
                    mechanism=double_auction
                )
                self.environments[env_name] = env
                self.trackers[env_name] = AuctionTracker()
                log_environment_setup(logger, env_name)
            else:
                logger.warning(f"Unknown environment type: {env_name}")

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
        results = await self.ai_utils.run_parallel_ai_completion(prompts, update_history=False)
        # Insert AI requests into the database
        ai_requests = self.ai_utils.get_all_requests()
        self.data_inserter.insert_ai_requests(ai_requests)
        return results

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

    async def run_parallel_reflect(self, env_name: str) -> List[LLMPromptContext]:
        reflect_prompts = []
        for agent in self.agents:
            if agent.last_observation:
                # Ensure agent.last_step is set
                environment = self.environments[env_name]
                last_step = environment.history.steps[-1][1] if environment.history.steps else None
                agent.last_step = last_step
                reflect_prompt = await agent.reflect(env_name, return_prompt=True)
                reflect_prompts.append(reflect_prompt)
            else:
                logger.info(f"Skipping reflection for agent {agent.index} due to no observation")
        return reflect_prompts

    async def run_environment(self, env_name: str, round_num: int) -> EnvironmentStep:
        import traceback
        env = self.environments[env_name]
        tracker = self.trackers[env_name]

        log_running(logger, env_name)

        # Reset pending orders at the beginning of the round
        for agent in self.agents:
            agent.economic_agent.reset_all_pending_orders()

        # Set system messages for agents
        self.set_agent_system_messages(round_num, env.mechanism.good_name)

        # Run parallel perceive
        perception_prompts = await self.run_parallel_perceive(env_name)
        perceptions = await self.run_parallel_ai_completion(perception_prompts)
        perceptions_map = {perception.source_id: perception for perception in perceptions}

        for agent in self.agents:
            perception = perceptions_map.get(agent.id)
            if perception:
                log_section(logger, f"Current Agent:\nAgent {agent.index} with persona:\n{agent.persona}")
                log_perception(logger, agent.index, f"{Fore.CYAN}{perception.json_object.object if perception.json_object else perception.str_content}{Style.RESET_ALL}")
                # Store the perception content
                agent.last_perception = perception.json_object.object if perception.json_object else None

            else:
                logger.warning(f"No perception found for agent {agent.index}")
                agent.last_perception = None  # Ensure it's set even if None

        # Extract perception content to pass to generate_action
        perception_contents = []
        for agent in self.agents:
            perception_content = agent.last_perception if agent.last_perception else ""
            perception_contents.append(perception_content)

        # Run parallel generate_action
        action_prompts = await self.run_parallel_generate_action(env_name, perception_contents)
        actions = await self.run_parallel_ai_completion(action_prompts)
        actions_map = {action.source_id: action for action in actions}

        agent_actions = {}
        for agent in self.agents:
            action = actions_map.get(agent.id)
            if action:
                try:
                    action_content = action.json_object.object if action.json_object else json.loads(action.str_content or '{}')
                    agent.last_action = action_content
                    if 'price' in action_content and 'quantity' in action_content:
                        action_content['quantity'] = 1

                        if agent.role == "buyer":
                            auction_action = Bid(**action_content)
                        elif agent.role == "seller":
                            auction_action = Ask(**action_content)
                        else:
                            raise ValueError(f"Invalid agent role: {agent.role}")

                        agent_actions[agent.id] = AuctionAction(agent_id=agent.id, action=auction_action)

                        # Update agent's pending orders
                        good_name = env.mechanism.good_name
                        agent.economic_agent.pending_orders.setdefault(good_name, []).append(auction_action)

                        action_type = "Bid" if isinstance(auction_action, Bid) else "Ask"
                        color = Fore.BLUE if action_type == "Bid" else Fore.GREEN
                        log_action(logger, agent.index, f"{color}{action_type}: {auction_action}{Style.RESET_ALL}")
                    else:
                        raise ValueError(f"Invalid action content: {action_content}")
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.error(f"Error creating AuctionAction for agent {agent.index}: {str(e)}")
            else:
                logger.warning(f"No action found for agent {agent.index}")

        global_action = GlobalAuctionAction(actions=agent_actions)
        try:
            env_state = env.step(global_action)
        except Exception as e:
            logger.error(f"Error in environment {env_name}: {str(e)}")
            traceback.print_exc()
            raise e  # Re-raise the exception to be caught in run_simulation

        logger.info(f"Completed {env_name} step")

        if isinstance(env_state.global_observation, AuctionGlobalObservation):
            self.process_trades(env_state.global_observation, tracker)

            # Include agent_surpluses in env_state.info
            env_state.info['agent_rewards'] = self.agent_surpluses

        return env_state

    def set_agent_system_messages(self, round_num: int, good_name: str):
        for agent in self.agents:
            current_cash = agent.economic_agent.current_cash
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

    def insert_ai_requests(self, ai_requests):
        self.data_inserter.insert_ai_requests(ai_requests)

    async def run_simulation(self):
        import traceback
        log_section(logger, "SIMULATION COMMENCING")

        try:
            for round_num in range(1, self.config.max_rounds + 1):
                log_round(logger, round_num)

                for env_name in self.simulation_order:
                    try:
                        env_state = await self.run_environment(env_name, round_num)
                        self.update_simulation_state(env_name, env_state)
                    except Exception as e:
                        logger.error(f"Error in environment {env_name}: {str(e)}")
                        traceback.print_exc()

                # Run parallel reflect
                reflect_prompts = await self.run_parallel_reflect(env_name)
                if reflect_prompts:
                    reflections = await self.run_parallel_ai_completion(reflect_prompts)
                    agents_with_observations = [a for a in self.agents if a.last_observation]
                    for agent, reflection in zip(agents_with_observations, reflections):
                        if reflection.json_object:
                            log_reflection(logger, agent.index, f"{Fore.MAGENTA}{reflection.json_object.object}{Style.RESET_ALL}")
                            # Access the surplus from agent's last_step.info
                            environment_reward = agent.last_step.info.get('agent_rewards', {}).get(agent.id, 0.0) if agent.last_step else 0.0
                            self_reward = reflection.json_object.object.get("self_reward", 0.0)
                            
                            # Normalize environment_reward
                            normalized_environment_reward = environment_reward / (1 + abs(environment_reward))
                            normalized_environment_reward = max(0.0, min(normalized_environment_reward, 1.0))
                            
                            # Weighted average of normalized environment_reward and self_reward
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
                        else:
                            logger.warning(f"No reflection JSON object for agent {agent.index}")
                else:
                    logger.info("No reflections generated this round.")

                self.save_round_data(round_num)
                self.save_agent_interactions(round_num)

                # Insert data after each round
                try:
                    self.data_inserter.insert_round_data(round_num, self.agents, self.agent_dict, self.trackers, self.config)
                    logger.info(f"Data for round {round_num} inserted successfully.")
                except Exception as e:
                    logger.error(f"Error inserting data for round {round_num}: {str(e)}")
        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
            traceback.print_exc()
        finally:
            log_completion(logger, "SIMULATION COMPLETED")
            self.print_summary()

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
            logger.info(f"Executed trade: {trade}")
            logger.info(f"Trade surplus: {trade_surplus}")

        tracker.add_round_data(round_surplus, round_quantity)
        logger.info(f"Round summary - Surplus: {round_surplus}, Quantity: {round_quantity}")

        # Store agent_surpluses for use in update_simulation_state
        self.agent_surpluses = agent_surpluses

    def update_simulation_state(self, env_name: str, env_state: EnvironmentStep):
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
        self.simulation_data[-1]['state']['trade_info'] = env_state.global_observation.all_trades

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

        for env_name, tracker in self.trackers.items():
            summary = tracker.get_summary()
            print(f"\nEnvironment: {env_name}")
            print(f"Total number of trades: {summary['total_trades']}")
            print(f"Total surplus: {summary['total_surplus']:.2f}")
            print(f"Total quantity traded: {summary['total_quantity']}")

        print("\nFinal Agent States:")
        for agent in self.agents:
            print(f"Agent {agent.index} ({agent.role}):")
            print(f"  Cash: {agent.economic_agent.endowment.current_basket.cash:.2f}")
            print(f"  Goods: {agent.economic_agent.endowment.current_basket.goods_dict}")
            surplus = agent.economic_agent.calculate_individual_surplus()
            print(f"  Individual Surplus: {surplus:.2f}")

    def run_dashboard(self):
        # Implement dashboard logic here
        pass

    async def start(self):
        log_section(logger, "MARKET SIMULATION INITIALIZING")
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
