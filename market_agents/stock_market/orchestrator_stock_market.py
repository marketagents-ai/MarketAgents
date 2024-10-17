# orchestrator_stock_market.py

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
from colorama import Fore, Style

import yaml
from market_agents.stock_market.setup_stock_database import create_database, create_tables
from market_agents.agents.market_agent import MarketAgent
from market_agents.agents.personas.persona import Persona, generate_persona, save_persona_to_file
from market_agents.agents.protocols.acl_message import ACLMessage
from market_agents.environments.environment import EnvironmentStep, MultiAgentEnvironment
from market_agents.stock_market.stock_models import (
    MarketAction,
    OrderType,
    Trade,
    Portfolio,
    Stock,
    Position,
    Endowment,
    StockOrder
)
from market_agents.stock_market.stock_agent import StockEconomicAgent
from market_agents.environments.mechanisms.stock_market import (
    StockMarket,
    StockMarketAction,
    StockMarketActionSpace,
    StockMarketObservationSpace,
    StockMarketMechanism,
    GlobalStockMarketAction,
    StockMarketGlobalObservation
)
from market_agents.inference.message_models import LLMConfig, LLMOutput, LLMPromptContext
from market_agents.inference.parallel_inference import ParallelAIUtilities, RequestLimits
from market_agents.stock_market.insert_stock_simulation_data import SimulationDataInserter
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
    initial_cash_min: float
    initial_cash_max: float
    initial_stocks_min: int
    initial_stocks_max: int
    risk_aversion: float
    expected_return: float
    use_llm: bool
    stock_symbol: str = "AAPL"
    max_relative_spread: float


class StockMarketConfig(BaseModel):
    name: str
    address: str
    max_rounds: int
    stock_symbol: str = "AAPL"


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
    environment_configs: Dict[str, StockMarketConfig]
    protocol: str
    database_config: DatabaseConfig = DatabaseConfig()

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')


def load_config(config_path: Path = Path("./market_agents/stock_market/orchestrator_config_stock.yaml")) -> OrchestratorConfig:
    with open(config_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return OrchestratorConfig(**yaml_data)


class Orchestrator:
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.agents: List[MarketAgent] = []
        self.environments: Dict[str, StockMarket] = {}
        self.dashboard = None
        self.database = None
        self.simulation_order = ['stock_market']
        self.simulation_data: List[Dict[str, Any]] = []
        self.latest_data = None
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

        agent_config = self.config.agent_config
        stock_symbol = agent_config.stock_symbol

        for i, persona in enumerate(personas):
            agent_uuid = str(uuid.uuid4())
            llm_config = random.choice(self.config.llm_configs).dict()

            # Randomize initial cash and stocks
            initial_cash = random.uniform(agent_config.initial_cash_min, agent_config.initial_cash_max)
            initial_stocks_quantity = random.randint(agent_config.initial_stocks_min, agent_config.initial_stocks_max)

            # Assume a base market price for initialization
            base_market_price = 150.0

            # Generate positions
            positions = []
            remaining_quantity = initial_stocks_quantity
            while remaining_quantity > 0:
                # Random quantity for the position (up to remaining quantity)
                position_quantity = random.randint(1, remaining_quantity)
                remaining_quantity -= position_quantity

                # Random purchase price (between base_market_price * 0.9 and base_market_price * 1.0)
                purchase_price = random.uniform(base_market_price * 0.9, base_market_price * 1.0)

                position = Position(quantity=position_quantity, purchase_price=purchase_price)
                positions.append(position)

            # Create Stock with positions
            initial_stock = Stock(symbol=stock_symbol, positions=positions)

            # Create initial portfolio and endowment
            initial_portfolio = Portfolio(cash=initial_cash, stocks=[initial_stock])
            endowment = Endowment(
                initial_portfolio=initial_portfolio,
                agent_id=agent_uuid
            )

            # **New Code Ends Here**

            # Create economic agent
            economic_agent = StockEconomicAgent(
                id=agent_uuid,
                endowment=endowment,
                max_relative_spread=agent_config.max_relative_spread,
                risk_aversion=agent_config.risk_aversion,
                expected_return=agent_config.expected_return,
                stock_symbol=stock_symbol
            )

            agent = MarketAgent.create(
                agent_id=agent_uuid,
                use_llm=agent_config.use_llm,
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
            log_agent_init(logger, agent.index, False, persona)

    def setup_environments(self):
        log_section(logger, "CONFIGURING MARKET ENVIRONMENTS")
        for env_name, env_config in self.config.environment_configs.items():
            if env_name == 'stock_market':
                stock_market_mechanism = StockMarketMechanism(
                    max_rounds=env_config.max_rounds,
                    stock_symbol=env_config.stock_symbol
                )
                env = StockMarket(
                    name=env_config.name,
                    address=env_config.address,
                    max_steps=env_config.max_rounds,
                    action_space=StockMarketActionSpace(),
                    observation_space=StockMarketObservationSpace(),
                    mechanism=stock_market_mechanism,
                    agents={agent.id: agent.economic_agent for agent in self.agents}
                )
                self.environments[env_name] = env
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
            environment = self.environments[env_name]
            last_step = environment.history.steps[-1][1] if environment.history.steps else None
            agent.last_step = last_step
            reflect_prompt = await agent.reflect(env_name, return_prompt=True)
            reflect_prompts.append(reflect_prompt)
        return reflect_prompts


    async def run_environment(self, env_name: str, round_num: int) -> EnvironmentStep:
        env = self.environments[env_name]
        log_running(logger, env_name)

        # Set system messages for agents
        self.set_agent_system_messages(round_num)

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
                agent.last_perception = perception.json_object.object if perception.json_object else json.loads(perception.str_content)
            else:
                logger.warning(f"No perception found for agent {agent.index}")
                agent.last_perception = None

        # Extract perception content to pass to generate_action
        perception_contents = [agent.last_perception for agent in self.agents]

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

                    order_type = OrderType(action_content.get('order_type', 'hold').lower())

                    if order_type == OrderType.HOLD or 'quantity' not in action_content:
                        market_action = MarketAction(order_type=OrderType.HOLD)
                    else:
                        if 'price' not in action_content:
                            raise ValueError(f"Missing price for {order_type.value} order: {action_content}")

                        price = float(action_content['price'])
                        quantity = int(action_content['quantity'])

                        if price <= 0 or quantity <= 0:
                            raise ValueError(f"Price and quantity must be positive for {order_type.value} order: price={price}, quantity={quantity}")

                        market_action = MarketAction(
                            order_type=order_type,
                            price=price,
                            quantity=quantity
                        )

                    stock_market_action = StockMarketAction(agent_id=agent.id, action=market_action)
                    agent_actions[agent.id] = stock_market_action

                    if market_action.order_type != OrderType.HOLD:
                        stock_order = StockOrder(agent_id=agent.id, **market_action.model_dump())
                        # Add the order to the agent's pending_orders
                        agent.economic_agent.pending_orders.append(stock_order)

                    color = Fore.BLUE if market_action.order_type == OrderType.BUY else Fore.GREEN if market_action.order_type == OrderType.SELL else Fore.YELLOW
                    log_action(logger, agent.index, f"{color}{market_action.order_type.value.capitalize()} Order: {market_action}{Style.RESET_ALL}")
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.error(f"Error creating StockMarketAction for agent {agent.index}: {str(e)}")
            else:
                logger.warning(f"No action found for agent {agent.index}")

        global_action = GlobalStockMarketAction(actions=agent_actions)
        env_state = env.step(global_action)

        # Update agent observations
        for agent in self.agents:
            agent_observation = env_state.global_observation.observations.get(agent.id)
            if agent_observation:
                agent.last_observation = agent_observation.observation
            else:
                agent.last_observation = None

        logger.info(f"Completed {env_name} step")

        return env_state

    def set_agent_system_messages(self, round_num: int):
        for agent in self.agents:
            current_cash = agent.economic_agent.current_cash
            current_stocks = agent.economic_agent.current_stock_quantity
            average_cost = agent.economic_agent.calculate_average_cost()
            market_price = self.environments['stock_market'].mechanism.current_price
            portfolio_value = current_cash + (current_stocks * market_price)
            cash_ratio = current_cash / portfolio_value if portfolio_value > 0 else 1
            stock_ratio = (current_stocks * market_price) / portfolio_value if portfolio_value > 0 else 0
            unrealized_profit = agent.economic_agent.calculate_unrealized_profit(market_price)
            expected_return = agent.economic_agent.expected_return

            agent.system = (
                f"Round {round_num}: You are an aggressive stock trader with the following portfolio:\n"
                f"- Cash: ${current_cash:.2f}\n"
                f"- Shares of {agent.economic_agent.stock_symbol}: {current_stocks}\n"
                f"- Average Cost Basis: ${average_cost:.2f}\n"
                f"- Current Market Price: ${market_price:.2f}\n"
                f"- Unrealized Profit/Loss: ${unrealized_profit:.2f}\n"
                f"- Total Portfolio Value: ${portfolio_value:.2f}\n"
                f"- Cash Ratio: {cash_ratio:.2f}\n"
                f"- Stock Ratio: {stock_ratio:.2f}\n\n"
                f"Your risk aversion level is {agent.economic_agent.risk_aversion:.2f} and expected return threshold is {expected_return:.2f}.\n\n"
                f"As an aggressive trader, your primary goal is to make frequent trades to maximize profits. "
                f"Sitting idle is not an option. You must either buy or sell in every round unless you don't have enough cash. "
                f"Analyze your current portfolio and the market conditions to decide your next action. "
                f"Remember, holding is not allowed. You MUST choose to either buy or sell. "
                f"When deciding, consider taking calculated risks for potentially higher returns. "
                f"Specify the quantity and price for your trade, aiming for larger trade sizes when possible."
            )

    def insert_ai_requests(self, ai_requests):
        self.data_inserter.insert_ai_requests(ai_requests)

    async def run_simulation(self):
        log_section(logger, "SIMULATION COMMENCING")

        try:
            for round_num in range(1, self.config.max_rounds + 1):
                log_round(logger, round_num)

                for env_name in self.simulation_order:
                    env_state = await self.run_environment(env_name, round_num)
                    self.update_simulation_state(env_name, env_state)

                # Process rewards using env_state
                self.calculate_environment_rewards(env_state)

                # Run parallel reflect
                reflect_prompts = await self.run_parallel_reflect(env_name)
                if reflect_prompts:
                    reflections = await self.run_parallel_ai_completion(reflect_prompts)
                    for agent, reflection in zip(self.agents, reflections):
                        if reflection.json_object:
                            log_reflection(logger, agent.index, f"{Fore.MAGENTA}{reflection.json_object.object}{Style.RESET_ALL}")
                            environment_reward = agent.environment_reward
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

            try:
                self.data_inserter.insert_round_data(round_num, self.agents, self.environments, self.config)
                logger.info(f"Data for round {round_num} inserted successfully.")
            except Exception as e:
                logger.error(f"Error inserting data for round {round_num}: {str(e)}")

        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
        finally:
            log_completion(logger, "SIMULATION COMPLETED")
            self.print_summary()

    
    def calculate_environment_rewards(self, env_state):
        # Initialize environment rewards
        for agent in self.agents:
            agent.environment_reward = 0.0

        # Get the list of trades executed in this round
        trades = env_state.global_observation.all_trades

        for trade in trades:
            # Process seller's reward
            seller_agent = self.agent_dict[trade.seller_id]
            cost_basis, quantity_sold = seller_agent.economic_agent.remove_positions(trade.quantity)
            realized_profit = (trade.price - cost_basis) * quantity_sold
            seller_agent.environment_reward += realized_profit
            # Update seller's cash
            seller_agent.economic_agent.current_portfolio.cash += trade.price * trade.quantity

            # Process buyer's reward (unrealized profit is zero at purchase)
            buyer_agent = self.agent_dict[trade.buyer_id]
            # Deduct cash from buyer
            if buyer_agent.economic_agent.current_portfolio.cash >= trade.price * trade.quantity:
                buyer_agent.economic_agent.current_portfolio.cash -= trade.price * trade.quantity
            else:
                raise ValueError(f"Agent {buyer_agent.id} does not have enough cash to complete the trade.")

            # Buyer does not have realized profit yet, set environment_reward to zero
            buyer_agent.environment_reward += 0.0

            # Update positions for buyer
            buyer_agent.economic_agent.add_position(trade.price, trade.quantity)

        # Handle agents who didn't trade
        for agent in self.agents:
            if not hasattr(agent, 'environment_reward'):
                agent.environment_reward = 0.0


    def update_simulation_state(self, env_name: str, env_state: EnvironmentStep):
        for agent in self.agents:
            agent_observation = env_state.global_observation.observations.get(agent.id)
            if agent_observation:
                agent.last_observation = agent_observation.observation  # Assuming .observation holds the observation dict
                agent.last_step = env_state
            else:
                agent.last_observation = None  # Ensure it's set even if None

        if not self.simulation_data or 'state' not in self.simulation_data[-1]:
            self.simulation_data.append({'state': {}})

        self.simulation_data[-1]['state'][env_name] = env_state.info

    def save_round_data(self, round_num):
        round_data = {
            'round': round_num,
            'agent_states': [{
                'id': agent.id,
                'cash': agent.economic_agent.current_cash,
                'stocks': agent.economic_agent.current_stock_quantity,
                'last_action': agent.last_action,
                'last_perception': agent.last_perception,
                'last_observation': agent.last_observation,
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
            file_path = self.log_folder / f"agent_{agent.index}_interactions.jsonl"
            with open(file_path, 'a') as f:
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

        print("\nFinal Agent States:")
        for agent in self.agents:
            print(f"Agent {agent.index}:")
            print(f"  Cash: ${agent.economic_agent.current_cash:.2f}")
            print(f"  Stocks: {agent.economic_agent.current_stock_quantity} shares")
            portfolio_value = agent.economic_agent.calculate_portfolio_value(self.environments['stock_market'].mechanism.current_price)
            print(f"  Total Portfolio Value: ${portfolio_value:.2f}")

        print("\nPrice History:")
        price_history = self.environments['stock_market'].mechanism.price_history
        for round_num, price in enumerate(price_history, start=1):
            print(f"  Round {round_num}: ${price:.2f}")

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
