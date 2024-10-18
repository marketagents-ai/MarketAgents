# orchestrator_groupchat_stock_market.py

import asyncio
from datetime import datetime
import json
import logging
import os
import random
import threading
import uuid
from pathlib import Path
from typing import List, Dict, Any, Union
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml
from colorama import Fore, Style

from market_agents.agents.market_agent import MarketAgent
from market_agents.environments.environment import MultiAgentEnvironment, EnvironmentStep
# Import the GroupChat environment classes from the provided script
from market_agents.environments.mechanisms.group_chat import (
    GroupChat, GroupChatAction, GroupChatActionSpace, GroupChatGlobalAction,
    GroupChatMessage, GroupChatObservationSpace, GroupChatGlobalObservation
)
from market_agents.inference.message_models import LLMConfig, LLMOutput, LLMPromptContext
from market_agents.agents.protocols.acl_message import ACLMessage
from market_agents.logger_utils import *
from market_agents.agents.personas.persona import generate_persona, save_persona_to_file, Persona
from market_agents.inference.parallel_inference import ParallelAIUtilities, RequestLimits
from market_agents.stock_market.stock_models import (
    Position, Portfolio, Stock, Endowment
)
from market_agents.stock_market.stock_agent import StockEconomicAgent
from market_agents.stock_market.insert_stock_simulation_data import SimulationDataInserter
# Import database functions
from market_agents.stock_market.setup_stock_database import create_database, create_tables

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
    initial_cash_min: float
    initial_cash_max: float
    initial_stocks_min: int
    initial_stocks_max: int
    risk_aversion: float
    expected_return: float
    use_llm: bool
    stock_symbol: str = "AAPL"
    max_relative_spread: float

class GroupChatConfig(BaseModel):
    name: str
    address: str
    max_rounds: int
    initial_topic: str

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
    environment_configs: Dict[str, Union[GroupChatConfig]]
    protocol: str
    database_config: DatabaseConfig = DatabaseConfig()

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

def load_config(config_path: Path = Path("./market_agents/stock_market/orchestrator_config_stock.yaml")) -> OrchestratorConfig:
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
        oai_request_limits = RequestLimits(max_requests_per_minute=20000, max_tokens_per_minute=2000000)
        anthropic_request_limits = RequestLimits(max_requests_per_minute=20000, max_tokens_per_minute=2000000)
        self.ai_utils = ParallelAIUtilities(
            oai_request_limits=oai_request_limits,
            anthropic_request_limits=anthropic_request_limits
        )
        self.topic_proposer = None
        self.current_topic = None
        # Initialize database parameters from config
        self.db_params = {
            'dbname': self.config.database_config.db_name,
            'user': self.config.database_config.db_user,
            'password': self.config.database_config.db_password,
            'host': self.config.database_config.db_host,
            'port': self.config.database_config.db_port
        }
        self.data_inserter = SimulationDataInserter(self.db_params)
        self.agent_dict: Dict[str, MarketAgent] = {}  # Mapping from agent IDs to agents

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

        self.topic_proposer = random.choice(self.agents)
        self.topic_proposer.system = "You are the group chat topic proposer agent. Your role is to propose interesting and relevant topics for the group discussion."

    def setup_environments(self):
        log_section(logger, "CONFIGURING GROUP CHAT ENVIRONMENT")
        if 'group_chat' not in self.config.environment_configs:
            raise KeyError("'group_chat' configuration is missing in environment_configs")

        group_chat_config = self.config.environment_configs['group_chat']
        if not isinstance(group_chat_config, GroupChatConfig):
            raise TypeError("Expected GroupChatConfig for 'group_chat' configuration")

        group_chat = GroupChat(
            max_rounds=group_chat_config.max_rounds,
            current_topic=group_chat_config.initial_topic,
            speaker_order=[agent.id for agent in self.agents],
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

    async def run_parallel_reflect(self, env_name: str) -> Union[List[LLMPromptContext], List[MarketAgent]]:
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
        if self.topic_proposer is None:
            raise ValueError("topic_proposer is not initialized")

        topic_action = await self.topic_proposer.generate_action(
            "group_chat",
            f"Consider recent economic events, market trends, or financial news. Propose a topic related to {self.current_topic}."
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
            default_topic = "Discuss recent trends in the stock market"
            logger.info(f"Using default topic: {Fore.YELLOW}{default_topic}{Style.RESET_ALL}")
            return default_topic

    async def run_environment(self, env_name: str, round_num: int) -> EnvironmentStep:
        env = self.environments[env_name]
        tracker = self.trackers[env_name]

        log_running(logger, env_name)

        if round_num == 1:
            self.current_topic = await self.generate_initial_topic()
            env.mechanism.current_topic = self.current_topic

        perception_prompts = await self.run_parallel_perceive(env_name)
        perceptions = await self.run_parallel_ai_completion(perception_prompts)

        # Create a mapping of agent IDs to perceptions
        perceptions_map = {perception.source_id: perception for perception in perceptions}

        for agent in self.agents:
            perception = perceptions_map.get(agent.id)
            if perception:
                perception_content = perception.json_object.object if perception.json_object else perception.str_content
                log_section(logger, f"Current Agent:\nAgent {agent.index} with persona:\n{agent.persona}")
                log_perception(logger, agent.index, f"{Fore.CYAN}{perception_content}{Style.RESET_ALL}")
                agent.last_perception = perception_content
            else:
                logger.warning(f"No perception found for agent {agent.id}")
                agent.last_perception = ""

        self.set_agent_system_messages(round_num)

        # Extract perception content to pass to generate_action
        perception_contents = []
        for agent in self.agents:
            perception_content = agent.last_perception
            perception_contents.append(perception_content)

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
            current_cash = agent.economic_agent.current_cash
            current_stocks = agent.economic_agent.current_stock_quantity
            average_cost = agent.economic_agent.calculate_average_cost()
            market_price = 150.0  # Replace with actual market price if available
            portfolio_value = current_cash + (current_stocks * market_price)
            cash_ratio = current_cash / portfolio_value if portfolio_value > 0 else 1
            stock_ratio = (current_stocks * market_price) / portfolio_value if portfolio_value > 0 else 0
            unrealized_profit = agent.economic_agent.calculate_unrealized_profit(market_price)
            expected_return = agent.economic_agent.expected_return

            agent.system = (
                f"Round {round_num}: You are a stock trader with the following portfolio:\n"
                f"- Cash: ${current_cash:.2f}\n"
                f"- Shares of {agent.economic_agent.stock_symbol}: {current_stocks}\n"
                f"- Average Cost Basis: ${average_cost:.2f}\n"
                f"- Current Market Price: ${market_price:.2f}\n"
                f"- Unrealized Profit/Loss: ${unrealized_profit:.2f}\n"
                f"- Total Portfolio Value: ${portfolio_value:.2f}\n"
                f"- Cash Ratio: {cash_ratio:.2f}\n"
                f"- Stock Ratio: {stock_ratio:.2f}\n\n"
                f"As a trader, your goal is to discuss the stock market, share insights, "
                f"and consider strategies that could improve your portfolio performance."
            )

    def process_group_chat_actions(self, actions_map: Dict[str, LLMOutput]) -> GroupChatGlobalAction:
        agent_actions = {}
        for agent_id, action_output in actions_map.items():
            try:
                action_content = action_output.json_object.object if action_output.json_object else json.loads(action_output.str_content or '')
                group_chat_message = GroupChatMessage(
                    content=action_content['action']['content'],
                    message_type=action_content.get('message_type', 'group_message'),
                    agent_id=agent_id
                )
                group_chat_action = GroupChatAction(agent_id=agent_id, action=group_chat_message)
                agent_actions[agent_id] = group_chat_action.model_dump()
                log_action(logger, self.agent_dict[agent_id].index, f"{Fore.BLUE}Message: {group_chat_message.content}{Style.RESET_ALL}")
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
                    local_step = env_step.get_local_step(agent.id)
                    if local_step:
                        agent.last_observation = local_step.observation

                reflect_prompts, agents_with_observations = await self.run_parallel_reflect("group_chat")
                reflections = await self.run_parallel_ai_completion(reflect_prompts)
                reflections_map = {reflection.source_id: reflection for reflection in reflections}

                # Process reflections
                for agent in agents_with_observations:
                    reflection = reflections_map.get(agent.id)
                    if reflection and reflection.json_object:
                        log_reflection(logger, agent.index, f"{Fore.MAGENTA}{reflection.json_object.object}{Style.RESET_ALL}")
                        agent.memory.append({
                            "type": "reflection",
                            "content": reflection.json_object.object.get("reflection", ""),
                            "strategy_update": reflection.json_object.object.get("strategy_update", ""),
                            "self_reward": reflection.json_object.object.get("self_reward", 0.0),
                            "observation": agent.last_observation,
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        logger.warning(f"No reflection found for agent {agent.id}")

                self.save_round_data(round_num)
                self.save_agent_interactions(round_num)

                # Insert data into the database
                try:
                    self.data_inserter.insert_round_data(round_num, self.agents, self.environments, self.config)
                    logger.info(f"Data for round {round_num} inserted successfully.")
                except Exception as e:
                    logger.error(f"Error inserting data for round {round_num}: {str(e)}")

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
        self.simulation_data[-1]['state']['messages'] = [message.dict() for message in env_step.global_observation.all_messages]

    def save_round_data(self, round_num):
        round_data = {
            'round': round_num,
            'agent_states': [{
                'id': agent.id,
                'cash': agent.economic_agent.current_cash,
                'stocks': agent.economic_agent.current_stock_quantity,
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
            file_path = self.log_folder / f"agent_{agent.index}_interactions.jsonl"
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
            print(f"Agent {agent.index}:")
            print(f"  Cash: ${agent.economic_agent.current_cash:.2f}")
            print(f"  Stocks: {agent.economic_agent.current_stock_quantity} shares")
            if agent.memory:
                print(f"  Last reflection: {agent.memory[-1]['content']}")

    async def start(self):
        log_section(logger, "GROUP CHAT SIMULATION INITIALIZING")
        self.generate_agents()
        self.setup_environments()
        self.setup_database()

        await self.run_simulation()

if __name__ == "__main__":
    config = load_config()
    orchestrator = Orchestrator(config)
    asyncio.run(orchestrator.start())
