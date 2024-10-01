from datetime import datetime
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Type, Tuple
from market_agents.economics.econ_models import Ask, Bid, Trade
from market_agents.inference.parallel_inference import ParallelAIUtilities, RequestLimits
from pydantic import BaseModel, Field
from colorama import Fore, Style
import threading
import os
import yaml
import random

from market_agents.agents.market_agent import MarketAgent
from market_agents.environments.environment import MultiAgentEnvironment, EnvironmentStep
from market_agents.environments.mechanisms.auction import AuctionAction, AuctionGlobalObservation, DoubleAuction, AuctionActionSpace, AuctionObservationSpace, GlobalAuctionAction
from market_agents.inference.message_models import LLMConfig, LLMOutput, LLMPromptContext
from market_agents.agents.protocols.acl_message import ACLMessage
from market_agents.logger_utils import *
from market_agents.agents.personas.persona import generate_persona, save_persona_to_file, Persona

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

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

class LLMConfig(BaseModel):
    name: str
    client: str
    model: str
    temperature: float
    max_tokens: int
    use_cache: bool

class AgentConfig(BaseModel):
    num_units: int
    base_value: float
    use_llm: bool
    initial_cash: float
    initial_goods: int
    good_name: str
    noise_factor: float
    max_relative_spread: float

class AuctionConfig(BaseModel):
    name: str
    address: str
    max_rounds: int
    good_name: str

class OrchestratorConfig(BaseSettings):
    num_agents: int
    max_rounds: int
    agent_config: AgentConfig
    llm_configs: list[LLMConfig]
    environment_configs: dict[str, AuctionConfig]
    protocol: str
    database_config: dict[str, str]

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

import yaml
from pathlib import Path

def load_config(config_path: Path = Path("./market_agents/orchestrator_config.yaml")) -> OrchestratorConfig:
    with open(config_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return OrchestratorConfig(**yaml_data)

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
        oai_request_limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=200000)
        anthropic_request_limits = RequestLimits(max_requests_per_minute=40, max_tokens_per_minute=10000)
        self.ai_utils = ParallelAIUtilities(
            oai_request_limits=oai_request_limits,
            anthropic_request_limits=anthropic_request_limits
        ) 

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

        for i, persona in enumerate(personas):
            llm_config = random.choice(self.config.llm_configs).model_dump()

            agent = MarketAgent.create(
                agent_id=i,
                is_buyer=persona.role.lower() == "buyer",
                **self.config.agent_config.dict(),
                llm_config=llm_config,
                protocol=ACLMessage,
                environments=self.environments,
                persona=persona
            )
            self.agents.append(agent)
            log_agent_init(logger, i, persona.role.lower() == "buyer")


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
        logger.info("Database setup skipped")

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

    async def run_parallel_reflect(self, env_name: str) -> List[LLMPromptContext]:
        reflect_prompts = []
        for agent in self.agents:
            if agent.last_observation:  # Only reflect if there's an observation
                reflect_prompt = await agent.reflect(env_name, return_prompt=True)
                reflect_prompts.append(reflect_prompt)
            else:
                logger.info(f"Skipping reflection for agent {agent.id} due to no observation")
        return reflect_prompts

    async def run_environment(self, env_name: str, round_num: int) -> EnvironmentStep:
        env = self.environments[env_name]
        tracker = self.trackers[env_name]
        
        log_running(logger, env_name)

        # Run parallel perceive
        perception_prompts = await self.run_parallel_perceive(env_name)
        perceptions = await self.run_parallel_ai_completion(perception_prompts)
        for agent, perception in zip(self.agents, perceptions):
            log_section(logger, f"Current Agent:\nAgent {agent.id} with persona:\n{agent.persona}")
            log_perception(logger, int(agent.id), f"{Fore.CYAN}{perception.json_object.object}{Style.RESET_ALL}")

        # Set system messages for agents
        self.set_agent_system_messages(round_num, env.mechanism.good_name)

        # Run parallel generate_action
        action_prompts = await self.run_parallel_generate_action(env_name, [p.json_object.object for p in perceptions])
        actions = await self.run_parallel_ai_completion(action_prompts)

        agent_actions = {}
        for agent, action in zip(self.agents, actions):
            try:
                action_content = action.json_object.object
                if 'price' in action_content and 'quantity' in action_content:
                    action_content['quantity'] = 1

                    if agent.role == "buyer":
                        auction_action = Bid(**action_content)
                    elif agent.role == "seller":
                        auction_action = Ask(**action_content)
                    else:
                        raise ValueError(f"Invalid agent role: {agent.role}")
                else:
                    raise ValueError(f"Invalid action content: {action_content}")
                
                agent_actions[agent.id] = AuctionAction(agent_id=agent.id, action=auction_action)

                action_type = "Bid" if isinstance(auction_action, Bid) else "Ask"
                color = Fore.BLUE if action_type == "Bid" else Fore.GREEN
                log_action(logger, int(agent.id), f"{color}{action_type}: {auction_action}{Style.RESET_ALL}")
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Error creating AuctionAction for agent {agent.id}: {str(e)}")
                continue

        global_action = GlobalAuctionAction(actions=agent_actions)
        env_state = env.step(global_action)
        logger.info(f"Completed {env_name} step")

        if isinstance(env_state.global_observation, AuctionGlobalObservation):
            self.process_trades(env_state.global_observation, tracker)
        
        return env_state

    def set_agent_system_messages(self, round_num: int, good_name: str):
        for agent in self.agents:
            if round_num == 1:
                if agent.role == "buyer":
                    agent.system = f"This is the first round of the market so there are no bids or asks yet. You can make a profit by buying at {agent.preference_schedule.get_value(1)*0.99} or lower"
                elif agent.role == "seller":
                    agent.system = f"This is the first round of the market so there are no bids or asks yet. You can make a profit by selling at {agent.preference_schedule.get_value(1)*1.01} or higher"
            else:
                if agent.role == "buyer":
                    current_value = agent.preference_schedule.get_value(agent.endowment.current_basket.goods_dict.get(good_name, 0) + 1)
                    suggested_price = current_value * 0.99
                    agent.system = f"Your current basket: {agent.endowment.current_basket}. Your current value for the good is {current_value}. You can make a profit by buying at {suggested_price} or lower."
                elif agent.role == "seller":
                    current_cost = agent.preference_schedule.get_value(agent.endowment.current_basket.goods_dict.get(good_name, 0))
                    suggested_price = current_cost * 1.01
                    agent.system = f"Your current basket: {agent.endowment.current_basket}. Your current cost for the good is {current_cost}. You can make a profit by selling at {suggested_price} or higher."

    async def run_simulation(self):
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
                
                # Run parallel reflect
                reflect_prompts = await self.run_parallel_reflect(env_name)
                reflections = await self.run_parallel_ai_completion(reflect_prompts)
                for agent, reflection in zip([a for a in self.agents if a.last_observation], reflections):
                    if reflection.json_object:
                        log_reflection(logger, int(agent.id), f"{Fore.MAGENTA}{reflection.json_object.object}{Style.RESET_ALL}")
                        agent.memory.append({
                            "type": "reflection",
                            "content": reflection.json_object.object["reflection"],
                            "strategy_update": reflection.json_object.object["strategy_update"],
                            "observation": agent.last_observation,
                            "reward": agent.last_observation.get('reward', 0),
                            "timestamp": datetime.now().isoformat()
                        })
                
                self.save_round_data(round_num)
                self.save_agent_interactions(round_num)
        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
        finally:
            log_completion(logger, "SIMULATION COMPLETED")
            self.print_summary()
    
    def process_trades(self, global_observation: AuctionGlobalObservation, tracker: AuctionTracker):
        round_surplus = 0
        round_quantity = 0
        logger.info(f"Processing {len(global_observation.all_trades)} trades")
        for trade in global_observation.all_trades:
            buyer = self.agents[int(trade.buyer_id)]
            seller = self.agents[int(trade.seller_id)]
            logger.info(f"Processing trade between buyer {buyer.id} and seller {seller.id}")
            
            buyer_utility_before = buyer.calculate_utility(buyer.endowment.current_basket)
            seller_utility_before = seller.calculate_utility(seller.endowment.current_basket)
            
            buyer.process_trade(trade)
            seller.process_trade(trade)
            
            buyer_utility_after = buyer.calculate_utility(buyer.endowment.current_basket)
            seller_utility_after = seller.calculate_utility(seller.endowment.current_basket)
            
            trade_surplus = (buyer_utility_after - buyer_utility_before) + (seller_utility_after - seller_utility_before)
            
            tracker.add_trade(trade)
            round_surplus += trade_surplus
            round_quantity += trade.quantity
            logger.info(f"Executed trade: {trade}")
            logger.info(f"Trade surplus: {trade_surplus}")

        tracker.add_round_data(round_surplus, round_quantity)
        logger.info(f"Round summary - Surplus: {round_surplus}, Quantity: {round_quantity}")
    
    def update_simulation_state(self, env_name: str, env_state: EnvironmentStep):
        for agent in self.agents:
            agent_observation = env_state.global_observation.observations.get(agent.id)
            if agent_observation:
                if not isinstance(agent_observation, dict):
                    agent_observation = agent_observation.dict()

                agent.last_observation = agent_observation
                
                new_cash = agent_observation.get('endowment', {}).get('cash')
                new_goods = agent_observation.get('endowment', {}).get('goods', {})
                
                if new_cash is not None:
                    agent.endowment.current_basket.cash = new_cash
                
                for good, quantity in new_goods.items():
                    agent.endowment.current_basket.update_good_quantity(good, quantity)

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
                'cash': agent.endowment.current_basket.cash,
                'goods': agent.endowment.current_basket.goods_dict,
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
            file_path = self.log_folder / f"agent_{agent.id}_interactions.jsonl"
            with open(file_path, 'a') as f:
                # Get all interactions that don't have a round number yet
                new_interactions = [interaction for interaction in agent.interactions if 'round' not in interaction]
                for interaction in new_interactions:
                    interaction_with_round = {
                        "round": round_num,
                        **interaction
                    }
                    json.dump(interaction_with_round, f)
                    f.write('\n')
                # Clear the processed interactions
                agent.interactions = [interaction for interaction in agent.interactions if 'round' in interaction]
        
        logger.info(f"Saved agent interactions for round {round_num} to {self.log_folder}")

    def print_summary(self):
        log_section(logger, "SIMULATION SUMMARY")
        
        total_buyer_surplus = sum(agent.calculate_individual_surplus() for agent in self.agents if agent.role == "buyer")
        total_seller_surplus = sum(agent.calculate_individual_surplus() for agent in self.agents if agent.role == "seller")
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
            print(f"Agent {agent.id} ({agent.role}):")
            print(f"  Cash: {agent.endowment.current_basket.cash:.2f}")
            print(f"  Goods: {agent.endowment.current_basket.goods_dict}")
            surplus = agent.calculate_individual_surplus()
            print(f"  Individual Surplus: {surplus:.2f}")

    def write_to_db(self):
        # Implement database write operation here
        pass

    async def start(self):
        log_section(logger, "MARKET SIMULATION INITIALIZING")
        self.generate_agents()
        self.setup_environments()
        #self.setup_database()

        if self.dashboard:
            dashboard_thread = threading.Thread(target=self.run_dashboard)
            dashboard_thread.start()

        await self.run_simulation()
        self.write_to_db()

        if self.dashboard:
            dashboard_thread.join()

if __name__ == "__main__":
    import asyncio
    config = load_config()
    orchestrator = Orchestrator(config)
    asyncio.run(orchestrator.start())

