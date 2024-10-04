# orchestrator.py
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Type, Tuple
from market_agents.economics.econ_models import Ask, Bid, Trade
from pydantic import BaseModel, Field
from colorama import Fore, Style
import threading
import os
import yaml
import random
import asyncio

from market_agents.agents.market_agent import MarketAgent
from market_agents.economics.econ_agent import create_economic_agent
from market_agents.environments.environment import MultiAgentEnvironment, EnvironmentStep
from market_agents.environments.mechanisms.auction import AuctionAction, AuctionGlobalObservation, DoubleAuction, AuctionActionSpace, AuctionObservationSpace, GlobalAuctionAction
from market_agents.inference.message_models import LLMConfig
from market_agents.agents.protocols.acl_message import ACLMessage
from market_agents.logger_utils import *
from market_agents.agents.personas.persona import generate_persona, save_persona_to_file, Persona

# Remove the root logger setup to prevent duplicate logging
logger = logging.getLogger(__name__)
logger.handlers = []  # Clear any existing handlers
logger.addHandler(logging.NullHandler())  # Add a null handler to prevent logging to the root logger

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    llm_configs: List[LLMConfig]
    environment_configs: Dict[str, AuctionConfig]
    protocol: str
    database_config: Dict[str, str]

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

import yaml
from pathlib import Path

def load_config(config_path: Path = Path("./market_agents/orchestrator_config.yaml")) -> OrchestratorConfig:
    with open(config_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    
    # Ensure initial_cash and initial_goods are present in agent_config
    if 'agent_config' in yaml_data:
        yaml_data['agent_config']['initial_cash'] = yaml_data['agent_config'].get('initial_cash', 1000)
        yaml_data['agent_config']['initial_goods'] = yaml_data['agent_config'].get('initial_goods', 10)
    
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
        agent_config = self.config.agent_config.dict()
        good_name = agent_config.get('good_name', 'apple')

        for i, persona in enumerate(personas):
            llm_config = random.choice(self.config.llm_configs)

            # Assign roles explicitly based on index
            if i < num_buyers:
                is_buyer = True
                persona.role = "buyer"
            else:
                is_buyer = False
                persona.role = "seller"

            if is_buyer:
                initial_cash = agent_config.get('initial_cash', 1000)
                initial_goods_quantity = 0
            else:
                initial_cash = agent_config.get('initial_cash', 0)
                initial_goods_quantity = agent_config.get('initial_goods', 10)

            initial_goods = {good_name: initial_goods_quantity}

            # Create economic agent
            economic_agent = create_economic_agent(
                agent_id=str(i),
                goods=[good_name],
                buy_goods=[good_name] if is_buyer else [],
                sell_goods=[good_name] if not is_buyer else [],
                base_values={good_name: agent_config.get('base_value', 100)},
                initial_cash=initial_cash,
                initial_goods=initial_goods,
                num_units=agent_config.get('num_units', 10),
                noise_factor=agent_config.get('noise_factor', 0.1),
                max_relative_spread=agent_config.get('max_relative_spread', 0.2),
            )

            agent = MarketAgent.create(
                agent_id=i,
                is_buyer=is_buyer,
                num_units=agent_config.get('num_units', 10),
                base_value=agent_config.get('base_value', 100),
                use_llm=agent_config.get('use_llm', True),
                initial_cash=initial_cash,
                initial_goods=initial_goods_quantity,
                good_name=good_name,
                noise_factor=agent_config.get('noise_factor', 0.1),
                max_relative_spread=agent_config.get('max_relative_spread', 0.2),
                llm_config=llm_config.dict(),
                environments=self.environments,
                protocol=ACLMessage,
                persona=persona
            )
            agent.economic_agent = economic_agent  # Assign the economic agent to the agent
            self.agents.append(agent)
            log_agent_init(logger, i, is_buyer, persona)

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

                for agent in self.agents:
                    try:
                        reflection = await agent.reflect(env_name)
                        if reflection:
                            log_reflection(logger, int(agent.id), f"{Fore.MAGENTA}{reflection}{Style.RESET_ALL}")
                    except Exception as e:
                        logger.error(f"Error in agent {agent.id} reflection: {str(e)}")

                self.save_round_data(round_num)
                # self.update_dashboard()

                # save interactions after each round
                self.save_agent_interactions(round_num)
        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
        finally:
            log_completion(logger, "SIMULATION COMPLETED")
            self.print_summary()

    async def run_environment(self, env_name: str, round_num: int) -> EnvironmentStep:
        env = self.environments[env_name]
        tracker = self.trackers[env_name]

        log_running(logger, env_name)

        # Reset pending orders at the beginning of the round
        for agent in self.agents:
            agent.economic_agent.reset_all_pending_orders()

        agent_actions = {}
        for agent in self.agents:
            log_section(logger, f"Current Agent:\nAgent {agent.id} with persona:\n{agent.persona}")
            perception = await agent.perceive(env_name)
            log_perception(logger, int(agent.id), f"{Fore.CYAN}{perception}{Style.RESET_ALL}")

            good_name = env.mechanism.good_name  # Use the good_name from the environment
            print(f"THE GOOD NAME IS {good_name}")

            # Set system messages for agents
            if round_num == 1:
                if agent.role == "buyer":
                    value_schedule = agent.economic_agent.value_schedules[good_name]
                    current_value = value_schedule.get_value(1)
                    suggested_price = current_value * 0.99
                    agent.system = (
                        f"This is the first round of the market so there are no bids or asks yet. "
                        f"You can make a profit by buying at {suggested_price:.2f} or lower."
                    )
                elif agent.role == "seller":
                    if agent.economic_agent.endowment.current_basket.goods_dict.get(good_name, 0) <= 0:
                        agent.system = f"You have no {good_name} to sell."
                    else:
                        cost_schedule = agent.economic_agent.cost_schedules[good_name]
                        current_cost = cost_schedule.get_value(1)
                        suggested_price = current_cost * 1.01
                        agent.system = (
                            f"This is the first round of the market so there are no bids or asks yet. "
                            f"You can make a profit by selling at {suggested_price:.2f} or higher."
                        )
            else:
                current_quantity = agent.economic_agent.endowment.current_basket.goods_dict.get(good_name, 0)
                if agent.role == "buyer":
                    value_schedule = agent.economic_agent.value_schedules[good_name]
                    value_q_plus_1 = value_schedule.get_value(current_quantity + 1)
                    value_q = value_schedule.get_value(current_quantity)
                    marginal_value = value_q_plus_1 - value_q
                    suggested_price = marginal_value * 0.99
                    agent.system = (
                        f"Your current basket: {agent.economic_agent.endowment.current_basket}. "
                        f"Your marginal value for the next unit of {good_name} is {marginal_value:.2f}. "
                        f"You can make a profit by buying at {suggested_price:.2f} or lower."
                    )
                elif agent.role == "seller":
                    if current_quantity <= 0:
                        agent.system = f"You have no {good_name} to sell."
                    else:
                        cost_schedule = agent.economic_agent.cost_schedules[good_name]
                        cost_q = cost_schedule.get_value(current_quantity)
                        cost_q_minus_1 = cost_schedule.get_value(current_quantity - 1)
                        marginal_cost = cost_q - cost_q_minus_1
                        suggested_price = marginal_cost * 1.01
                        agent.system = (
                            f"Your current basket: {agent.economic_agent.endowment.current_basket}. "
                            f"Your marginal cost for selling the next unit of {good_name} is {marginal_cost:.2f}. "
                            f"You can make a profit by selling at {suggested_price:.2f} or higher."
                        )
            action = await agent.generate_action(env_name, perception)
            log_raw_action(logger, int(agent.id), {"action": json.dumps(action, indent=2)})
            logger.info(f"{Fore.LIGHTBLUE_EX}Agent {agent.id} action: {json.dumps(action, indent=2)}{Style.RESET_ALL}")

            try:
                action_content = action['content']
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

                # Update agent's pending orders
                agent.economic_agent.pending_orders.setdefault(good_name, []).append(auction_action)

                agent_actions[agent.id] = AuctionAction(agent_id=agent.id, action=auction_action)
            except (KeyError, ValueError) as e:
                logger.error(f"Error creating AuctionAction for agent {agent.id}: {str(e)}")
                continue

            action_type = "Bid" if isinstance(auction_action, Bid) else "Ask"
            color = Fore.BLUE if action_type == "Bid" else Fore.GREEN
            log_action(logger, int(agent.id), f"{color}{action_type}: {auction_action}{Style.RESET_ALL}")

        global_action = GlobalAuctionAction(actions=agent_actions)
        env_state = env.step(global_action)
        logger.info(f"Completed {env_name} step")

        if isinstance(env_state.global_observation, AuctionGlobalObservation):
            logger.info(f"Processing trades with: {tracker}")
            self.process_trades(env_state.global_observation, tracker)

        return env_state

    def process_trades(self, global_observation: AuctionGlobalObservation, tracker: AuctionTracker):
        round_surplus = 0
        round_quantity = 0
        logger.info(f"Processing {len(global_observation.all_trades)} trades")
        for trade in global_observation.all_trades:
            buyer = self.agents[int(trade.buyer_id)]
            seller = self.agents[int(trade.seller_id)]
            logger.info(f"Processing trade between buyer {buyer.id} and seller {seller.id}")
            
            buyer_utility_before = buyer.economic_agent.calculate_utility(buyer.economic_agent.endowment.current_basket)
            seller_utility_before = seller.economic_agent.calculate_utility(seller.economic_agent.endowment.current_basket)
            
            buyer.economic_agent.process_trade(trade)
            seller.economic_agent.process_trade(trade)
            
            buyer_utility_after = buyer.economic_agent.calculate_utility(buyer.economic_agent.endowment.current_basket)
            seller_utility_after = seller.economic_agent.calculate_utility(seller.economic_agent.endowment.current_basket)
            
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
                    agent.economic_agent.endowment.current_basket.cash = new_cash

                for good, quantity in new_goods.items():
                    agent.economic_agent.endowment.current_basket.update_good_quantity(good, quantity)

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
            print(f"Agent {agent.id} ({agent.role}):")
            print(f"  Cash: {agent.economic_agent.endowment.current_basket.cash:.2f}")
            print(f"  Goods: {agent.economic_agent.endowment.current_basket.goods_dict}")
            surplus = agent.economic_agent.calculate_individual_surplus()
            print(f"  Individual Surplus: {surplus:.2f}")

    def write_to_db(self):
        # Implement database write operation here
        pass
    
    def run_dashboard(self):
        # Implement dashboard logic here
        pass

    async def start(self):
        log_section(logger, "MARKET SIMULATION INITIALIZING")
        self.generate_agents()
        self.setup_environments()
        # self.setup_database()

        if self.dashboard:
            dashboard_thread = threading.Thread(target=self.run_dashboard)
            dashboard_thread.start()

        await self.run_simulation()
        self.write_to_db()

        if self.dashboard:
            dashboard_thread.join()

if __name__ == "__main__":
    config = load_config()
    orchestrator = Orchestrator(config)
    asyncio.run(orchestrator.start())
