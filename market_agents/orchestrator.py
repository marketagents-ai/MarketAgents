import logging
from pathlib import Path
from typing import List, Dict, Any, Type
from market_agents.economics.econ_models import Ask, Bid
from pydantic import BaseModel, Field
from colorama import Fore, Style
import threading
import os
import yaml

from market_agents.agents.market_agent import MarketAgent
from market_agents.environments.environment import MultiAgentEnvironment
from market_agents.environments.mechanisms.auction import AuctionAction, DoubleAuction, AuctionActionSpace, AuctionObservationSpace, GlobalAuctionAction
from market_agents.inference.message_models import LLMConfig
from market_agents.agents.protocols.acl_message import ACLMessage
from market_agents.logger_utils import *
from market_agents.agents.personas.persona import generate_persona, save_persona_to_file, Persona

# Remove the root logger setup to prevent duplicate logging
logger = logging.getLogger(__name__)
logger.handlers = []  # Clear any existing handlers
logger.addHandler(logging.NullHandler())  # Add a null handler to prevent logging to the root logger

class AgentConfig(BaseModel):
    num_units: int
    base_value: float
    use_llm: bool
    initial_cash: float
    initial_goods: int
    good_name: str
    noise_factor: float = Field(default=0.1)
    max_relative_spread: float = Field(default=0.2)

class AuctionConfig(BaseModel):
    name: str
    address: str
    max_rounds: int
    good_name: str

class OrchestratorConfig(BaseModel):
    num_agents: int
    max_rounds: int
    agent_config: AgentConfig
    llm_config: LLMConfig
    environment_configs: Dict[str, AuctionConfig]
    protocol: Type[ACLMessage]
    database_config: Dict[str, Any]

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
            agent = MarketAgent.create(
                agent_id=i,
                is_buyer=persona.role.lower() == "buyer",
                **self.config.agent_config.dict(),
                llm_config=self.config.llm_config,
                protocol=self.config.protocol,
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
                log_environment_setup(logger, env_name)
            else:
                logger.warning(f"Unknown environment type: {env_name}")

        logger.info(f"Set up {len(self.environments)} environments")

        for agent in self.agents:
            agent.environments = self.environments

    #def setup_dashboard(self):
    #    log_section(logger, "INITIALIZING DASHBOARD")
    #    self.dashboard = create_dashboard(self.data_source)
    #    logger.info("Dashboard setup complete")

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
                        env_state = await self.run_environment(env_name)
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
                self.update_dashboard()
        except Exception as e:
            logger.error(f"Simulation failed: {str(e)}")
        finally:
            log_completion(logger, "SIMULATION COMPLETED")
            self.print_summary()

    async def run_environment(self, env_name: str):
        env = self.environments[env_name]
        
        log_running(logger, env_name)
        agent_actions = {}
        for agent in self.agents:
            log_section(logger, f"Current Agent:\nAgent {agent.id} with persona:\n{agent.persona}")
            perception = await agent.perceive(env_name)
            log_perception(logger, int(agent.id), f"{Fore.CYAN}{perception}{Style.RESET_ALL}")

            action = await agent.generate_action(env_name, perception)
            log_raw_action(logger, int(agent.id), f"{Fore.LIGHTBLUE_EX}{action}{Style.RESET_ALL}")      
    
            try:
                # Extract the actual action from the content
                action_content = action['content']['action']
                if 'price' in action_content and 'quantity' in action_content:
                    #fix quantity to 1 for now
                    action_content['quantity'] = 1
                    # Determine if it's a Bid or Ask based on the agent's role
                    if agent.role == "buyer":
                        auction_action = Bid(**action_content)
                    elif agent.role == "seller":
                        auction_action = Ask(**action_content)
                    else:
                        raise ValueError(f"Invalid agent role: {agent.role}")
                else:
                    raise ValueError(f"Invalid action content: {action_content}")
                
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
        return env_state

    def update_simulation_state(self, env_name: str, env_state: Dict[str, Any]):
        for agent in self.agents:
            agent_observation = env_state.global_observation.observations.get(agent.id)
            if agent_observation:
                if not isinstance(agent_observation, dict):
                    agent_observation = agent_observation.dict()
                
                # Update the agent's endowment directly
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
                'cash': agent.endowment.current_basket.cash,  # Change this line
                'goods': agent.endowment.current_basket.goods_dict,  # Change this line
                'last_action': agent.last_action,
                'memory': agent.memory[-1] if agent.memory else None
            } for agent in self.agents],
            'environment_states': {name: env.get_global_state() for name, env in self.environments.items()},
        }
        
        if self.simulation_data and 'state' in self.simulation_data[-1]:
            round_data['state'] = self.simulation_data[-1]['state']
        
        self.simulation_data.append(round_data)

    #def data_source(self):
    #    env = self.environments['auction']
    #    global_state = env.get_global_state()
    #    
    #    return {
    #        'current_step': global_state.get('current_step', 0),
    #        'max_steps': self.config.max_rounds,
    #        'total_utility': global_state.get('total_utility', 0),
    #        'goods': global_state.get('goods', []),
    #        'equilibria': global_state.get('equilibria', {}),
    #        'order_books': global_state.get('order_books', {}),
    #        'trade_history': global_state.get('trade_history', []),
    #        'successful_trades': global_state.get('successful_trades', []),
    #        'current_supply_curves': global_state.get('current_supply_curves', {}),
    #        'current_demand_curves': global_state.get('current_demand_curves', {})
    #    }
#
    #def update_dashboard(self):
    #    if self.dashboard:
    #        logger.info("Updating dashboard data...")
    #        self.latest_data = self.data_source()
#
    #def run_dashboard(self):
    #    if self.dashboard:
    #        log_section(logger, "LAUNCHING DASHBOARD UI")
    #        self.dashboard.run_server(debug=True, use_reloader=False)

    def print_summary(self):
        log_section(logger, "SIMULATION SUMMARY")
        
        # Calculate total empirical surplus
        total_buyer_surplus = sum(agent.calculate_individual_surplus() for agent in self.agents if agent.role == "buyer")
        total_seller_surplus = sum(agent.calculate_individual_surplus() for agent in self.agents if agent.role == "seller")
        total_empirical_surplus = total_buyer_surplus + total_seller_surplus

        print(f"Total Empirical Buyer Surplus: {total_buyer_surplus:.2f}")
        print(f"Total Empirical Seller Surplus: {total_seller_surplus:.2f}")
        print(f"Total Empirical Surplus: {total_empirical_surplus:.2f}")

        # Calculate theoretical equilibrium (if available)
        env = self.environments['auction']
        global_state = env.get_global_state()
        equilibria = global_state.get('equilibria', {})
        
        if equilibria:
            theoretical_total_surplus = sum(data['total_surplus'] for data in equilibria.values())
            print(f"\nTheoretical Total Surplus: {theoretical_total_surplus:.2f}")

            # Compute and print the empirical efficiency
            efficiency = (total_empirical_surplus / theoretical_total_surplus) * 100 if theoretical_total_surplus > 0 else 0
            print(f"\nEmpirical Efficiency: {efficiency:.2f}%")
        else:
            print("\nTheoretical equilibrium data not available.")

        # Print trade statistics
        successful_trades = global_state.get('successful_trades', [])
        print(f"\nTotal number of successful trades: {len(successful_trades)}")
        
        if successful_trades:
            average_price = sum(trade['price'] for trade in successful_trades) / len(successful_trades)
            print(f"Average trade price: {average_price:.2f}")

        # Print final agent states
        print("\nFinal Agent States:")
        for agent in self.agents:
            print(f"Agent {agent.id} ({agent.role}):")
            print(f"  Cash: {agent.endowment.current_basket.cash:.2f}")
            print(f"  Goods: {agent.endowment.current_basket.goods_dict}")
            print(f"  Individual Surplus: {agent.calculate_individual_surplus():.2f}")

    async def start(self):
        log_section(logger, "MARKET SIMULATION INITIALIZING")
        self.generate_agents()
        self.setup_environments()
        #self.setup_dashboard()
        self.setup_database()

        if self.dashboard:
            dashboard_thread = threading.Thread(target=self.run_dashboard)
            dashboard_thread.start()

        await self.run_simulation()

        if self.dashboard:
            dashboard_thread.join()

if __name__ == "__main__":
    config = OrchestratorConfig(
        num_agents=8,
        max_rounds=4,
        agent_config=AgentConfig(
            num_units=5,
            base_value=100,
            use_llm=True,
            initial_cash=1000,
            initial_goods=0,
            good_name="apple",
            noise_factor=0.1,
            max_relative_spread=0.2
        ),
        llm_config=LLMConfig(
            client='openai',
            model='gpt-4o-mini',
            temperature=0.5,
            max_tokens=4096,
            use_cache=True
        ),
        environment_configs={
            'auction': AuctionConfig(
                name='Apple Market',
                address='apple_market',
                max_rounds=100,
                good_name='apple'
            ),
        },
        protocol=ACLMessage,
        database_config={
            'db_type': 'postgres',
            'db_name': 'market_simulation'
        }
    )
    
    import asyncio
    orchestrator = Orchestrator(config)
    asyncio.run(orchestrator.start())