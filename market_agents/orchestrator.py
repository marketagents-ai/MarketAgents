import logging
from typing import List, Dict, Any, Type
from environments.auction.auction import DoubleAuction
from pydantic import BaseModel, Field
from colorama import Fore, Style
import threading
import os
import yaml

from market_agent.market_agents import MarketAgent
from environments.environment import Environment
from environments.auction.auction_environment import AuctionEnvironment
from base_agent.aiutilities import LLMConfig
from base_agent.agent import Agent
from protocols.protocol import Protocol
from protocols.acl_message import ACLMessage
from simulation_app import create_dashboard
from logger_utils import *
from personas.persona import generate_persona, save_persona_to_file, Persona

logger = setup_logger(__name__)

class AgentConfig(BaseModel):
    num_units: int
    base_value: float
    use_llm: bool
    initial_cash: float
    initial_goods: int
    noise_factor: float = Field(default=0.1)
    max_relative_spread: float = Field(default=0.2)

class AuctionConfig(BaseModel):
    name: str
    address: str
    auction_type: str
    max_steps: int

class OrchestratorConfig(BaseModel):
    num_agents: int
    max_rounds: int
    agent_config: AgentConfig
    llm_config: LLMConfig
    environment_configs: Dict[str, AuctionConfig]
    protocol: Type[Protocol]
    database_config: Dict[str, Any]

class Orchestrator:
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.agents: List[MarketAgent] = []
        self.environments: Dict[str, Environment] = {}
        self.dashboard = None
        self.database = None
        self.simulation_order = ['auction']
        self.simulation_data: List[Dict[str, Any]] = []
        self.latest_data = None

    def load_or_generate_personas(self) -> List[Persona]:
        personas_dir = "./personas/generated_personas"
        existing_personas = []

        # Check if the directory exists and load existing personas
        if os.path.exists(personas_dir):
            for filename in os.listdir(personas_dir):
                if filename.endswith(".yaml"):
                    with open(os.path.join(personas_dir, filename), 'r') as file:
                        persona_data = yaml.safe_load(file)
                        existing_personas.append(Persona(**persona_data))

        # Generate additional personas if needed
        while len(existing_personas) < self.config.num_agents:
            new_persona = generate_persona()
            existing_personas.append(new_persona)
            save_persona_to_file(new_persona, os.path.join(personas_dir))

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
        
        logger.info(f"Generated {len(self.agents)} agents")

    def setup_environments(self):
        log_section(logger, "CONFIGURING MARKET ENVIRONMENTS")
        for env_name, env_config in self.config.environment_configs.items():
            if env_name == 'auction':
                env = AuctionEnvironment(
                    agents=self.agents,
                    max_steps=env_config.max_steps,
                    protocol=self.config.protocol,
                    name=env_config.name,
                    address=env_config.address,
                    auction_type=env_config.auction_type
                )
                self.environments[env_name] = env
                log_environment_setup(logger, env_name)
            else:
                logger.warning(f"Unknown environment type: {env_name}")

        logger.info(f"Set up {len(self.environments)} environments")

        for agent in self.agents:
            agent.environments = self.environments

    def setup_dashboard(self):
        log_section(logger, "INITIALIZING DASHBOARD")
        self.dashboard = create_dashboard(self.data_source)
        logger.info("Dashboard setup complete")

    def setup_database(self):
        log_section(logger, "CONFIGURING SIMULATION DATABASE")
        logger.info("Database setup skipped")

    def run_simulation(self):
        log_section(logger, "SIMULATION COMMENCING")
        
        for round_num in range(1, self.config.max_rounds + 1):
            log_round(logger, round_num)
            
            for env_name in self.simulation_order:
                env_state = self.run_environment(env_name)
                self.update_simulation_state(env_name, env_state)
            
            for agent in self.agents:
                reflection = agent.reflect(env_name)
                if reflection:
                    log_reflection(logger, int(agent.id), f"{Fore.MAGENTA}{reflection}{Style.RESET_ALL}")
                else:
                    logger.warning(f"Agent {agent.id} returned empty reflection")
            
            self.save_round_data(round_num)
            self.update_dashboard()
        
        log_completion(logger, "SIMULATION COMPLETED")

    def run_environment(self, env_name: str):
        env = self.environments[env_name]
        
        log_running(logger, env_name)
        agent_actions = {}
        for agent in self.agents:
            log_section(logger, f"Current Agent:\nAgent {agent.id} with persona:\n{agent.persona}")
            perception = agent.perceive(env_name)
            log_perception(logger, int(agent.id), f"{Fore.CYAN}{perception}{Style.RESET_ALL}")

            action = agent.generate_action(env_name, perception)
            agent_actions[agent.id] = action
            action_type = action['content'].get('action', 'Unknown')
            if action_type == 'bid':
                color = Fore.BLUE
            elif action_type == 'ask':
                color = Fore.GREEN
            else:
                color = Fore.YELLOW
            log_action(logger, int(agent.id), f"{color}{action_type}: {action['content']}{Style.RESET_ALL}")
        
        env_state = env.update(agent_actions)
        logger.info(f"Completed {env_name} step")
        return env_state

    def update_simulation_state(self, env_name: str, env_state: Dict[str, Any]):
        # Update agent states based on the environment state
        for agent in self.agents:
            agent_observation = env_state['observations'].get(agent.id)
            if agent_observation:
                # Convert the observation to a dictionary if it's not already
                if not isinstance(agent_observation, dict):
                    agent_observation = agent_observation.dict()
                # Call the EconomicAgent's update_state method directly
                agent.update_state(agent_observation)

        # Create a new state dictionary for this round if it doesn't exist
        if not self.simulation_data or 'state' not in self.simulation_data[-1]:
            self.simulation_data.append({'state': {}})

        # Update the simulation state
        self.simulation_data[-1]['state'][env_name] = env_state.get('market_info', {})
        self.simulation_data[-1]['state']['trade_info'] = env_state.get('trade_info', {})

    def save_round_data(self, round_num):
        round_data = {
            'round': round_num,
            'agent_states': [{
                'id': agent.id,
                'is_buyer': agent.is_buyer,
                'cash': agent.endowment.cash,
                'goods': agent.endowment.goods,
                'last_action': agent.last_action,
                'memory': agent.memory[-1] if agent.memory else None
            } for agent in self.agents],
            'environment_states': {name: env.get_global_state() for name, env in self.environments.items()},
        }
        
        # Add the state data if it exists
        if self.simulation_data and 'state' in self.simulation_data[-1]:
            round_data['state'] = self.simulation_data[-1]['state']
        
        self.simulation_data.append(round_data)

    def data_source(self):
        env = self.environments['auction']
        return env.get_global_state()

    def update_dashboard(self):
        if self.dashboard:
            logger.info("Updating dashboard data...")
            self.latest_data = self.data_source()

    def run_dashboard(self):
        if self.dashboard:
            log_section(logger, "LAUNCHING DASHBOARD UI")
            self.dashboard.run_server(debug=True, use_reloader=False)

    def start(self):
        log_section(logger, "MARKET SIMULATION INITIALIZING")
        self.generate_agents()
        self.setup_environments()
        self.setup_dashboard()
        self.setup_database()

        # Start the dashboard in a separate thread
        if self.dashboard:
            dashboard_thread = threading.Thread(target=self.run_dashboard)
            dashboard_thread.start()

        # Run the simulation
        self.run_simulation()

        # Wait for the dashboard thread to finish if it's running
        if self.dashboard:
            dashboard_thread.join()

if __name__ == "__main__":
    config = OrchestratorConfig(
        num_agents=2,
        max_rounds=2,
        agent_config=AgentConfig(
            num_units=5,
            base_value=100,
            use_llm=True,
            initial_cash=1000,
            initial_goods=0,
            noise_factor=0.1,
            max_relative_spread=0.2
        ),
        llm_config=LLMConfig(
            client='openai',
            model='gpt-4o-mini',
            temperature=0.5,
            response_format='json_object',
            max_tokens=4096,
            use_cache=True
        ),
        environment_configs={
            'auction': AuctionConfig(
                name='Auction',
                address='auction_env_1',
                auction_type='double',
                max_steps=5,
            ),
        },
        protocol=ACLMessage,
        database_config={
            'db_type': 'postgres',
            'db_name': 'market_simulation'
        }
    )
    
    orchestrator = Orchestrator(config)
    orchestrator.start()
