import logging
from typing import List, Dict, Any
from environments.auction.auction import DoubleAuction
from pydantic import BaseModel, Field

from market_agent.market_agent_todo import MarketAgent
from environments.environment import Environment
from environments.auction.auction_environment import AuctionEnvironment
# from environments.group_chat.group_chat_environment import GroupChatEnvironment
# from environments.information_board.information_board_environment import InformationBoardEnvironment
from base_agent.aiutilities import LLMConfig
from protocols.acl_message import ACLMessage
from simulation_app import create_dashboard

logger = logging.getLogger(__name__)

class OrchestratorConfig(BaseModel):
    num_agents: int
    max_rounds: int
    agent_config: Dict[str, Any]
    llm_config: Dict[str, Any]
    environment_configs: Dict[str, Dict[str, Any]]
    protocol: Any
    database_config: Dict[str, Any]

class Orchestrator:
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.agents: List[MarketAgent] = []
        self.environments: Dict[str, Environment] = {}
        self.dashboard = None
        self.database = None
        #self.database: SimulationDatabase = None
        self.simulation_order = ['auction']
        #self.simulation_order = ['group_chat', 'information_board', 'auction']

    def generate_agents(self):
        llm_config = LLMConfig(**self.config.llm_config)
        
        for i in range(self.config.num_agents):
            agent = MarketAgent.create(
                agent_id=i,
                is_buyer=i % 2 == 0,  # Alternate between buyers and sellers
                **self.config.agent_config,
                llm_config=llm_config,
                protocol=self.config.protocol(),
                environments=self.environments
            )
            self.agents.append(agent)
        
        logger.info(f"Generated {len(self.agents)} agents")

    def setup_environments(self):
        for env_name, env_config in self.config.environment_configs.items():
            if env_name == 'auction':
                env = AuctionEnvironment(
                    agents=self.agents,
                    max_steps=env_config['max_steps'],
                    protocol=self.config.protocol(),
                    name=env_config['name'],
                    address=env_config['address'],
                    auction_type=env_config['auction_type']
                )
                self.environments[env_name] = env
            elif env_name == 'group_chat':
                # env = GroupChatEnvironment(
                #     agents=self.agents,
                #     name="Group Chat Environment",
                #     address=f"{env_name}_env_1",
                #     max_steps=1, 
                #     **env_config
                # )
                raise NotImplementedError("GroupChatEnvironment is not implemented yet.")
            elif env_name == 'information_board':
                # env = InformationBoardEnvironment(
                #     agents=self.agents,
                #     name="Information Board Environment",
                #     address=f"{env_name}_env_1",
                #     max_steps=1,
                #     **env_config
                # )
                raise NotImplementedError("InformationBoardEnvironment is not implemented yet.")
            else:
                raise ValueError(f"Unknown environment type: {env_name}")
            
            self.environments[env_name] = env

        logger.info(f"Set up {len(self.environments)} environments")

    def setup_dashboard(self):
        self.dashboard = create_dashboard(self.environments)
        logger.info("Dashboard setup complete")

    def setup_database(self):
        #self.database = SimulationDatabase(**self.config.database_config)
        #logger.info("Simulation database setup complete")
        pass

    def run_simulation(self):
        logger.info("Starting simulation")
        
        for round_num in range(1, self.config.max_rounds + 1):
            logger.info(f"Starting round {round_num}")
            
            # Run environments in order
            for env_name in self.simulation_order:
                self.run_environment(env_name)
            
            # Agent reflection after each round
            for agent in self.agents:
                agent.reflect()
            
            # Save round data to database
            self.save_round_data(round_num)
            
            # Update dashboard
            self.update_dashboard()
        
        logger.info("Simulation completed")

    def run_environment(self, env_name: str):
        env = self.environments[env_name]
        if env_name in ['group_chat', 'information_board']:
            raise NotImplementedError(f"{env_name} environment is not implemented yet.")
        
        env.step()
        logger.info(f"Completed {env_name} step")
        
        for agent in self.agents:
            observation = env.get_observation(agent.id)
            agent.perceive(env_name, observation)
            action = agent.generate_action(env_name)
            env.update({agent.id: action})

    def save_round_data(self, round_num: int):
        round_data = {
            'round': round_num,
            'global_states': {env_name: env.get_global_state() for env_name, env in self.environments.items()},
            'agent_states': [agent.get_state() for agent in self.agents]
        }
        self.database.save_round_data(round_data)

    def update_dashboard(self):
        if self.dashboard:
            self.dashboard.update_data(self.environments)

    def start(self):
        self.generate_agents()
        self.setup_environments()
        self.setup_dashboard()
        self.setup_database()
        self.run_simulation()
        
        if self.dashboard:
            logger.info("Starting dashboard")
            self.dashboard.run_server(debug=True)

if __name__ == "__main__":
    config = OrchestratorConfig(
        num_agents=10,
        max_rounds=5,
        agent_config={
            'num_units': 5,
            'base_value': 100,
            'use_llm': True,
            'initial_cash': 1000,
            'initial_goods': 0,
            'noise_factor': 0.1,
            'max_relative_spread': 0.2
        },
        llm_config={
            'client': 'openai',
            'model': 'gpt-4-0613',
            'temperature': 0.5,
            'response_format': 'json_object',
            'max_tokens': 4096,
            'use_cache': True
        },
        environment_configs={
            'auction': {
                'name': 'Auction',
                'address': 'auction_env_1',
                'auction_type': 'double',
                'max_steps': 5,
            },
            'group_chat': {
                'name': 'Group Chat',
                'address': 'group_chat_env_1',
                'max_messages_per_round': 5,
                'speaker_selection': 'round_robin',
                'chat_duration': 300,
                'max_steps': 1,
            },
            'information_board': {
                'name': 'Information Board',
                'address': 'info_board_env_1',
                'update_frequency': 2, 
                'max_steps': 1,
            }
        },
        protocol=ACLMessage,
        database_config={
            'db_type': 'postgres',
            'db_name': 'market_simulation'
        }
    )
    
    orchestrator = Orchestrator(config)
    orchestrator.start()


