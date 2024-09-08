import unittest
from unittest.mock import Mock, patch
import logging
from colorama import Fore, Style
from environments.auction.auction import DoubleAuction
from orchestrator import Orchestrator, OrchestratorConfig
from llm_agents.market_agent.market_agents import MarketAgent
from environments.auction.auction_environment import AuctionEnvironment
from protocols.acl_message import ACLMessage
from simulation_app import create_dashboard

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        logger.info(f"{Fore.CYAN}Setting up TestOrchestrator...{Style.RESET_ALL}")
        self.config = OrchestratorConfig(
            num_agents=5,
            max_rounds=3,
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
                }
            },
            protocol=ACLMessage,
            database_config={
                'db_type': 'postgres',
                'db_name': 'market_simulation'
            }
        )
        self.orchestrator = Orchestrator(self.config)
        logger.info(f"{Fore.GREEN}TestOrchestrator setup complete.{Style.RESET_ALL}")

    @patch('market_agent.market_agent_todo.MarketAgent.create')
    def test_generate_agents(self, mock_create):
        logger.info(f"{Fore.YELLOW}Testing generate_agents method...{Style.RESET_ALL}")
        logger.info("This test ensures that the correct number of agents are created and they are of the right type.")
        mock_agent = Mock(spec=MarketAgent)
        mock_create.return_value = mock_agent
        self.orchestrator.generate_agents()
        self.assertEqual(len(self.orchestrator.agents), self.config.num_agents)
        for agent in self.orchestrator.agents:
            self.assertIsInstance(agent, Mock)
            self.assertEqual(agent, mock_agent)
        logger.info(f"{Fore.GREEN}generate_agents test passed successfully.{Style.RESET_ALL}")

    @patch('orchestrator.AuctionEnvironment')
    def test_setup_environments(self, mock_auction_env):
        logger.info(f"{Fore.YELLOW}Testing setup_environments method...{Style.RESET_ALL}")
        logger.info("This test checks if the auction environment is set up correctly with the right parameters.")
        mock_env = Mock(spec=AuctionEnvironment)
        mock_auction_env.return_value = mock_env

        self.orchestrator.agents = [Mock(spec=MarketAgent) for _ in range(self.config.num_agents)]
        self.orchestrator.setup_environments()

        self.assertIn('auction', self.orchestrator.environments)
        self.assertEqual(self.orchestrator.environments['auction'], mock_env)
        
        mock_auction_env.assert_called_once_with(
            agents=self.orchestrator.agents,
            max_steps=5,
            protocol=self.config.protocol(),
            name='Auction',
            address='auction_env_1',
            auction_type='double'
        )
        logger.info(f"{Fore.GREEN}setup_environments test passed successfully.{Style.RESET_ALL}")

    @patch('orchestrator.create_dashboard')
    def test_setup_dashboard(self, mock_create_dashboard):
        logger.info(f"{Fore.YELLOW}Testing setup_dashboard method...{Style.RESET_ALL}")
        logger.info("This test ensures that the dashboard is created and set up correctly.")
        mock_dashboard = Mock()
        mock_create_dashboard.return_value = mock_dashboard
        self.orchestrator.setup_dashboard()
        self.assertEqual(self.orchestrator.dashboard, mock_dashboard)
        mock_create_dashboard.assert_called_once_with(self.orchestrator.environments)
        logger.info(f"{Fore.GREEN}setup_dashboard test passed successfully.{Style.RESET_ALL}")

    @patch.object(Orchestrator, 'run_environment')
    @patch.object(Orchestrator, 'save_round_data')
    @patch.object(Orchestrator, 'update_dashboard')
    def test_run_simulation(self, mock_update_dashboard, mock_save_round_data, mock_run_environment):
        logger.info(f"{Fore.YELLOW}Testing run_simulation method...{Style.RESET_ALL}")
        logger.info("This test checks if the simulation runs for the correct number of rounds and if all necessary methods are called.")
        self.orchestrator.agents = [Mock(spec=MarketAgent, id=f"agent_{i}") for i in range(self.config.num_agents)]
        self.orchestrator.environments = {'auction': Mock(spec=AuctionEnvironment)}
        self.orchestrator.run_simulation()
        
        self.assertEqual(mock_run_environment.call_count, self.config.max_rounds)
        self.assertEqual(mock_save_round_data.call_count, self.config.max_rounds)
        self.assertEqual(mock_update_dashboard.call_count, self.config.max_rounds)
        
        for agent in self.orchestrator.agents:
            self.assertEqual(agent.reflect.call_count, self.config.max_rounds)

        mock_run_environment.assert_called_with('auction')
        logger.info(f"{Fore.GREEN}run_simulation test passed successfully.{Style.RESET_ALL}")

    def test_run_environment(self):
        logger.info(f"{Fore.YELLOW}Testing run_environment method...{Style.RESET_ALL}")
        logger.info("This test ensures that the environment steps correctly and agents perceive and generate actions.")
        mock_env = Mock(spec=AuctionEnvironment)
        mock_env.step.return_value = Mock()
        mock_env.get_observation.return_value = Mock()
        self.orchestrator.environments = {'auction': mock_env}
        self.orchestrator.agents = [Mock(spec=MarketAgent, id=f"agent_{i}") for i in range(self.config.num_agents)]
        
        self.orchestrator.run_environment('auction')
        
        mock_env.step.assert_called_once()
        self.assertEqual(mock_env.get_observation.call_count, self.config.num_agents)
        self.assertEqual(mock_env.update.call_count, self.config.num_agents)
        for agent in self.orchestrator.agents:
            agent.perceive.assert_called_once()
            agent.generate_action.assert_called_once()
        logger.info(f"{Fore.GREEN}run_environment test passed successfully.{Style.RESET_ALL}")

if __name__ == '__main__':
    logger.info(f"{Fore.MAGENTA}Starting TestOrchestrator...{Style.RESET_ALL}")
    unittest.main()
    logger.info(f"{Fore.MAGENTA}TestOrchestrator completed.{Style.RESET_ALL}")