import unittest
from unittest.mock import Mock, patch
import logging
from colorama import Fore, Style
from market_agents.environments.mechanisms.auction import DoubleAuction
from market_agents.orchestrator import Orchestrator, OrchestratorConfig, AgentConfig, AuctionConfig, LLMConfig
from market_agents.agents.market_agent import MarketAgent
from market_agents.environments.environment import MultiAgentEnvironment
from market_agents.agents.protocols.acl_message import ACLMessage
from market_agents.simulation_app import create_dashboard

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        logger.info(f"{Fore.CYAN}Setting up TestOrchestrator...{Style.RESET_ALL}")
        self.config = OrchestratorConfig(
            num_agents=5,
            max_rounds=3,
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
                model='gpt-4-0613',
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
                )
            },
            protocol=ACLMessage,
            database_config={
                'db_type': 'postgres',
                'db_name': 'market_simulation'
            }
        )
        self.orchestrator = Orchestrator(self.config)
        logger.info(f"{Fore.GREEN}TestOrchestrator setup complete.{Style.RESET_ALL}")

    @patch('market_agents.agents.market_agent.MarketAgent')
    def test_generate_agents(self, mock_market_agent):
        logger.info(f"{Fore.YELLOW}Testing generate_agents method...{Style.RESET_ALL}")
        logger.info("This test ensures that the correct number of agents are created and they are of the right type.")
        mock_agent = Mock(spec=MarketAgent)
        mock_market_agent.return_value = mock_agent
        self.orchestrator.generate_agents()
        self.assertEqual(len(self.orchestrator.agents), self.config.num_agents)
        for agent in self.orchestrator.agents:
            self.assertIsInstance(agent, Mock)
            self.assertEqual(agent, mock_agent)
        logger.info(f"{Fore.GREEN}generate_agents test passed successfully.{Style.RESET_ALL}")

    @patch('market_agents.environments.environment.MultiAgentEnvironment')
    @patch('market_agents.environments.mechanisms.auction.DoubleAuction')
    def test_setup_environments(self, mock_double_auction, mock_multi_agent_env):
        logger.info(f"{Fore.YELLOW}Testing setup_environments method...{Style.RESET_ALL}")
        logger.info("This test checks if the auction environment is set up correctly with the right parameters.")
        mock_env = Mock(spec=MultiAgentEnvironment)
        mock_multi_agent_env.return_value = mock_env
        mock_mechanism = Mock(spec=DoubleAuction)
        mock_double_auction.return_value = mock_mechanism

        self.orchestrator.agents = [Mock(spec=MarketAgent) for _ in range(self.config.num_agents)]
        self.orchestrator.setup_environments()

        self.assertIn('auction', self.orchestrator.environments)
        self.assertEqual(self.orchestrator.environments['auction'], mock_env)
        
        mock_double_auction.assert_called_once_with(
            max_rounds=100,
            good_name='apple'
        )
        mock_multi_agent_env.assert_called_once_with(
            name='Apple Market',
            address='apple_market',
            max_steps=100,
            mechanism=mock_mechanism
        )
        logger.info(f"{Fore.GREEN}setup_environments test passed successfully.{Style.RESET_ALL}")

    @patch('market_agents.simulation_app.create_dashboard')
    def test_setup_dashboard(self, mock_create_dashboard):
        logger.info(f"{Fore.YELLOW}Testing setup_dashboard method...{Style.RESET_ALL}")
        logger.info("This test ensures that the dashboard is created and set up correctly.")
        mock_dashboard = Mock()
        mock_create_dashboard.return_value = mock_dashboard
        self.orchestrator.setup_dashboard()
        self.assertEqual(self.orchestrator.dashboard, mock_dashboard)
        mock_create_dashboard.assert_called_once_with(self.orchestrator.environments)
        logger.info(f"{Fore.GREEN}setup_dashboard test passed successfully.{Style.RESET_ALL}")

    @patch.object(Orchestrator, 'run_auction_step')
    def test_run_simulation(self, mock_run_auction_step):
        logger.info(f"{Fore.YELLOW}Testing run_simulation method...{Style.RESET_ALL}")
        logger.info("This test checks if the simulation runs for the correct number of rounds and if all necessary methods are called.")
        self.orchestrator.agents = [Mock(spec=MarketAgent, id=f"agent_{i}") for i in range(self.config.num_agents)]
        self.orchestrator.environments = {'auction': Mock(spec=MultiAgentEnvironment)}
        self.orchestrator.run_simulation()
        
        self.assertEqual(mock_run_auction_step.call_count, self.config.max_rounds)
        
        mock_run_auction_step.assert_called_with('apple')
        logger.info(f"{Fore.GREEN}run_simulation test passed successfully.{Style.RESET_ALL}")

    def test_run_auction_step(self):
        logger.info(f"{Fore.YELLOW}Testing run_auction_step method...{Style.RESET_ALL}")
        logger.info("This test ensures that the environment steps correctly and agents generate actions.")
        mock_env = Mock(spec=MultiAgentEnvironment)
        mock_env.step.return_value = Mock()
        self.orchestrator.environments = {'auction': mock_env}
        self.orchestrator.agents = [Mock(spec=MarketAgent, id=f"agent_{i}") for i in range(self.config.num_agents)]
        
        self.orchestrator.run_auction_step('apple')
        
        mock_env.step.assert_called_once()
        for agent in self.orchestrator.agents:
            agent.generate_action.assert_called_once()
        logger.info(f"{Fore.GREEN}run_auction_step test passed successfully.{Style.RESET_ALL}")

if __name__ == '__main__':
    logger.info(f"{Fore.MAGENTA}Starting TestOrchestrator...{Style.RESET_ALL}")
    unittest.main()
    logger.info(f"{Fore.MAGENTA}TestOrchestrator completed.{Style.RESET_ALL}")