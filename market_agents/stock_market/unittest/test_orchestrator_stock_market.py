import unittest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path
import asyncio
from market_agents.stock_market.orchestrator_stock_market import (
    Orchestrator, OrchestratorConfig, AgentConfig, StockMarketConfig, 
    LLMConfig, DatabaseConfig
)
from market_agents.stock_market.stock_models import OrderType, MarketAction
from market_agents.agents.personas.persona import Persona

class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        # Create minimal test configuration
        self.agent_config = AgentConfig(
            initial_cash_min=10000,
            initial_cash_max=20000,
            initial_stocks_min=10,
            initial_stocks_max=20,
            risk_aversion=0.5,
            expected_return=0.1,
            use_llm=True,
            stock_symbol="AAPL",
            max_relative_spread=0.1
        )

        self.llm_config = LLMConfig(
            name="test_llm",
            client="anthropic",
            model="claude-3-sonnet",
            temperature=0.7,
            max_tokens=1000,
            use_cache=True
        )

        self.stock_market_config = StockMarketConfig(
            name="test_market",
            address="test_address",
            max_rounds=10,
            stock_symbol="AAPL"
        )

        self.config = OrchestratorConfig(
            num_agents=2,
            max_rounds=10,
            agent_config=self.agent_config,
            llm_configs=[self.llm_config],
            environment_configs={"stock_market": self.stock_market_config},
            protocol="acl",
            database_config=DatabaseConfig(
                db_user="test_user",
                db_password="test_password"
            )
        )

        self.orchestrator = Orchestrator(self.config)

    @patch('market_agents.agents.personas.persona.generate_persona')
    def test_load_or_generate_personas(self, mock_generate_persona):
        # Mock persona generation
        mock_persona = Persona(
            name="Test Trader",
            background="Test background",
            trading_style="aggressive",
            risk_tolerance="high",
            investment_goals="growth"
        )
        mock_generate_persona.return_value = mock_persona

        # Test persona generation
        personas = self.orchestrator.load_or_generate_personas()
        
        self.assertEqual(len(personas), self.config.num_agents)
        self.assertIsInstance(personas[0], Persona)
        mock_generate_persona.assert_called()

    @patch('market_agents.agents.market_agent.MarketAgent.create')
    def test_generate_agents(self, mock_create_agent):
        # Mock agent creation
        mock_agent = Mock()
        mock_agent.id = "test_id"
        mock_create_agent.return_value = mock_agent

        # Test agent generation
        self.orchestrator.generate_agents()
        
        self.assertEqual(len(self.orchestrator.agents), self.config.num_agents)
        mock_create_agent.assert_called()

    def test_setup_environments(self):
        # Test environment setup
        self.orchestrator.setup_environments()
        
        self.assertIn('stock_market', self.orchestrator.environments)
        self.assertEqual(
            self.orchestrator.environments['stock_market'].name,
            self.config.environment_configs['stock_market'].name
        )

    @patch('market_agents.stock_market.insert_stock_simulation_data.SimulationDataInserter')
    def test_setup_database(self, mock_data_inserter):
        # Mock database setup
        mock_data_inserter.return_value.check_tables_exist.return_value = False
        
        # Test database setup
        self.orchestrator.setup_database()
        
        mock_data_inserter.return_value.check_tables_exist.assert_called_once()

class TestOrchestratorAsync(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Same setup as TestOrchestrator
        self.config = OrchestratorConfig(
            num_agents=2,
            max_rounds=10,
            agent_config=AgentConfig(
                initial_cash_min=10000,
                initial_cash_max=20000,
                initial_stocks_min=10,
                initial_stocks_max=20,
                risk_aversion=0.5,
                expected_return=0.1,
                use_llm=True,
                stock_symbol="AAPL",
                max_relative_spread=0.1
            ),
            llm_configs=[LLMConfig(
                name="test_llm",
                client="anthropic",
                model="claude-3-sonnet",
                temperature=0.7,
                max_tokens=1000,
                use_cache=True
            )],
            environment_configs={
                "stock_market": StockMarketConfig(
                    name="test_market",
                    address="test_address",
                    max_rounds=10,
                    stock_symbol="AAPL"
                )
            },
            protocol="acl",
            database_config=DatabaseConfig(
                db_user="test_user",
                db_password="test_password"
            )
        )
        self.orchestrator = Orchestrator(self.config)

    @patch('market_agents.stock_market.orchestrator_stock_market.ParallelAIUtilities')
    async def test_run_parallel_ai_completion(self, mock_ai_utils):
        # Mock AI completion
        mock_result = Mock()
        mock_ai_utils.return_value.run_parallel_ai_completion = AsyncMock(return_value=[mock_result])
        
        # Test AI completion
        prompts = [Mock()]
        results = await self.orchestrator.run_parallel_ai_completion(prompts)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], mock_result)

    @patch('market_agents.stock_market.orchestrator_stock_market.Orchestrator.run_environment')
    async def test_run_simulation(self, mock_run_environment):
        # Mock environment execution
        mock_run_environment.return_value = Mock()
        
        # Test simulation run
        await self.orchestrator.run_simulation()
        
        # Verify that run_environment was called max_rounds times
        self.assertEqual(
            mock_run_environment.call_count,
            self.config.max_rounds * len(self.orchestrator.simulation_order)
        )

if __name__ == '__main__':
    unittest.main()