import unittest
from market_agent.market_agent_todo import MarketAgent
from base_agent.aiutilities import LLMConfig
import logging
import warnings
import json
from colorama import Fore, Style
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMarketAgentBase(unittest.TestCase):

    def setUp(self):
        # Suppress deprecation warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        llm_config = LLMConfig(
            client="anthropic",
            model="claude-3-5-sonnet-20240620",
            response_format="json_beg",
            temperature=0.7
        )
        
        self.agent = MarketAgent.create(
            agent_id=1,
            is_buyer=True,
            num_units=5,
            base_value=100.0,
            use_llm=True,
            initial_cash=1000.0,
            initial_goods=0,
            noise_factor=0.1,
            max_relative_spread=0.2,
            llm_config=llm_config
        )

        # Add dummy memory for testing
        self.agent.memory = [
            {
                "type": "observation",
                "content": "The market opened with high volatility.",
                "timestamp": (datetime.now() - timedelta(minutes=30)).isoformat()
            },
            {
                "type": "action",
                "content": "Placed a buy order for 2 units at $98 each.",
                "timestamp": (datetime.now() - timedelta(minutes=20)).isoformat()
            },
            {
                "type": "reflection",
                "content": "The market seems to be trending upwards. I should consider increasing my bid price.",
                "observation": {"price_change": 2.5},
                "timestamp": (datetime.now() - timedelta(minutes=10)).isoformat()
            }
        ]

    def test_create(self):
        self.assertIsInstance(self.agent, MarketAgent)
        self.assertTrue(self.agent.is_buyer)
        self.assertEqual(self.agent.address, "agent_1_address")
        self.assertTrue(self.agent.use_llm)

    def test_generate_action(self):
        market_info = {"current_price": 100.0, "last_trade": 98.0}
        
        action = self.agent.generate_action(market_info)
        
        self.assertIsNotNone(action)
        self.assertIsInstance(action, str)
        print(f"{Fore.GREEN}LLM output for generate_action: {action}{Style.RESET_ALL}")

    def test_percieve(self):
        monologue = self.agent.percieve()
        
        self.assertIsNotNone(monologue)
        self.assertIsInstance(monologue, str)
        print(f"{Fore.BLUE}LLM output for monologue: {monologue}{Style.RESET_ALL}")

    def test_reflect(self):
        observation = {"price_change": 5.0}
        
        self.agent.reflect(observation)
        
        self.assertGreater(len(self.agent.memory), 3)  # Now we expect at least 4 items in memory
        last_memory = self.agent.memory[-1]
        self.assertEqual(last_memory["type"], "reflection")
        self.assertIsNotNone(last_memory["content"])
        self.assertEqual(last_memory["observation"], observation)
        print(f"{Fore.YELLOW}LLM output for memory update: {last_memory['content']}{Style.RESET_ALL}")

if __name__ == '__main__':
    unittest.main()
