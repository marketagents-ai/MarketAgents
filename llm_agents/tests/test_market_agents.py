import unittest
from market_agent.market_agents import MarketAgent
from base_agent.aiutilities import LLMConfig
from environments.auction.auction_environment import AuctionEnvironment
from protocols.acl_message import ACLMessage
import logging
import warnings
from colorama import Fore, Style
from datetime import datetime, timedelta
from econ_agents.econ_agent import create_economic_agent

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMarketAgentBase(unittest.TestCase):

    def setUp(self):
        # Suppress deprecation warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        self.llm_config = LLMConfig(
            client="openai",
            model="gpt-4o-mini",
            response_format="json_object",
            temperature=0.7
        )
        
        # Create a single market agent for testing
        self.agent = MarketAgent.create(
            agent_id=0,
            is_buyer=True,
            num_units=5,
            base_value=100.0,
            use_llm=True,
            initial_cash=1000,
            initial_goods=5,
            noise_factor=0.1,
            max_relative_spread=0.2,
            llm_config=self.llm_config,
            protocol=ACLMessage
        )

        self.auction_env = AuctionEnvironment(
            agents=[self.agent],
            max_steps=5,
            protocol=ACLMessage,
            name="TestAuction",
            address="test_auction_1",
            auction_type='double'
        )

        # Add the auction environment to the agent's environments
        self.agent.environments = {"auction": self.auction_env}

        # Add dummy memory for testing
        self.agent.memory = [
            {
                "type": "observation",
                "content": "The auction opened with 10 buyers and 10 sellers.",
                "timestamp": (datetime.now() - timedelta(minutes=30)).isoformat()
            },
            {
                "type": "action",
                "content": "Placed a bid for 2 units at $98 each.",
                "timestamp": (datetime.now() - timedelta(minutes=20)).isoformat()
            },
            {
                "type": "reflection",
                "content": "The last trade was at $100. I should consider increasing my bid price.",
                "strategy_update": "Increase bid price by 2%",
                "observation": {"last_trade_price": 100},
                "reward": 0,
                "timestamp": (datetime.now() - timedelta(minutes=10)).isoformat()
            }
        ]

    def test_create(self):
        self.assertIsInstance(self.agent, MarketAgent)
        self.assertTrue(self.agent.is_buyer)
        self.assertEqual(self.agent.id, "0")
        self.assertTrue(self.agent.use_llm)

    def test_generate_action(self):
        # Create a dummy auction state
        self.auction_env.current_step = 5
        
        perception = self.agent.perceive("auction")
        action = self.agent.generate_action("auction", perception)
        
        self.assertIsNotNone(action)
        self.assertIsInstance(action, dict)
        print(f"{Fore.GREEN}LLM output for generate_action: {action}{Style.RESET_ALL}")
        
        # Log and print the action prompt
        action_prompt = self.agent.prompt_manager.get_action_prompt({
            "environment_name": "auction",
            "environment_info": self.auction_env.get_global_state(),
            "recent_memories": self.agent.memory[-5:] if self.agent.memory else 'No recent memories',
            "observation": perception,
            "action_space": self.auction_env.get_action_space(),
            "last_action": self.agent.last_action
        })
        logger.info(f"Action prompt: {action_prompt}")
        print(f"{Fore.CYAN}Action prompt: {action_prompt}{Style.RESET_ALL}")

    def test_perceive(self):
        # Update the auction environment with some dummy data
        self.auction_env.current_step = 10
        
        perception = self.agent.perceive("auction")
        
        self.assertIsNotNone(perception)
        self.assertIsInstance(perception, dict)
        print(f"{Fore.BLUE}LLM output for perception: {perception}{Style.RESET_ALL}")
        
        # Log and print the perception prompt
        perception_prompt = self.agent.prompt_manager.get_perception_prompt({
            "environment_name": "auction",
            "environment_info": self.auction_env.get_global_state(),
            "recent_memories": self.agent.memory[-5:] if self.agent.memory else 'No recent memories'
        })
        logger.info(f"Perception prompt: {perception_prompt}")
        print(f"{Fore.MAGENTA}Perception prompt: {perception_prompt}{Style.RESET_ALL}")

    def test_reflect(self):
        self.agent.reflect("auction")
        
        self.assertGreater(len(self.agent.memory), 3)  # Now we expect at least 4 items in memory
        last_memory = self.agent.memory[-1]
        self.assertEqual(last_memory["type"], "reflection")
        print(f"{Fore.YELLOW}LLM output for memory update: {last_memory['content']}{Style.RESET_ALL}")
        
        # Log and print the reflection prompt
        reflection_prompt = self.agent.prompt_manager.get_reflection_prompt({
            "environment_name": "auction",
            "environment_info": self.auction_env.get_global_state(),
            "observation": self.auction_env.get_observation(self.agent.id),
            "last_action": self.agent.last_action,
            "reward": self.auction_env.get_observation(self.agent.id).content.get('reward', 0),
            "previous_strategy": self.agent.memory[-2].get('strategy_update', 'No previous strategy') if len(self.agent.memory) > 1 else 'No previous strategy'
        })
        logger.info(f"Reflection prompt: {reflection_prompt}")
        print(f"{Fore.WHITE}Reflection prompt: {reflection_prompt}{Style.RESET_ALL}")

if __name__ == '__main__':
    unittest.main()