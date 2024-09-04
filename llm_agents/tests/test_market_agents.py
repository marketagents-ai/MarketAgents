import unittest
import logging
from market_agent.market_agents import MarketAgent
from econ_agents.econ_agent import create_economic_agent
from base_agent.agent import Agent as LLMAgent
from base_agent.aiutilities import LLMConfig
from market_agent.market_schemas import MarketActionSchema, DoubleAuctionMessage
from acl_message.acl_message import ACLMessage
import warnings

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestMarketAgent(unittest.TestCase):

    def setUp(self):
        # Suppress deprecation warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        # Create a real EconomicAgent
        self.econ_agent = create_economic_agent(
            agent_id=1,
            is_buyer=True,
            num_units=10,
            base_value=100,
            initial_cash=1000,
            initial_goods=0,
            utility_function_type="step",
            noise_factor=0.1,
            max_relative_spread=0.2
        )

        # Create a real LLMAgent
        llm_config = LLMConfig(
            client="openai",
            model="gpt-4-0613",
            response_format="json_object",
            temperature=0.7
        )
        self.llm_agent = LLMAgent(
            role="buyer",
            system="You are a buyer agent in a double auction market. Your goal is to maximize your profit.",
            llm_config=llm_config,
            output_format=MarketActionSchema.model_json_schema()
        )

        self.market_agent = MarketAgent.create(
            agent_id=1,
            is_buyer=True,
            num_units=10,
            base_value=100,
            use_llm=True,
            initial_cash=1000,
            initial_goods=0,
            llm_config=llm_config
        )

    def test_generate_llm_bid_normal_conditions(self):
        market_info = {
            "last_trade_price": 100,
            "average_price": 100,
            "total_trades": 10,
            "current_round": 1
        }
        bid = self.market_agent.generate_bid(market_info, round_num=1)
        logger.info(f"Normal conditions bid: {bid}")
        self.assertIsNotNone(bid)
        if bid:
            self.assertIsInstance(bid, ACLMessage)
            self.assertIsInstance(bid.content, DoubleAuctionMessage)
            self.assertEqual(bid.content.action, "bid")

    def test_generate_llm_bid_high_market_price(self):
        market_info = {
            "last_trade_price": 115,
            "average_price": 115,
            "total_trades": 10,
            "current_round": 1
        }
        bid = self.market_agent.generate_bid(market_info, round_num=1)
        logger.info(f"High market price bid: {bid}")
        self.assertIsNotNone(bid)
        if bid:
            self.assertIsInstance(bid, ACLMessage)
            self.assertIsInstance(bid.content, DoubleAuctionMessage)

    def test_generate_llm_bid_low_cash(self):
        self.market_agent.endowment.cash = 50
        market_info = {
            "last_trade_price": 100,
            "average_price": 100,
            "total_trades": 10,
            "current_round": 1
        }
        bid = self.market_agent.generate_bid(market_info, round_num=1)
        logger.info(f"Low cash bid: {bid}")
        if bid is None:
            logger.info("Bid is None. This is acceptable when cash is low.")
        else:
            logger.info(f"Bid price: {bid.content.price}, Cash: {self.market_agent.endowment.cash}")
            self.assertLessEqual(bid.content.price, 50, f"Bid price {bid.content.price} should be <= 50")
            self.assertGreater(bid.content.price, 0, "Bid price should be positive")
            self.assertGreater(bid.content.quantity, 0, "Bid quantity should be positive")

    def test_multiple_bids(self):
        market_info = {
            "last_trade_price": 100,
            "average_price": 100,
            "total_trades": 10,
            "current_round": 1
        }
        bids = []
        for i in range(3):
            bid = self.market_agent.generate_bid(market_info, round_num=i+1)
            logger.info(f"Multiple bids - Bid {i+1}: {bid}")
            if bid:
                bids.append(bid)
                self.market_agent.endowment.cash -= bid.content.price * bid.content.quantity
                self.market_agent.endowment.goods += bid.content.quantity
            logger.info(f"After bid {i+1}: Cash: {self.market_agent.endowment.cash}, Goods: {self.market_agent.endowment.goods}")
        
        self.assertTrue(len(bids) > 0)
        self.assertLess(self.market_agent.endowment.cash, 1000, "Cash should have decreased")

    def test_get_market_info(self):
        market_info = {
            "last_trade_price": 100,
            "average_price": 100,
            "total_trades": 10,
            "current_round": 1
        }
        info_str = self.market_agent._get_market_info(market_info)
        logger.info(f"Market info:\n{info_str}")
        self.assertIn(f"Current Cash: {self.market_agent.endowment.cash}", info_str)
        self.assertIn(f"Current Goods: {self.market_agent.endowment.goods}", info_str)
        self.assertIn("Last Trade Price:", info_str)
        self.assertIn("Base Value/Cost:", info_str)

if __name__ == '__main__':
    unittest.main()