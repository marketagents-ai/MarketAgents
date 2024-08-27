import unittest
from market_agents import MarketAgent
from ziagents import ZIAgent, PreferenceSchedule, Allocation
from agents import Agent as LLMAgent

class TestMarketAgent(unittest.TestCase):

    def setUp(self):
        # Create a real ZIAgent
        preference_schedule = PreferenceSchedule.generate(is_buyer=True, num_units=10, base_value=100)
        allocation = Allocation(cash=1000, goods=0, initial_cash=1000, initial_goods=0)
        self.zi_agent = ZIAgent(id=1, preference_schedule=preference_schedule, allocation=allocation)

        # Create a real LLMAgent
        llm_config = {
            "client": "openai",  # or "azure_openai" depending on your setup
            "model": "gpt-4o-mini",  # or your preferred model
            "response_format": {"type": "json_object"},
            "temperature": 0.7
        }
        self.llm_agent = LLMAgent(
            role="buyer",
            system="You are a buyer agent in a double auction market. Your goal is to maximize your profit. You must respond in JSON",
            llm_config=llm_config,
            output_format="MarketActionSchema"
        )

        self.market_agent = MarketAgent(zi_agent=self.zi_agent, llm_agent=self.llm_agent, use_llm=True, llm_threshold=1.0)

    def test_generate_llm_bid_normal_conditions(self):
        bid = self.market_agent._generate_llm_bid()
        print(f"Normal conditions bid: {bid}")
        self.assertIsNotNone(bid)
        if bid:
            self.assertTrue(hasattr(bid, 'price'))
            self.assertTrue(hasattr(bid, 'quantity'))
            self.assertTrue(bid.is_buy)

    def test_generate_llm_bid_high_market_price(self):
        # Simulate a high last trade price
        self.zi_agent.posted_orders.append(self.zi_agent.generate_bid())
        self.zi_agent.posted_orders[-1].price = 115  # Set a high price

        bid = self.market_agent._generate_llm_bid()
        print(f"High market price bid: {bid}")
        self.assertIsNotNone(bid)
        if bid:
            self.assertTrue(hasattr(bid, 'price'))
            self.assertTrue(hasattr(bid, 'quantity'))

    def test_generate_llm_bid_low_cash(self):
        # Simulate low cash scenario
        self.zi_agent.allocation.cash = 50
        bid = self.market_agent._generate_llm_bid()
        print(f"Low cash bid: {bid}")
        # The LLM might decide to hold or place a low bid
        if bid is not None:
            self.assertTrue(bid.price <= 50)

    def test_multiple_bids(self):
        # Test generating multiple bids to see how the agent adapts
        bids = []
        for i in range(3):
            bid = self.market_agent._generate_llm_bid()
            print(f"Multiple bids - Bid {i+1}: {bid}")
            if bid:
                bids.append(bid)
                self.zi_agent.posted_orders.append(bid)
                self.zi_agent.allocation.cash -= bid.price * bid.quantity
                self.zi_agent.allocation.goods += bid.quantity
        
        self.assertTrue(len(bids) > 0)
        self.assertLess(self.zi_agent.allocation.cash, 1000)  # Cash should have decreased

    def test_get_market_info(self):
        market_info = self.market_agent._get_market_info()
        print(f"Market info:\n{market_info}")
        self.assertIn("Role: Buyer", market_info)
        self.assertIn(f"Current Cash: {self.zi_agent.allocation.cash}", market_info)
        self.assertIn(f"Current Goods: {self.zi_agent.allocation.goods}", market_info)
        self.assertIn("Last Trade Price:", market_info)
        self.assertIn("Base Value/Cost:", market_info)

if __name__ == '__main__':
    unittest.main()