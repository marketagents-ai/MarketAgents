import unittest
import logging
from econ_agents.econ_agent import EconomicAgent, create_economic_agent, UtilityFunction, Endowment, PreferenceSchedule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestEconomicAgent(unittest.TestCase):
    def setUp(self):
        logger.info("Setting up test agent...")
        self.agent = create_economic_agent(
            agent_id=1,
            is_buyer=True,
            num_units=10,
            base_value=100.0,
            initial_cash=1000.0,
            initial_goods=0,
            utility_function_type="cobb-douglas",
            noise_factor=0.1,
            max_relative_spread=0.2
        )
        logger.info(f"Test agent created: {self.agent}")

    def test_agent_initialization(self):
        logger.info("Testing agent initialization...")
        self.assertIsInstance(self.agent, EconomicAgent)
        self.assertEqual(self.agent.id, 1)
        self.assertTrue(self.agent.is_buyer)
        self.assertIsInstance(self.agent.preference_schedule, PreferenceSchedule)
        self.assertIsInstance(self.agent.endowment, Endowment)
        self.assertIsInstance(self.agent.utility_function, UtilityFunction)
        logger.info("Agent initialization test passed.")

    def test_generate_bid(self):
        logger.info("Testing bid generation...")
        bid = self.agent.generate_bid()
        logger.info(f"Generated bid: {bid}")
        self.assertIsNotNone(bid)
        self.assertIn('price', bid)
        self.assertIn('quantity', bid)
        self.assertLessEqual(bid['price'], self.agent.endowment.cash)
        self.assertEqual(bid['quantity'], 1)
        logger.info("Bid generation test passed.")

    def test_generate_ask_for_buyer(self):
        logger.info("Testing ask generation for buyer...")
        ask = self.agent.generate_ask()
        logger.info(f"Generated ask: {ask}")
        self.assertIsNone(ask)
        logger.info("Ask generation for buyer test passed.")

    def test_finalize_trade(self):
        logger.info("Testing trade finalization...")
        initial_cash = self.agent.endowment.cash
        initial_goods = self.agent.endowment.goods
        trade = {
            'buyer_id': self.agent.id,
            'price': 50.0,
            'quantity': 1
        }
        logger.info(f"Initial state: cash={initial_cash}, goods={initial_goods}")
        logger.info(f"Finalizing trade: {trade}")
        self.agent.finalize_trade(trade)
        logger.info(f"Final state: cash={self.agent.endowment.cash}, goods={self.agent.endowment.goods}")
        self.assertEqual(self.agent.endowment.cash, initial_cash - 50.0)
        self.assertEqual(self.agent.endowment.goods, initial_goods + 1)
        logger.info("Trade finalization test passed.")

    def test_calculate_utility(self):
        logger.info("Testing utility calculation...")
        utility = self.agent.calculate_utility()
        logger.info(f"Calculated utility: {utility}")
        self.assertGreaterEqual(utility, 0)
        logger.info("Utility calculation test passed.")

    def test_calculate_individual_surplus(self):
        logger.info("Testing individual surplus calculation...")
        surplus = self.agent.calculate_individual_surplus()
        logger.info(f"Calculated individual surplus: {surplus}")
        self.assertIsInstance(surplus, float)
        logger.info("Individual surplus calculation test passed.")

    def test_current_quantity(self):
        logger.info("Testing current quantity calculation...")
        self.assertEqual(self.agent.current_quantity, 0)
        logger.info(f"Initial current quantity: {self.agent.current_quantity}")
        self.agent.endowment.goods = 5
        logger.info(f"Updated current quantity: {self.agent.current_quantity}")
        self.assertEqual(self.agent.current_quantity, 5)
        logger.info("Current quantity calculation test passed.")

    def test_base_value(self):
        logger.info("Testing base value calculation...")
        base_value = self.agent.base_value
        logger.info(f"Calculated base value: {base_value}")
        self.assertIsInstance(base_value, float)
        self.assertGreater(base_value, 0)
        logger.info("Base value calculation test passed.")

if __name__ == '__main__':
    unittest.main()
