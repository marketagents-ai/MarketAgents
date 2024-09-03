import unittest
from unittest.mock import patch, MagicMock
from ziagents import PreferenceSchedule, ZIAgent, Allocation, Order, Trade, run_simulation

class TestPreferenceSchedule(unittest.TestCase):
    def test_generate(self):
        schedule = PreferenceSchedule.generate(is_buyer=True, num_units=5, base_value=100)
        self.assertTrue(schedule.is_buyer)
        self.assertEqual(len(schedule.values), 5)
        self.assertGreater(schedule.initial_endowment, 0)

    def test_get_value(self):
        schedule = PreferenceSchedule(values={1: 100, 2: 90, 3: 80}, is_buyer=True, initial_endowment=300)
        self.assertEqual(schedule.get_value(1), 100)
        self.assertEqual(schedule.get_value(2), 90)
        self.assertEqual(schedule.get_value(4), 0)  # Non-existent quantity

class TestZIAgent(unittest.TestCase):
    def setUp(self):
        self.buyer_agent = ZIAgent.generate(agent_id=1, is_buyer=True, num_units=5, base_value=100)
        self.seller_agent = ZIAgent.generate(agent_id=2, is_buyer=False, num_units=5, base_value=50)

    def test_generate(self):
        self.assertEqual(self.buyer_agent.id, 1)
        self.assertTrue(self.buyer_agent.preference_schedule.is_buyer)
        self.assertEqual(self.seller_agent.id, 2)
        self.assertFalse(self.seller_agent.preference_schedule.is_buyer)

    def test_calculate_trade_surplus(self):
        trade = Trade(
            trade_id=1,
            buyer_id=1,
            seller_id=2,
            quantity=1,
            price=75,
            buyer_value=100,
            seller_cost=50,
            round=1
        )
        buyer_surplus = self.buyer_agent.calculate_trade_surplus(trade)
        seller_surplus = self.seller_agent.calculate_trade_surplus(trade)
        self.assertEqual(buyer_surplus, 25)
        self.assertEqual(seller_surplus, 25)

    def test_generate_bid(self):
        buyer_bid = self.buyer_agent.generate_bid()
        seller_bid = self.seller_agent.generate_bid()
        self.assertIsNotNone(buyer_bid)
        self.assertIsNotNone(seller_bid)
        self.assertTrue(buyer_bid.is_buy)
        self.assertFalse(seller_bid.is_buy)

    def test_finalize_trade(self):
            initial_buyer_cash = self.buyer_agent.allocation.cash
            initial_seller_goods = self.seller_agent.allocation.goods
            trade = Trade(
                trade_id=1,
                buyer_id=1,
                seller_id=2,
                quantity=1,
                price=75,
                buyer_value=100,
                seller_cost=50,
                round=1
            )
            self.buyer_agent.finalize_trade(trade)
            self.seller_agent.finalize_trade(trade)
            self.assertAlmostEqual(self.buyer_agent.allocation.cash, initial_buyer_cash - 75, places=7)
            self.assertEqual(self.buyer_agent.allocation.goods, 1)
            self.assertEqual(self.seller_agent.allocation.cash, 75)
            self.assertEqual(self.seller_agent.allocation.goods, initial_seller_goods - 1)

    def test_respond_to_order(self):
        order = Order(agent_id=1, is_buy=True, quantity=1, price=100)
        self.buyer_agent.respond_to_order(order, accepted=True)
        self.assertEqual(len(self.buyer_agent.posted_orders), 1)
        self.buyer_agent.respond_to_order(order, accepted=False)
        self.assertEqual(len(self.buyer_agent.rejected_orders), 1)

    def test_calculate_individual_surplus(self):
        # Set up a simple scenario
        self.buyer_agent.allocation.goods = 2
        self.buyer_agent.allocation.cash = 80
        self.buyer_agent.allocation.initial_cash = 100
        self.buyer_agent.preference_schedule.values = {1: 60, 2: 50}
        
        surplus = self.buyer_agent.calculate_individual_surplus()
        expected_surplus = (60 + 50) - (100 - 80)  # Total value of goods - cash spent
        self.assertEqual(surplus, expected_surplus)

class TestRunSimulation(unittest.TestCase):
    @patch('ziagents.ZIAgent.generate_bid')
    @patch('ziagents.ZIAgent.finalize_trade')
    @patch('ziagents.ZIAgent.respond_to_order')
    def test_run_simulation(self, mock_respond, mock_finalize, mock_generate_bid):
        agent = ZIAgent.generate(agent_id=1, is_buyer=True, num_units=5, base_value=100)
        mock_bid = MagicMock(agent_id=1, is_buy=True, quantity=1, price=90)
        mock_generate_bid.return_value = mock_bid
        
        # Instead of mocking get_value, we'll set a value in the values dictionary
        agent.preference_schedule.values = {1: 100}  # Assuming the key is the quantity
        
        run_simulation(agent, num_rounds=10)
        
        self.assertEqual(mock_generate_bid.call_count, 10)
        self.assertEqual(mock_finalize.call_count, 10)
        self.assertEqual(mock_respond.call_count, 10)

if __name__ == '__main__':
    unittest.main()