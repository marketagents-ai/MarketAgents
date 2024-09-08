import unittest
from unittest.mock import MagicMock, patch
<<<<<<< HEAD
from llm_agents.environments.auction.auction_environment import Environment, generate_market_agents
=======
from llm_agents.environments.environment import Environment, generate_market_agents
>>>>>>> main
import math

class TestEnvironment(unittest.TestCase):

    def setUp(self):
        self.agents = generate_market_agents(num_agents=10, num_units=5, buyer_base_value=100, seller_base_value=80, spread=0.5)
        self.env = Environment(agents=self.agents)

    def test_initialization(self):
        self.assertEqual(len(self.env.agents), 10)
        self.assertEqual(len(self.env.buyers), 5)
        self.assertEqual(len(self.env.sellers), 5)

    def test_get_agent(self):
        agent = self.env.get_agent(0)
        self.assertIsNotNone(agent)
        self.assertEqual(agent.id, 0)

        non_existent_agent = self.env.get_agent(100)
        self.assertIsNone(non_existent_agent)

    def test_total_utility(self):
        total_utility = self.env.total_utility
        self.assertGreaterEqual(round(total_utility, 10), 0)

    def test_remaining_trade_opportunities(self):
        remaining_trades = self.env.remaining_trade_opportunities
        self.assertGreaterEqual(remaining_trades, 0)

    def test_remaining_surplus(self):
        remaining_surplus = self.env.remaining_surplus
        self.assertGreaterEqual(round(remaining_surplus, 10), 0)

    def test_initial_demand_curve(self):
        demand_curve = self.env.initial_demand_curve
        self.assertIsNotNone(demand_curve)
        self.assertGreater(len(demand_curve.points), 0)

    def test_initial_supply_curve(self):
        supply_curve = self.env.initial_supply_curve
        self.assertIsNotNone(supply_curve)
        self.assertGreater(len(supply_curve.points), 0)

    def test_current_demand_curve(self):
        demand_curve = self.env.current_demand_curve
        self.assertIsNotNone(demand_curve)
        self.assertGreater(len(demand_curve.points), 0)

    def test_current_supply_curve(self):
        supply_curve = self.env.current_supply_curve
        self.assertIsNotNone(supply_curve)
        self.assertGreater(len(supply_curve.points), 0)

    def test_calculate_equilibrium(self):
        ce_price, ce_quantity, buyer_surplus, seller_surplus, total_surplus = self.env.calculate_equilibrium()
        
        self.assertGreater(round(ce_price, 10), 0)
        self.assertGreater(ce_quantity, 0)
        self.assertGreaterEqual(round(buyer_surplus, 10), 0)
        self.assertGreaterEqual(round(seller_surplus, 10), 0)
        self.assertGreater(round(total_surplus, 10), 0)

    def test_efficiency(self):
        efficiency = self.env.efficiency
        self.assertGreaterEqual(round(efficiency, 10), 0)
        self.assertLessEqual(efficiency, 1)

if __name__ == '__main__':
    unittest.main()