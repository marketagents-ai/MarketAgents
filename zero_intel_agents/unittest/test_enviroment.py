import unittest
from unittest.mock import patch, MagicMock
from environment import Environment, ZIAgent, PreferenceSchedule, Allocation, generate_agents

class TestEnvironment(unittest.TestCase):

    def setUp(self):
        self.agents = generate_agents(num_agents=10, num_units=5, buyer_base_value=100, seller_base_value=80, spread=0.5)
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

    def test_calculate_initial_utility(self):
        buyer = self.env.buyers[0]
        seller = self.env.sellers[0]

        buyer_utility = self.env.calculate_initial_utility(buyer)
        seller_utility = self.env.calculate_initial_utility(seller)

        self.assertGreater(buyer_utility, 0)
        self.assertGreater(seller_utility, 0)

    def test_get_agent_utility(self):
        buyer = self.env.buyers[0]
        seller = self.env.sellers[0]

        buyer_utility = self.env.get_agent_utility(buyer)
        seller_utility = self.env.get_agent_utility(seller)

        self.assertGreater(buyer_utility, 0)
        self.assertGreater(seller_utility, 0)

    def test_get_total_utility(self):
        total_utility = self.env.get_total_utility()
        self.assertGreater(total_utility, 0)

    def test_calculate_remaining_trade_opportunities(self):
        remaining_trades = self.env.calculate_remaining_trade_opportunities()
        self.assertGreaterEqual(remaining_trades, 0)

    def test_calculate_remaining_surplus(self):
        remaining_surplus = self.env.calculate_remaining_surplus()
        self.assertGreaterEqual(remaining_surplus, 0)

    def test_calculate_theoretical_supply_demand(self):
        demand_x, demand_y, supply_x, supply_y = self.env.calculate_theoretical_supply_demand()
        
        self.assertGreater(len(demand_x), 0)
        self.assertGreater(len(demand_y), 0)
        self.assertGreater(len(supply_x), 0)
        self.assertGreater(len(supply_y), 0)
        
        self.assertEqual(len(demand_x), len(demand_y))
        self.assertEqual(len(supply_x), len(supply_y))

    def test_calculate_equilibrium(self):
        ce_price, ce_quantity, buyer_surplus, seller_surplus, total_surplus = self.env.calculate_equilibrium()
        
        self.assertGreater(ce_price, 0)
        self.assertGreater(ce_quantity, 0)
        self.assertGreaterEqual(buyer_surplus, 0)
        self.assertGreaterEqual(seller_surplus, 0)
        self.assertGreater(total_surplus, 0)

    @patch('matplotlib.pyplot.savefig')
    def test_plot_theoretical_supply_demand(self, mock_savefig):
        fig = self.env.plot_theoretical_supply_demand(save_location='/tmp')
        self.assertIsNotNone(fig)
        mock_savefig.assert_called_once()

if __name__ == '__main__':
    unittest.main()