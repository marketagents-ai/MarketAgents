import unittest
from unittest.mock import patch, MagicMock
from dash.testing.application_runners import import_app
from dash.testing.composite import DashComposite
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go

# Import your app
from app import app, update_dashboard, generate_market_state_table, generate_order_book_table, generate_trade_history_table, generate_supply_demand_chart, generate_price_vs_trade_chart, generate_cumulative_quantity_surplus_chart

class TestDashApp(unittest.TestCase):

    def setUp(self):
        self.app = import_app(app)
        self.dash_duo = DashComposite(self.app)

    def test_app_initialization(self):
        self.dash_duo.start_server()
        self.assertIn("Market Simulation Live Dashboard", self.dash_duo.find_element("h1").text)

    @patch('app.auction')
    @patch('app.env')
    def test_update_dashboard(self, mock_env, mock_auction):
        # Mock auction and environment
        mock_auction.current_round = 0
        mock_auction.max_rounds = 1
        mock_auction.successful_trades = [MagicMock(price=100, quantity=1)]
        mock_auction.total_surplus_extracted = 10

        # Test update_dashboard function
        result = update_dashboard(1)
        
        self.assertEqual(len(result), 6)  # Check if function returns 6 outputs
        self.assertIsInstance(result[0], str)  # Market state table
        self.assertIsInstance(result[1], str)  # Order book table
        self.assertIsInstance(result[2], str)  # Trade history table
        self.assertIsInstance(result[3], go.Figure)  # Supply demand chart
        self.assertIsInstance(result[4], go.Figure)  # Price vs trade chart
        self.assertIsInstance(result[5], go.Figure)  # Cumulative quantity surplus chart

    def test_generate_market_state_table(self):
        mock_env = MagicMock()
        mock_env.agents = [
            MagicMock(id=1, preference_schedule=MagicMock(is_buyer=True), 
                      allocation=MagicMock(goods=10, cash=100))
        ]
        mock_env.get_agent_utility.return_value = 50

        table = generate_market_state_table(mock_env)
        self.assertIn("Agent ID", str(table))
        self.assertIn("Role", str(table))
        self.assertIn("Goods", str(table))
        self.assertIn("Cash", str(table))
        self.assertIn("Utility", str(table))

    def test_generate_order_book_table(self):
        mock_auction = MagicMock()
        mock_auction.get_order_book.return_value = [
            {'price': 100, 'shares': 10, 'total': 1000}
        ]

        table = generate_order_book_table(mock_auction)
        self.assertIn("Price", str(table))
        self.assertIn("Shares", str(table))
        self.assertIn("Total", str(table))

    def test_generate_trade_history_table(self):
        mock_auction = MagicMock()
        mock_auction.get_trade_history.return_value = [
            MagicMock(trade_id=1, buyer_id=1, seller_id=2, price=100, quantity=1)
        ]

        table = generate_trade_history_table(mock_auction)
        self.assertIn("Trade ID", str(table))
        self.assertIn("Buyer ID", str(table))
        self.assertIn("Seller ID", str(table))
        self.assertIn("Price", str(table))
        self.assertIn("Quantity", str(table))

    @patch('app.env')
    def test_generate_supply_demand_chart(self, mock_env):
        mock_env.calculate_theoretical_supply_demand.return_value = ([0, 1], [10, 9], [0, 1], [8, 9])
        mock_env.calculate_equilibrium.return_value = (9, 1, 0, 0, 0)

        chart = generate_supply_demand_chart(mock_env)
        self.assertIsInstance(chart, go.Figure)
        self.assertEqual(len(chart.data), 4)  # Demand, Supply, CE Quantity, CE Price

    @patch('app.auction')
    def test_generate_price_vs_trade_chart(self, mock_auction):
        mock_auction.successful_trades = [MagicMock(price=100), MagicMock(price=101)]

        chart = generate_price_vs_trade_chart()
        self.assertIsInstance(chart, go.Figure)
        self.assertEqual(len(chart.data), 1)

    @patch('app.auction')
    def test_generate_cumulative_quantity_surplus_chart(self, mock_auction):
        mock_auction.max_rounds = 2
        mock_auction.successful_trades = [
            MagicMock(round=1, quantity=1, buyer_value=110, price=100, seller_cost=90),
            MagicMock(round=2, quantity=1, buyer_value=120, price=105, seller_cost=95)
        ]

        chart = generate_cumulative_quantity_surplus_chart()
        self.assertIsInstance(chart, go.Figure)
        self.assertEqual(len(chart.data), 2)  # Cumulative Quantity and Cumulative Surplus

if __name__ == '__main__':
    unittest.main()