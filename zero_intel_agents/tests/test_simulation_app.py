import unittest
from unittest.mock import patch, MagicMock
import dash
from dash import html
import plotly.graph_objs as go

# Import only the necessary functions
from simulation_app import (
    generate_market_state_table,
    generate_order_book_table,
    generate_trade_history_table,
    generate_supply_demand_chart,
    generate_price_vs_trade_chart,
    generate_cumulative_quantity_surplus_chart
)

class TestSimulationApp(unittest.TestCase):

    def test_generate_market_state_table(self):
        mock_env = MagicMock()
        mock_env.agents = [
            MagicMock(id=1, preference_schedule=MagicMock(is_buyer=True), 
                      allocation=MagicMock(goods=10, cash=100))
        ]
        mock_env.get_agent_utility.return_value = 50

        table = generate_market_state_table(mock_env)
        self.assertIsInstance(table, html.Table)

    def test_generate_order_book_table(self):
        mock_auction = MagicMock()
        mock_auction.get_order_book.return_value = [
            {'price': 100, 'shares': 10, 'total': 1000}
        ]

        table = generate_order_book_table(mock_auction)
        self.assertIsInstance(table, html.Table)

    def test_generate_trade_history_table(self):
        mock_auction = MagicMock()
        mock_auction.get_trade_history.return_value = [
            MagicMock(trade_id=1, buyer_id=1, seller_id=2, price=100, quantity=1)
        ]

        table = generate_trade_history_table(mock_auction)
        self.assertIsInstance(table, html.Table)

    @patch('simulation_app.env')
    def test_generate_supply_demand_chart(self, mock_env):
        mock_env.calculate_theoretical_supply_demand.return_value = ([0, 1], [10, 9], [0, 1], [8, 9])
        mock_env.calculate_equilibrium.return_value = (9, 1, 0, 0, 0)

        chart = generate_supply_demand_chart(mock_env)
        self.assertIsInstance(chart, go.Figure)

    @patch('simulation_app.auction')
    def test_generate_price_vs_trade_chart(self, mock_auction):
        mock_auction.successful_trades = [MagicMock(price=100), MagicMock(price=101)]

        chart = generate_price_vs_trade_chart()
        self.assertIsInstance(chart, go.Figure)

    @patch('simulation_app.auction')
    def test_generate_cumulative_quantity_surplus_chart(self, mock_auction):
        mock_auction.max_rounds = 2
        mock_auction.successful_trades = [
            MagicMock(round=1, quantity=1, buyer_value=110, price=100, seller_cost=90),
            MagicMock(round=2, quantity=1, buyer_value=120, price=105, seller_cost=95)
        ]

        chart = generate_cumulative_quantity_surplus_chart()
        self.assertIsInstance(chart, go.Figure)

if __name__ == '__main__':
    unittest.main()