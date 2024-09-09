import unittest
from unittest.mock import Mock, PropertyMock
from simulation_app import (
    generate_market_state_table,
    generate_order_book_table,
    generate_trade_history_table,
    generate_supply_demand_chart,
    generate_price_vs_trade_chart,
    generate_cumulative_quantity_surplus_chart
)

class TestSimulationApp(unittest.TestCase):

    def setUp(self):
        # Create mock objects for testing
        self.mock_environment = Mock()
        self.mock_order_book = Mock()
        self.mock_trade_history = []
        self.mock_successful_trades = []
        self.mock_auction = Mock()

        # Set up mock data
        self.mock_environment.get_global_state.return_value = {
            'current_step': 1,
            'max_steps': 5,
            'ce_price': 100,
            'ce_quantity': 10
        }
        self.mock_environment.current_supply_curve.get_x_y_values.return_value = ([0, 1], [0, 100])
        self.mock_environment.current_demand_curve.get_x_y_values.return_value = ([0, 1], [100, 0])
        
        # Mock bids and asks
        mock_bid = Mock()
        type(mock_bid).agent_id = PropertyMock(return_value=1)
        type(mock_bid.market_action).price = PropertyMock(return_value=90)
        type(mock_bid.market_action).quantity = PropertyMock(return_value=5)
        
        mock_ask = Mock()
        type(mock_ask).agent_id = PropertyMock(return_value=2)
        type(mock_ask.market_action).price = PropertyMock(return_value=110)
        type(mock_ask.market_action).quantity = PropertyMock(return_value=5)
        
        self.mock_order_book.bids = [mock_bid]
        self.mock_order_book.asks = [mock_ask]
        
        # Mock trade history
        mock_trade = Mock()
        type(mock_trade).trade_id = PropertyMock(return_value=1)
        type(mock_trade.bid).agent_id = PropertyMock(return_value=1)
        type(mock_trade.ask).agent_id = PropertyMock(return_value=2)
        type(mock_trade).price = PropertyMock(return_value=100)
        type(mock_trade).quantity = PropertyMock(return_value=5)
        type(mock_trade).round = PropertyMock(return_value=1)
        type(mock_trade).total_surplus = PropertyMock(return_value=10)  # Add this line
        
        self.mock_trade_history = [mock_trade]
        self.mock_successful_trades = [mock_trade]
        self.mock_auction.successful_trades = [mock_trade]

    def test_generate_market_state_table(self):
        result = generate_market_state_table(self.mock_environment)
        self.assertIsNotNone(result)
        self.assertTrue(len(result.children) > 0)

    def test_generate_order_book_table(self):
        result = generate_order_book_table(self.mock_order_book)
        self.assertIsNotNone(result)
        self.assertTrue(len(result.children) > 0)

    def test_generate_trade_history_table(self):
        result = generate_trade_history_table(self.mock_trade_history)
        self.assertIsNotNone(result)
        self.assertTrue(len(result.children) > 0)

    def test_generate_supply_demand_chart(self):
        result = generate_supply_demand_chart(self.mock_environment)
        self.assertIsNotNone(result)
        self.assertTrue(len(result.data) > 0)

    def test_generate_price_vs_trade_chart(self):
        result = generate_price_vs_trade_chart(self.mock_successful_trades)
        self.assertIsNotNone(result)
        self.assertTrue(len(result.data) > 0)

    def test_generate_cumulative_quantity_surplus_chart(self):
        result = generate_cumulative_quantity_surplus_chart(self.mock_auction)
        self.assertIsNotNone(result)
        self.assertTrue(len(result.data) > 0)

if __name__ == '__main__':
    unittest.main()