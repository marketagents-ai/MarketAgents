import unittest
from unittest.mock import Mock, patch
from typing import List
from auction import DoubleAuction, run_market_simulation, Environment, Order, Trade

class TestDoubleAuction(unittest.TestCase):
    def setUp(self):
        self.environment = Mock(spec=Environment)
        self.auction = DoubleAuction(self.environment, max_rounds=10)

    def test_init(self):
        self.assertEqual(self.auction.max_rounds, 10)
        self.assertEqual(self.auction.current_round, 0)
        self.assertEqual(self.auction.successful_trades, [])
        self.assertEqual(self.auction.total_surplus_extracted, 0.0)
        self.assertEqual(self.auction.average_prices, [])
        self.assertEqual(self.auction.order_book, [])
        self.assertEqual(self.auction.trade_history, [])
        self.assertEqual(self.auction.trade_counter, 0)

    def test_match_orders(self):
        bids = [Order(agent_id=1, price=10, quantity=1, base_value=12, is_buy=True),
                Order(agent_id=2, price=9, quantity=1, base_value=11, is_buy=True)]
        asks = [Order(agent_id=3, price=8, quantity=1, base_cost=7, is_buy=False),
                Order(agent_id=4, price=11, quantity=1, base_cost=10, is_buy=False)]
        
        trades = self.auction.match_orders(bids, asks, round_num=1)
        
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0].buyer_id, 1)
        self.assertEqual(trades[0].seller_id, 3)
        self.assertEqual(trades[0].price, 9)
        self.assertEqual(trades[0].quantity, 1)

    @patch('auction.DoubleAuction.execute_trades')
    def test_run_auction(self, mock_execute_trades):
        self.environment.buyers = [Mock(), Mock()]
        self.environment.sellers = [Mock(), Mock()]
        
        for buyer in self.environment.buyers:
            buyer.allocation.goods = 0
            buyer.preference_schedule.values = {1: 10}
            buyer.generate_bid.return_value = Order(agent_id=1, price=10, quantity=1, base_value=12, is_buy=True)
        
        for seller in self.environment.sellers:
            seller.allocation.goods = 1
            seller.generate_bid.return_value = Order(agent_id=2, price=8, quantity=1, base_cost=7, is_buy=False)
        
        # Mock the calculate_equilibrium method
        self.environment.calculate_equilibrium.return_value = (9, 2, 10, 10, 20)
        
        self.auction.run_auction()
        
        self.assertEqual(self.auction.current_round, 11)
        self.assertTrue(mock_execute_trades.called)

    def test_execute_trades(self):
        buyer = Mock()
        seller = Mock()
        self.environment.get_agent.side_effect = [buyer, seller]
        
        trade = Trade(trade_id=1, buyer_id=1, seller_id=2, quantity=1, price=10, buyer_value=12, seller_cost=8, round=1)
        
        self.auction.execute_trades([trade])
        
        self.assertEqual(len(self.auction.successful_trades), 1)
        self.assertEqual(self.auction.total_surplus_extracted, 4.0)  # Changed from 6 to 4.0
        self.assertEqual(self.auction.average_prices, [10])
        self.assertTrue(buyer.finalize_trade.called)
        self.assertTrue(seller.finalize_trade.called)

class TestRunMarketSimulation(unittest.TestCase):
    @patch('auction.generate_agents')
    @patch('auction.Environment')
    @patch('auction.DoubleAuction')
    @patch('auction.analyze_and_plot_auction_results')
    def test_run_market_simulation(self, mock_analyze, mock_auction, mock_env, mock_generate_agents):
        mock_agent = Mock()
        mock_agent.id = 1
        mock_agent.preference_schedule = Mock()
        mock_agent.preference_schedule.is_buyer = True
        mock_generate_agents.return_value = [mock_agent]
        mock_env.return_value.agents = mock_generate_agents.return_value
        mock_env.return_value.get_agent_utility.return_value = 100.0
        
        run_market_simulation(num_buyers=5, num_sellers=5, num_units=2, buyer_base_value=100, seller_base_value=90, spread=0.5, max_rounds=10)
        
        self.assertTrue(mock_generate_agents.called)
        self.assertTrue(mock_env.called)
        self.assertTrue(mock_auction.called)
        self.assertTrue(mock_analyze.called)

if __name__ == '__main__':
    unittest.main()