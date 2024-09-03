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
import unittest
from unittest.mock import Mock, patch
from typing import List
from auction import DoubleAuction, Environment, OrderBook, Trade
from ziagents import Order, MarketAction, Bid, Ask, MarketInfo

class TestDoubleAuction(unittest.TestCase):
    def setUp(self):
        self.environment = Mock(spec=Environment)
        self.auction = DoubleAuction(environment=self.environment, max_rounds=10)

    def test_init(self):
        self.assertEqual(self.auction.max_rounds, 10)
        self.assertEqual(self.auction.current_round, 0)
        self.assertEqual(self.auction.successful_trades, [])
        self.assertEqual(self.auction.total_surplus_extracted, 0.0)
        self.assertEqual(self.auction.average_prices, [])
        self.assertIsInstance(self.auction.order_book, OrderBook)
        self.assertEqual(self.auction.trade_history, [])

    def test_execute_trades(self):
        buyer = Mock()
        seller = Mock()
        self.environment.get_agent.side_effect = [buyer, seller]
        
        bid = Bid(agent_id=1, market_action=MarketAction(agent_id=1, price=12, quantity=1), base_value=15)
        ask = Ask(agent_id=2, market_action=MarketAction(agent_id=2, price=8, quantity=1), base_cost=5)
        trade = Trade(trade_id=0, bid=bid, ask=ask, price=10, round=1)
        
        self.auction.execute_trades([trade])
        
        self.assertEqual(len(self.auction.successful_trades), 1)
        self.assertEqual(self.auction.total_surplus_extracted, 10.0)  # (15 - 5) = 10
        self.assertEqual(self.auction.average_prices, [10])
        self.assertTrue(buyer.finalize_bid.called)
        self.assertTrue(seller.finalize_ask.called)

    @patch('auction.DoubleAuction.execute_trades')
    def test_run_auction(self, mock_execute_trades):
        self.environment.buyers = [Mock(), Mock()]
        self.environment.sellers = [Mock(), Mock()]
        
        # Set concrete base values for buyers and sellers
        for i, buyer in enumerate(self.environment.buyers):
            buyer.base_value = 100 + i * 10
            buyer.generate_bid.return_value = Bid(agent_id=i, market_action=MarketAction(agent_id=i, price=90 + i * 5, quantity=1), base_value=buyer.base_value)
        
        for i, seller in enumerate(self.environment.sellers):
            seller.base_value = 80 - i * 10
            seller.generate_ask.return_value = Ask(agent_id=i+2, market_action=MarketAction(agent_id=i+2, price=85 - i * 5, quantity=1), base_cost=seller.base_value)
        
        # Mock the calculate_equilibrium method
        self.environment.calculate_equilibrium.return_value = (90, 2, 10, 10, 20)
        
        self.auction.run_auction()
        
        self.assertEqual(self.auction.current_round, 10)
        self.assertTrue(mock_execute_trades.called)

    def test_generate_bids(self):
        self.environment.buyers = [Mock(), Mock()]
        market_info = MarketInfo(last_trade_price=10, average_price=9, total_trades=5, current_round=1)
        
        for i, buyer in enumerate(self.environment.buyers):
            buyer.generate_bid.return_value = Bid(agent_id=i, market_action=MarketAction(agent_id=i, price=10, quantity=1), base_value=12)
        
        bids = self.auction.generate_bids(market_info)
        
        self.assertEqual(len(bids), 2)
        for bid in bids:
            self.assertIsInstance(bid, Bid)
            self.assertEqual(bid.market_action.price, 10)
            self.assertEqual(bid.market_action.quantity, 1)
            self.assertEqual(bid.base_value, 12)

    def test_generate_asks(self):
        self.environment.sellers = [Mock(), Mock()]
        market_info = MarketInfo(last_trade_price=10, average_price=9, total_trades=5, current_round=1)
        
        for i, seller in enumerate(self.environment.sellers):
            seller.generate_ask.return_value = Ask(agent_id=i+2, market_action=MarketAction(agent_id=i+2, price=8, quantity=1), base_cost=6)
        
        asks = self.auction.generate_asks(market_info)
        
        self.assertEqual(len(asks), 2)
        for ask in asks:
            self.assertIsInstance(ask, Ask)
            self.assertEqual(ask.market_action.price, 8)
            self.assertEqual(ask.market_action.quantity, 1)
            self.assertEqual(ask.base_cost, 6)

    def test_get_market_info(self):
        self.auction.average_prices = [9, 10, 11]
        self.auction.successful_trades = [Mock(), Mock(), Mock()]
        
        market_info = self.auction._get_market_info()
        
        self.assertEqual(market_info.last_trade_price, 11)
        self.assertEqual(market_info.average_price, 10)
        self.assertEqual(market_info.total_trades, 3)
        self.assertEqual(market_info.current_round, 0)

if __name__ == '__main__':
    unittest.main()