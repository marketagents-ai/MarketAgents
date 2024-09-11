import unittest
from unittest.mock import Mock, patch
from typing import List
<<<<<<< HEAD
from llm_agents.environments.auction.auction import DoubleAuction, Environment, OrderBook, Trade
=======
from llm_agents.environments.auction import DoubleAuction, Environment, OrderBook, Trade
>>>>>>> main
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