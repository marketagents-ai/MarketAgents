import unittest
from unittest.mock import patch, MagicMock
from ziagents import ZIAgent, Allocation, Trade, MarketAction, Bid, Ask, create_zi_agent

class TestZIAgent(unittest.TestCase):
    def setUp(self):
        self.buyer_agent = create_zi_agent(agent_id=1, is_buyer=True, num_units=10, base_value=100, initial_cash=1000, initial_goods=0)
        self.seller_agent = create_zi_agent(agent_id=2, is_buyer=False, num_units=10, base_value=80, initial_cash=0, initial_goods=10)

    def test_generate_bid(self):
        bid = self.buyer_agent.generate_bid()
        self.assertIsInstance(bid, Bid)
        self.assertEqual(bid.agent_id, 1)
        self.assertLessEqual(bid.market_action.price, self.buyer_agent.base_value)

    def test_generate_ask(self):
        ask = self.seller_agent.generate_ask()
        self.assertIsInstance(ask, Ask)
        self.assertEqual(ask.agent_id, 2)
        self.assertGreaterEqual(ask.market_action.price, self.seller_agent.base_value)

    def test_finalize_bid(self):
        initial_cash = self.buyer_agent.allocation.cash
        initial_goods = self.buyer_agent.allocation.goods
        trade = Trade(trade_id=1, bid=Bid(agent_id=1, market_action=MarketAction(price=90), base_value=100),
                      ask=Ask(agent_id=2, market_action=MarketAction(price=85), base_cost=80),
                      price=87.5, round=1)
        self.buyer_agent.finalize_bid(trade)
        self.assertEqual(self.buyer_agent.allocation.cash, initial_cash - 87.5)
        self.assertEqual(self.buyer_agent.allocation.goods, initial_goods + 1)

    def test_finalize_ask(self):
        initial_cash = self.seller_agent.allocation.cash
        initial_goods = self.seller_agent.allocation.goods
        trade = Trade(trade_id=1, bid=Bid(agent_id=1, market_action=MarketAction(price=90), base_value=100),
                      ask=Ask(agent_id=2, market_action=MarketAction(price=85), base_cost=80),
                      price=87.5, round=1)
        self.seller_agent.finalize_ask(trade)
        self.assertEqual(self.seller_agent.allocation.cash, initial_cash + 87.5)
        self.assertEqual(self.seller_agent.allocation.goods, initial_goods - 1)

    def test_individual_surplus(self):
        # Simulate a trade
        trade = Trade(trade_id=1, bid=Bid(agent_id=1, market_action=MarketAction(price=90), base_value=100),
                      ask=Ask(agent_id=2, market_action=MarketAction(price=85), base_cost=80),
                      price=87.5, round=1)
        self.buyer_agent.finalize_bid(trade)
        self.seller_agent.finalize_ask(trade)

        buyer_surplus = self.buyer_agent.individual_surplus
        seller_surplus = self.seller_agent.individual_surplus

        self.assertGreater(buyer_surplus, 0)
        self.assertGreater(seller_surplus, 0)

class TestSimulation(unittest.TestCase):
    @patch('ziagents.ZIAgent.generate_bid')
    @patch('ziagents.ZIAgent.generate_ask')
    def test_simulation(self, mock_generate_ask, mock_generate_bid):
        buyer = create_zi_agent(agent_id=1, is_buyer=True, num_units=10, base_value=100, initial_cash=1000, initial_goods=0)
        seller = create_zi_agent(agent_id=2, is_buyer=False, num_units=10, base_value=80, initial_cash=0, initial_goods=10)

        # Mock the generate_bid and generate_ask methods to return predetermined values
        mock_generate_bid.side_effect = [
            Bid(agent_id=1, market_action=MarketAction(price=99.16), base_value=100),
            Bid(agent_id=1, market_action=MarketAction(price=83.77), base_value=100),
            None,  # Simulate no bid for subsequent rounds
        ]
        mock_generate_ask.side_effect = [
            Ask(agent_id=2, market_action=MarketAction(price=93.80), base_cost=80),
            Ask(agent_id=2, market_action=MarketAction(price=91.73), base_cost=80),
            None,  # Simulate no ask for subsequent rounds
        ]

        trade_history = []
        for round in range(1, 3):  # Simulate 2 rounds
            bid = buyer.generate_bid()
            ask = seller.generate_ask()

            if bid and ask:
                if bid.market_action.price >= ask.market_action.price:
                    trade_price = (bid.market_action.price + ask.market_action.price) / 2
                    trade = Trade(trade_id=len(trade_history) + 1, bid=bid, ask=ask, price=trade_price, round=round)
                    buyer.finalize_bid(trade)
                    seller.finalize_ask(trade)
                    trade_history.append(trade)

        # Check the results
        self.assertEqual(len(trade_history), 1)
        self.assertAlmostEqual(buyer.allocation.cash, 903.52, places=2)
        self.assertEqual(buyer.allocation.goods, 1)
        self.assertAlmostEqual(seller.allocation.cash, 96.48, places=2)
        self.assertEqual(seller.allocation.goods, 9)

        # Calculate expected surpluses based on the actual implementation
        expected_buyer_surplus = buyer.preference_schedule.get_value(1) - (buyer.allocation.initial_cash - buyer.allocation.cash)
        expected_seller_surplus = seller.allocation.cash - seller.allocation.initial_cash - seller.preference_schedule.get_value(1)

        self.assertAlmostEqual(buyer.individual_surplus, expected_buyer_surplus, places=2)
        self.assertAlmostEqual(seller.individual_surplus, expected_seller_surplus, places=2)

        # Print debug information
        print(f"Buyer's cash: {buyer.allocation.cash}")
        print(f"Buyer's goods: {buyer.allocation.goods}")
        print(f"Buyer's base value: {buyer.base_value}")
        print(f"Buyer's individual surplus: {buyer.individual_surplus}")
        print(f"Expected buyer surplus: {expected_buyer_surplus}")
        print(f"Seller's cash: {seller.allocation.cash}")
        print(f"Seller's goods: {seller.allocation.goods}")
        print(f"Seller's base value: {seller.base_value}")
        print(f"Seller's individual surplus: {seller.individual_surplus}")
        print(f"Expected seller surplus: {expected_seller_surplus}")

if __name__ == '__main__':
    unittest.main()