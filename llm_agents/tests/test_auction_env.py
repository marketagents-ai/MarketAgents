import unittest
from unittest.mock import Mock
from environments.auction.auction_environment import AuctionEnvironment, InitialDemandCurve, InitialSupplyCurve, CurvePoint
from environments.auction.auction import DoubleAuction
from protocols.acl_message import ACLMessage, AgentID, Performative
from protocols.protocol import Protocol

class TestAuctionEnvironment(unittest.TestCase):
    def setUp(self):
        initial_demand_curve = InitialDemandCurve(points=[
            CurvePoint(quantity=0, price=100),
            CurvePoint(quantity=10, price=50),
            CurvePoint(quantity=20, price=0)
        ])
        initial_supply_curve = InitialSupplyCurve(points=[
            CurvePoint(quantity=0, price=0),
            CurvePoint(quantity=10, price=50),
            CurvePoint(quantity=20, price=100)
        ])
        self.auction_environment = AuctionEnvironment(
            max_steps=5,
            protocol=ACLMessage,
            auction_type='double',
            name='Test Auction',
            address='test_auction_address',
            agents=[],
            initial_demand_curve=initial_demand_curve,
            initial_supply_curve=initial_supply_curve
        )

    def test_auction_environment_initialization(self):
        self.assertEqual(self.auction_environment.max_steps, 5)
        self.assertEqual(self.auction_environment.current_step, 0)
        self.assertEqual(self.auction_environment.auction_type, 'double')
        self.assertIsInstance(self.auction_environment.auction, DoubleAuction)

    def test_auction_environment_get_global_state(self):
        global_state = self.auction_environment.get_global_state()
        self.assertIn('current_step', global_state)
        self.assertIn('max_steps', global_state)
        self.assertIn('current_demand_curve', global_state)
        self.assertIn('current_supply_curve', global_state)
        self.assertIn('remaining_trade_opportunities', global_state)
        self.assertIn('remaining_surplus', global_state)
        self.assertIn('total_utility', global_state)
        self.assertIn('ce_price', global_state)
        self.assertIn('ce_quantity', global_state)
        self.assertIn('efficiency', global_state)

    def test_auction_environment_step(self):
        agent_actions = {
            '1': {'type': 'bid', 'price': 100, 'quantity': 1},
            '2': {'type': 'ask', 'price': 90, 'quantity': 1}
        }
        new_state = self.auction_environment.step(agent_actions)
        
        self.assertIsInstance(new_state, dict)
        self.assertIn('observations', new_state)
        self.assertIn('market_info', new_state)
        self.assertIn('done', new_state)
        self.assertIn('current_step', new_state)
        
        self.assertEqual(new_state['current_step'], 1)
        self.assertFalse(new_state['done'])

    def test_auction_environment_reset(self):
        agent_actions = {
            '1': {'type': 'bid', 'price': 60, 'quantity': 1},
            '2': {'type': 'ask', 'price': 40, 'quantity': 1}
        }
        self.auction_environment.step(agent_actions)
        reset_state = self.auction_environment.reset()
        self.assertEqual(reset_state['current_step'], 0)
        self.assertEqual(len(reset_state['current_demand_curve']['points']), 0)
        self.assertEqual(len(reset_state['current_supply_curve']['points']), 0)

    def test_auction_environment_get_observation(self):
        observation = self.auction_environment.get_observation('1')
        self.assertIsInstance(observation, ACLMessage)
        self.assertEqual(observation.performative, "inform")
        self.assertEqual(observation.sender, AgentID(name="market"))
        self.assertEqual(observation.content['agent_id'], 1)

    def test_auction_environment_parse_action(self):
        action_message = ACLMessage(
            performative=Performative.PROPOSE,
            sender=AgentID(name="1"),
            receivers=[AgentID(name="market")],
            content={'type': 'bid', 'price': 60, 'quantity': 1}
        )
        
        parsed_action = self.auction_environment.parse_action(action_message)
        self.assertIsInstance(parsed_action, dict)
        self.assertIn('type', parsed_action)
        self.assertIn('price', parsed_action)
        self.assertIn('quantity', parsed_action)

        # Test with REQUEST performative
        action_message.performative = Performative.REQUEST
        parsed_action = self.auction_environment.parse_action(action_message)
        self.assertIsInstance(parsed_action, dict)
        self.assertIn('type', parsed_action)
        self.assertIn('price', parsed_action)
        self.assertIn('quantity', parsed_action)

        # Test with invalid performative
        action_message.performative = Performative.INFORM
        parsed_action = self.auction_environment.parse_action(action_message)
        self.assertIsInstance(parsed_action, dict)
        self.assertEqual(parsed_action['type'], 'hold')

        # Test with invalid content
        action_message.performative = Performative.PROPOSE
        action_message.content = "invalid content"
        parsed_action = self.auction_environment.parse_action(action_message)
        self.assertIsInstance(parsed_action, dict)
        self.assertEqual(parsed_action['type'], 'hold')

        # Test with invalid action type
        action_message.content = {'type': 'invalid', 'price': 60, 'quantity': 1}
        parsed_action = self.auction_environment.parse_action(action_message)
        self.assertIsInstance(parsed_action, dict)
        self.assertEqual(parsed_action['type'], 'hold')

    def test_auction_environment_get_action_space(self):
        action_space = self.auction_environment.get_action_space()
        self.assertEqual(action_space['type'], 'continuous')
        self.assertEqual(action_space['shape'], (2,))
        self.assertEqual(action_space['low'], [0, 0])
        self.assertEqual(action_space['high'], [float('inf'), float('inf')])

    def test_auction_environment_get_action_schema(self):
        action_schema = self.auction_environment.get_action_schema()
        self.assertIn('properties', action_schema)
        self.assertIn('thought', action_schema['properties'])
        self.assertIn('action', action_schema['properties'])
        self.assertIn('bid', action_schema['properties'])
        self.assertEqual(action_schema['properties']['action']['enum'], ['bid', 'ask', 'hold'])

if __name__ == '__main__':
    unittest.main()
