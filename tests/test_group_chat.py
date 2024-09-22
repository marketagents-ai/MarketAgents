# test_group_chat.py

import unittest
from market_agents.environments.mechanisms.group_chat import GroupChat, GroupChatAction, GroupChatGlobalAction, GroupChatMessage

class TestGroupChatMechanism(unittest.TestCase):
    def setUp(self):
        self.mechanism = GroupChat(
            max_rounds=2,
            current_topic="Initial Topic",
            speaker_order=["0", "1", "2", "3", "4"]
        )

    def test_step_with_correct_action(self):
        actions = {
            "0": GroupChatAction(agent_id="0", action=GroupChatMessage(content="Hello", message_type="group_message")),
            "1": GroupChatAction(agent_id="1", action=GroupChatMessage(content="Hi there", message_type="group_message")),
        }
        global_action = GroupChatGlobalAction(actions=actions)
        step_result = self.mechanism.step(global_action)
        self.assertEqual(step_result.done, False)
        self.assertEqual(len(step_result.global_observation.all_messages), 2)

    def test_step_with_incorrect_action(self):
        # Passing a single GroupChatAction instead of GroupChatGlobalAction
        single_action = GroupChatAction(agent_id="0", action=GroupChatMessage(content="Hello", message_type="group_message"))
        with self.assertRaises(TypeError):
            self.mechanism.step(single_action)

    def test_step_with_dict_action(self):
        # Passing a dict instead of GroupChatGlobalAction
        actions = {
            "actions": {
                "0": {"agent_id": "0", "action": {"content": "Hello", "message_type": "group_message"}},
                "1": {"agent_id": "1", "action": {"content": "Hi there", "message_type": "group_message"}},
            }
        }
        step_result = self.mechanism.step(actions)
        self.assertEqual(step_result.done, False)
        self.assertEqual(len(step_result.global_observation.all_messages), 2)

if __name__ == '__main__':
    unittest.main()
