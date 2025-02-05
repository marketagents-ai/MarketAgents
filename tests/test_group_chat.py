import unittest
from market_agents.environments.mechanisms.group_chat import (
    GroupChat,
    GroupChatAction,
    GroupChatGlobalAction,
    GroupChatMessage
)


class TestGroupChatMechanism(unittest.TestCase):
    def setUp(self):
        """
        Create two GroupChat instances:
        1) Non-sequential mode (expects GroupChatGlobalAction).
        2) Sequential mode (expects GroupChatAction).
        """
        self.mechanism_non_sequential = GroupChat(
            max_rounds=2,
            # 'sequential' inherited from Mechanism, so specifying here for clarity
            sequential=False
        )
        self.mechanism_sequential = GroupChat(
            max_rounds=2,
            sequential=True
        )
        # Provide speaker_order and current_speaker_index for sequential mode
        self.mechanism_sequential.speaker_order = ["agent-1", "agent-2"]
        self.mechanism_sequential.current_speaker_index = 0

    def test_step_with_global_action_non_sequential(self):
        """
        In non-sequential mode, the mechanism expects a GroupChatGlobalAction.
        This test verifies that passing a valid GlobalAction with agent_id keys
        and corresponding action dicts works correctly.
        """
        actions_dict = {
            "agent-1": {
                "agent_id": "agent-1",
                "action": {"content": "Hello from agent-1"}
            },
            "agent-2": {
                "agent_id": "agent-2",
                "action": {"content": "Greetings from agent-2"}
            }
        }
        global_action = GroupChatGlobalAction(actions=actions_dict)
        step_result = self.mechanism_non_sequential.step(global_action)

        # Basic checks
        self.assertIsNotNone(step_result)
        self.assertFalse(step_result.done, "Should not be done after the first step.")
        self.assertIn("agent_rewards", step_result.info)
        self.assertEqual(step_result.info["agent_rewards"].get("agent-1"), 1.0)
        self.assertEqual(step_result.info["agent_rewards"].get("agent-2"), 1.0)

        self.assertEqual(len(self.mechanism_non_sequential.messages), 2)
        self.assertEqual(self.mechanism_non_sequential.messages[0].content, "Hello from agent-1")
        self.assertEqual(self.mechanism_non_sequential.messages[1].content, "Greetings from agent-2")

    def test_step_with_dict_for_global_action(self):
        """
        If someone passes a pure dictionary to step() in non-sequential mode,
        the mechanism attempts to parse it as a GroupChatGlobalAction.
        """
        actions_dict = {
            "actions": {
                "agent-10": {
                    "agent_id": "agent-10",
                    "action": {"content": "A random message for testing"}
                }
            }
        }
        step_result = self.mechanism_non_sequential.step(actions_dict)
        self.assertIsNotNone(step_result)
        self.assertFalse(step_result.done)
        self.assertEqual(len(self.mechanism_non_sequential.messages), 1)
        self.assertEqual(
            self.mechanism_non_sequential.messages[0].content,
            "A random message for testing"
        )

    def test_step_with_local_action_sequential(self):
        """
        In sequential mode, the mechanism expects a single GroupChatAction each time step.
        Because our speaker_order is ['agent-1', 'agent-2'], the first action must come
        from 'agent-1'. Then it auto-advances to the next speaker.
        """
        action_obj = GroupChatAction(
            agent_id="agent-1",
            action=GroupChatMessage(content="Hello from the first agent")
        )
        step_result = self.mechanism_sequential.step(action_obj)

        self.assertIsNotNone(step_result)
        # Confirm speaker_order advanced to 'agent-2'
        self.assertEqual(
            self.mechanism_sequential.speaker_order[self.mechanism_sequential.current_speaker_index],
            "agent-2"
        )
        self.assertIn("agent_rewards", step_result.info)
        self.assertEqual(len(self.mechanism_sequential.messages), 1)
        self.assertEqual(self.mechanism_sequential.messages[0].content, "Hello from the first agent")

    def test_step_with_incorrect_action_non_seq(self):
        """
        If we pass a GroupChatAction directly in non-sequential mode,
        it will raise an error because the code explicitly expects
        GroupChatGlobalAction in non-sequential mode.
        """
        invalid_action = GroupChatAction(
            agent_id="agent-oops",
            action=GroupChatMessage(content="I'm incorrectly formatted!")
        )
        with self.assertRaises(TypeError) as context:
            self.mechanism_non_sequential.step(invalid_action)
        self.assertIn("Expected GroupChatGlobalAction", str(context.exception))

    def test_step_with_incorrect_agent_turn_sequential(self):
        """
        In sequential mode, if we pass an action from the wrong speaker
        (agent-2 when it's still agent-1's turn), it should raise an error.
        """
        invalid_action = GroupChatAction(
            agent_id="agent-2",
            action=GroupChatMessage(content="I'm out of turn!")
        )
        with self.assertRaises(ValueError) as context:
            self.mechanism_sequential.step(invalid_action)
        self.assertIn("It's not agent-2's turn", str(context.exception))


if __name__ == "__main__":
    unittest.main()