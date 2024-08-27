import unittest
from unittest.mock import patch
from datetime import datetime
from agents import Agent
from aiutilities import AIUtilities

class TestAgent(unittest.TestCase):
    def test_agent_initialization(self):
        agent = Agent(role="test_role")
        self.assertEqual(agent.role, "test_role")
        self.assertEqual(agent.max_iter, 2)
        self.assertIsInstance(agent.id, str)

    def test_agent_with_custom_config(self):
        agent = Agent(
            role="custom_role",
            system="Custom system prompt",
            max_iter=5,
            llm_config={"model": "gpt-4"}
        )
        self.assertEqual(agent.role, "custom_role")
        self.assertEqual(agent.system, "Custom system prompt")
        self.assertEqual(agent.max_iter, 5)
        self.assertEqual(agent.llm_config, {"model": "gpt-4"})

    def test_agent_execute(self):
        agent = Agent(role="test_role", system="Test system", llm_config={"client": "openai", "model": "gpt-4o-mini"})
        result = agent.execute()
        print(result)

        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")

    def test_agent_with_task(self):
        agent = Agent(
            role="task_executor",
            system="You are a task execution agent. Respond with a brief confirmation.",
            task="Confirm that you received this task.",
            llm_config={"client": "openai", "model": "gpt-3.5-turbo"}
        )

        result = agent.execute()

        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")
        
        # Print the result for inspection
        print(f"Agent response: {result}")

    def test_load_output_schema(self):
        agent = Agent(role="test_role", output_format="TestSchema")
        schema = agent.output_format

        expected_schema = '{"title": "TestSchema", "type": "object", "properties": {"test": {"title": "Test", "default": "schema", "type": "string"}}}'
        self.assertEqual(schema, expected_schema)
    
if __name__ == '__main__':
    unittest.main()