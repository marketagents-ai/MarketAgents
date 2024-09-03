import json
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
from base_agent.agent import Agent
from base_agent.aiutilities import AIUtilities
from base_agent.schema import ReActSchema

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
        agent = Agent(role="default", system="Test system", llm_config={"client": "openai", "model": "gpt-4o-mini"})
        result = agent.execute()
        print(result)

        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")

    def test_agent_with_task(self):
        agent = Agent(
            role="default",
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
    
        # Convert both to dictionaries
        expected_dict = json.loads(expected_schema)
    
        self.assertEqual(schema, expected_dict)

    def test_react_agent_with_real_llm_call(self):
        agent = Agent(
            role="react_agent",
            system="You are a ReAct agent. Your task is to solve problems step-by-step, using the ReAct framework.",
            output_format="ReActSchema",
            llm_config={"client": "openai", "model": "gpt-3.5-turbo", "response_format": "json_object"}
        )

        task = "Calculate the sum of the first 10 prime numbers. Show your reasoning step by step."

        result = agent.execute(task)

        # Parse the result
        react_output = ReActSchema.parse_raw(result)

        # Assert that all fields are present and non-empty
        self.assertIsNotNone(react_output.thought)
        self.assertNotEqual(react_output.thought, "")
        
        self.assertIsNotNone(react_output.action)
        self.assertIsNotNone(react_output.action.name)
        self.assertNotEqual(react_output.action.name, "")
        self.assertIsNotNone(react_output.action.input)
        self.assertNotEqual(react_output.action.input, "")
        
        self.assertIsNotNone(react_output.observation)
        self.assertNotEqual(react_output.observation, "")
        
        self.assertIsNotNone(react_output.final_answer)
        self.assertNotEqual(react_output.final_answer, "")

        # Print the result for inspection
        print(f"ReAct Agent response:\n{json.dumps(react_output.dict(), indent=2)}")

        # Additional assertions to check the content
        self.assertIn("prime", react_output.thought.lower())
        self.assertIn("calculate", react_output.action.name.lower())
        self.assertTrue(any(char.isdigit() for char in react_output.final_answer))

if __name__ == '__main__':
    unittest.main()