import json
import unittest
import logging
from datetime import datetime
from base_agent.agent import Agent
from base_agent.aiutilities import AIUtilities, LLMConfig, LLMOutput
from base_agent.schemas import ReActSchema, Action

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestAgent(unittest.TestCase):
    def test_agent_initialization(self):
        agent = Agent(role="test_role", llm_config=LLMConfig(client="openai", model="gpt-3.5-turbo"))
        logger.debug(f"Agent role: {agent.role}")
        logger.debug(f"Agent ID: {agent.id}")
        logger.debug(f"Agent LLM config: {agent.llm_config}")
        logger.debug(f"Agent max retries: {agent.max_retries}")
        self.assertEqual(agent.role, "test_role")
        self.assertIsInstance(agent.id, str)
        self.assertIsInstance(agent.llm_config, LLMConfig)
        self.assertEqual(agent.max_retries, 2)

    def test_agent_with_custom_config(self):
        agent = Agent(
            role="custom_role",
            system="Custom system prompt",
            max_retries=5,
            llm_config=LLMConfig(client="openai", model="gpt-4")
        )
        logger.debug(f"Agent role: {agent.role}")
        logger.debug(f"Agent system prompt: {agent.system}")
        logger.debug(f"Agent max retries: {agent.max_retries}")
        logger.debug(f"Agent LLM model: {agent.llm_config.model}")
        self.assertEqual(agent.role, "custom_role")
        self.assertEqual(agent.system, "Custom system prompt")
        self.assertEqual(agent.max_retries, 5)
        self.assertEqual(agent.llm_config.model, "gpt-4")

    def test_agent_execute(self):
        agent = Agent(role="default", system="Test system", llm_config=LLMConfig(client="openai", model="gpt-4"))
        result = agent.execute()
        logger.debug(f"Agent execution result: {result}")
        self.assertIsInstance(result, str)
        # Here you might want to assert against an expected outcome if you have a way to predict the response
        # self.assertEqual(result, "Expected response")

    def test_agent_with_task(self):
        agent = Agent(
            role="default",
            system="You are a task execution agent. Respond with a brief confirmation.",
            task="Confirm that you received this task.",
            llm_config=LLMConfig(client="openai", model="gpt-3.5-turbo")
        )

        result = agent.execute()
        logger.debug(f"Agent task execution result: {result}")
        self.assertIsInstance(result, str)
        # Here you might want to assert against an expected outcome if you have a way to predict the response
        # self.assertEqual(result, "Expected confirmation response")

    def test_load_output_schema(self):
        agent = Agent(role="test_role", output_format="TestSchema", llm_config=LLMConfig(client="openai", model="gpt-3.5-turbo"))
        logger.debug(f"Agent output format: {agent.output_format}")
        self.assertIsInstance(agent.output_format, dict)
        self.assertEqual(agent.output_format.get("title"), "TestSchema")

    def test_react_agent_with_real_llm_call(self):
        agent = Agent(
            role="react_agent",
            system="You are a ReAct agent. Your task is to solve problems step-by-step, using the ReAct framework.",
            output_format="ReActSchema",
            llm_config=LLMConfig(client="openai", model="gpt-4o-mini", response_format="json_object")
        )

        task = "Calculate the sum of the first 10 prime numbers. Show your reasoning step by step."

        # This will make the actual API call
        result = agent.execute(task)
        logger.debug(f"ReAct agent execution result: {result}")

        # Parse the result
        react_output = ReActSchema(**result)

        logger.debug(f"ReAct thought: {react_output.thought}")
        logger.debug(f"ReAct action: {react_output.action}")
        logger.debug(f"ReAct observation: {react_output.observation}")
        logger.debug(f"ReAct final answer: {react_output.final_answer}")

        # Assert that all fields are present and non-empty
        self.assertIsNotNone(react_output.thought)
        self.assertNotEqual(react_output.thought, "")
        
        self.assertIsNotNone(react_output.action)
        self.assertIsInstance(react_output.action, Action)
        self.assertIsNotNone(react_output.action.name)
        self.assertNotEqual(react_output.action.name, "")
        self.assertIsNotNone(react_output.action.input)
        self.assertNotEqual(react_output.action.input, "")
        
        self.assertIsNotNone(react_output.observation)
        self.assertNotEqual(react_output.observation, "")
        
        self.assertIsNotNone(react_output.final_answer)
        self.assertNotEqual(react_output.final_answer, "")

        # Additional assertions to check the content
        self.assertIn("calculate", react_output.action.name.lower())
        self.assertTrue(any(char.isdigit() for char in react_output.final_answer))

    def test_agent_interactions_logging(self):
        agent = Agent(role="test_role", llm_config=LLMConfig(client="openai", model="gpt-3.5-turbo"))
        agent.execute("Test task")

        logger.debug(f"Number of agent interactions: {len(agent.interactions)}")
        if agent.interactions:
            logger.debug(f"First interaction: {agent.interactions[0]}")

        self.assertEqual(len(agent.interactions), 1)
        interaction = agent.interactions[0]
        self.assertIn("id", interaction)
        self.assertIn("name", interaction)
        self.assertIn("prompt_context", interaction)
        self.assertIn("response", interaction)
        self.assertIn("timestamp", interaction)

if __name__ == '__main__':
    unittest.main()
