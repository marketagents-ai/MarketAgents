import json
import unittest
import logging
from datetime import datetime
import asyncio
from market_agents.agents.base_agent.agent import Agent
from market_agents.inference.parallel_inference import ParallelAIUtilities
from market_agents.inference.message_models import StructuredTool, LLMConfig, LLMPromptContext, LLMOutput
from market_agents.agents.base_agent.schemas import ReActSchema, Action, ChainOfThoughtSchema

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.get_event_loop()

    def test_agent_initialization(self):
        agent = Agent(role="test_role", llm_config=LLMConfig(client="openai", model="gpt-4o-mini"))
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
            llm_config=LLMConfig(client="openai", model="gpt-4o-mini")
        )
        logger.debug(f"Agent role: {agent.role}")
        logger.debug(f"Agent system prompt: {agent.system}")
        logger.debug(f"Agent max retries: {agent.max_retries}")
        logger.debug(f"Agent LLM model: {agent.llm_config.model}")
        self.assertEqual(agent.role, "custom_role")
        self.assertEqual(agent.system, "Custom system prompt")
        self.assertEqual(agent.max_retries, 5)
        self.assertEqual(agent.llm_config.model, "gpt-4o-mini")

    def test_agent_execute(self):
        agent = Agent(role="default", system="Test system", llm_config=LLMConfig(client="openai", model="gpt-4o-mini"))
        task = "Provide a brief summary of the benefits of exercise."
        result = self.loop.run_until_complete(agent.execute(task=task, output_format=ChainOfThoughtSchema))
        logger.debug(f"Agent execution result: {result}")
        self.assertIsInstance(result, dict)

    def test_agent_with_task(self):
        agent = Agent(
            role="default",
            system="You are a task execution agent. Respond with a brief confirmation.",
            task="Confirm that you received this task.",
            llm_config=LLMConfig(client="openai", model="gpt-4o-mini")
        )

        result = self.loop.run_until_complete(agent.execute(output_format=Action))
        logger.debug(f"Agent task execution result: {result}")
        self.assertIsInstance(result, dict)

    def test_load_output_schema(self):
        agent = Agent(role="test_role", output_format="TestSchema", llm_config=LLMConfig(client="openai", model="gpt-4o-mini"))
        logger.debug(f"Agent output format: {agent.output_format}")
        self.assertEqual(agent.output_format, "TestSchema")
        loaded_schema = agent._load_output_schema(agent.output_format)
        self.assertIsInstance(loaded_schema, dict)
        self.assertEqual(loaded_schema.get("title"), "TestSchema")

    def test_react_agent_with_real_llm_call(self):
        agent = Agent(
            role="react_agent",
            system="You are a ReAct agent. Your task is to solve problems step-by-step, using the ReAct framework.",
            llm_config=LLMConfig(client="openai", model="gpt-4o-mini", response_format="json_object")
        )

        task = "Calculate the sum of the first 10 prime numbers. Show your reasoning step by step. For action please write python code to execute this calculation. Provide the expected output of code execution in final answer"

        result = self.loop.run_until_complete(agent.execute(task, ReActSchema))
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

        self.assertIsInstance(react_output.action, Action)
        self.assertIsNotNone(react_output.action.name)
        self.assertNotEqual(react_output.action.name, "")
        self.assertIsNotNone(react_output.action.input)
        self.assertNotEqual(react_output.action.input, "")
        
        #self.assertIsNotNone(react_output.observation)
        self.assertNotEqual(react_output.observation, "")
        
        self.assertIsNotNone(react_output.final_answer)
        self.assertNotEqual(react_output.final_answer, "")

        # Additional assertions to check the content
        self.assertIn("calculate", react_output.action.name.lower())
        self.assertTrue(any(char.isdigit() for char in react_output.final_answer))

    def test_agent_interactions_logging(self):
        agent = Agent(role="test_role", llm_config=LLMConfig(client="openai", model="gpt-4o-mini"))
        self.loop.run_until_complete(agent.execute("Test task"))

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