import json
import unittest
import logging
from datetime import datetime
import asyncio
from unittest.mock import patch, MagicMock

from market_agents.agents.base_agent.agent import Agent
from market_agents.inference.parallel_inference import ParallelAIUtilities
from market_agents.inference.message_models import StructuredTool, LLMConfig, LLMPromptContext, LLMOutput
from market_agents.agents.base_agent.schemas import ReActSchema, Action, ChainOfThoughtSchema

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MOCK_RESPONSES = {
    "chain_of_thought": {
        "thought": "Here's my analytical process",
        "reasoning": "Breaking this down step by step...",
        "conclusion": "Based on the analysis, we can conclude..."
    },
    
    "react_basic": {
        "thought": "I need to solve this problem step by step",
        "action": {
            "name": "calculate",
            "input": "2 + 2"
        },
        "observation": "The result is 4",
        "final_answer": "After calculation, the answer is 4"
    },
    
    "react_complex": {
        "thought": "To find the sum of prime numbers, I need to first generate them",
        "action": {
            "name": "calculate_primes",
            "input": """
def get_primes(n):
    primes = []
    num = 2
    while len(primes) < n:
        if all(num % prime != 0 for prime in primes):
            primes.append(num)
        num += 1
    return primes

primes = get_primes(10)
sum_primes = sum(primes)
print(f"First 10 primes: {primes}")
print(f"Sum: {sum_primes}")
            """
        },
        "observation": "First 10 primes: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]\nSum: 129",
        "final_answer": "The sum of the first 10 prime numbers is 129"
    },
    
    "action_response": {
        "action": {
            "name": "test_action",
            "input": "test input"
        }
    },
    
    "error_response": {
        "error": "Invalid input parameters",
        "error_code": "E1001",
        "suggestions": ["Check input format", "Verify parameters", "Try again"]
    }
}

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.get_event_loop()
        self.mock_responses = MOCK_RESPONSES

    @patch('market_agents.agents.base_agent.agent.ParallelAIUtilities')
    def test_agent_initialization(self, mock_ai):
        agent = Agent(role="test_role", llm_config=LLMConfig(client="mock", model="mock-model"))
        logger.debug(f"Agent role: {agent.role}")
        logger.debug(f"Agent ID: {agent.id}")
        logger.debug(f"Agent LLM config: {agent.llm_config}")
        logger.debug(f"Agent max retries: {agent.max_retries}")
        
        self.assertEqual(agent.role, "test_role")
        self.assertIsInstance(agent.id, str)
        self.assertIsInstance(agent.llm_config, LLMConfig)
        self.assertEqual(agent.max_retries, 2)

    @patch('market_agents.agents.base_agent.agent.ParallelAIUtilities')
    def test_agent_with_custom_config(self, mock_ai):
        agent = Agent(
            role="custom_role",
            system="Custom system prompt",
            max_retries=5,
            llm_config=LLMConfig(client="mock", model="mock-model")
        )
        logger.debug(f"Agent role: {agent.role}")
        logger.debug(f"Agent system prompt: {agent.system}")
        logger.debug(f"Agent max retries: {agent.max_retries}")
        logger.debug(f"Agent LLM model: {agent.llm_config.model}")
        
        self.assertEqual(agent.role, "custom_role")
        self.assertEqual(agent.system, "Custom system prompt")
        self.assertEqual(agent.max_retries, 5)
        self.assertEqual(agent.llm_config.model, "mock-model")

    @patch('market_agents.agents.base_agent.agent.ParallelAIUtilities')
    def test_agent_execute(self, mock_ai):
        mock_instance = mock_ai.return_value
        mock_instance.generate.return_value = LLMOutput(
            content=json.dumps(self.mock_responses["chain_of_thought"])
        )

        agent = Agent(
            role="default",
            system="Test system",
            llm_config=LLMConfig(client="mock", model="mock-model")
        )
        task = "Provide a brief summary of the benefits of exercise."
        result = self.loop.run_until_complete(agent.execute(task=task, output_format=ChainOfThoughtSchema))
        logger.debug(f"Agent execution result: {result}")
        self.assertIsInstance(result, dict)

    @patch('market_agents.agents.base_agent.agent.ParallelAIUtilities')
    def test_agent_with_task(self, mock_ai):
        mock_instance = mock_ai.return_value
        mock_instance.generate.return_value = LLMOutput(
            content=json.dumps(self.mock_responses["action_response"])
        )

        agent = Agent(
            role="default",
            system="You are a task execution agent. Respond with a brief confirmation.",
            task="Confirm that you received this task.",
            llm_config=LLMConfig(client="mock", model="mock-model")
        )

        result = self.loop.run_until_complete(agent.execute(output_format=Action))
        logger.debug(f"Agent task execution result: {result}")
        self.assertIsInstance(result, dict)


    @patch('market_agents.agents.base_agent.agent.ParallelAIUtilities')
    def test_react_agent(self, mock_ai):
        mock_instance = mock_ai.return_value
        current_time = datetime.now().timestamp()
        
            # Create a mock response with all required fields
        mock_output = LLMOutput(
            content=json.dumps(self.mock_responses["react_complex"]),
            raw_result={
                "provider": "openai",  
                "model": "gpt-4o-mini",
                "response": self.mock_responses["react_complex"],
                "raw_response": {
                    "id": "mock-id",
                    "object": "chat.completion",
                    "created": int(current_time),
                    "model": "gpt-4o-mini",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(self.mock_responses["react_complex"])
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 100,
                        "total_tokens": 200
                    }
                }
            },
            start_time=current_time,
            end_time=current_time + 1.0,
            source_id="mock-source"
        )
        # Set up the mock to return our properly formatted output
        async def mock_completion(*args, **kwargs):
            return [mock_output]
        
        mock_instance.run_parallel_ai_completion.side_effect = mock_completion

        # Update the agent configuration to use a valid client
        agent = Agent(
            role="react_agent",
            system="You are a ReAct agent. Your task is to solve problems step by step.",
            llm_config=LLMConfig(client="openai", model="gpt-4o-mini")  # Use a valid client
        )


        task = "Calculate the sum of the first 10 prime numbers."
        result = self.loop.run_until_complete(agent.execute(task, ReActSchema))
        logger.debug(f"ReAct agent execution result: {result}")

        # Parse the result
        react_output = ReActSchema(**result)

        logger.debug(f"ReAct thought: {react_output.thought}")
        logger.debug(f"ReAct action: {react_output.action}")
        logger.debug(f"ReAct observation: {react_output.observation}")
        logger.debug(f"ReAct final answer: {react_output.final_answer}")

        # Assertions
        self.assertIsNotNone(react_output.thought)
        self.assertNotEqual(react_output.thought, "")

        self.assertIsInstance(react_output.action, Action)
        self.assertIsNotNone(react_output.action.name)
        self.assertNotEqual(react_output.action.name, "")
        self.assertIsNotNone(react_output.action.input)
        self.assertNotEqual(react_output.action.input, "")
        
        self.assertNotEqual(react_output.observation, "")
        self.assertIsNotNone(react_output.final_answer)
        self.assertNotEqual(react_output.final_answer, "")

        self.assertIn("calculate", react_output.action.name.lower())
        self.assertTrue(any(char.isdigit() for char in react_output.final_answer))

    @patch('market_agents.agents.base_agent.agent.ParallelAIUtilities')
    def test_agent_interactions_logging(self, mock_ai):
        mock_instance = mock_ai.return_value
        mock_instance.generate.return_value = LLMOutput(
            content=json.dumps(self.mock_responses["action_response"])
        )

        agent = Agent(role="test_role", llm_config=LLMConfig(client="mock", model="mock-model"))
        self.loop.run_until_complete(agent.execute("Test task", output_format=Action))

        logger.debug(f"Number of agent interactions: {len(agent.interactions)}")
        if agent.interactions:
            logger.debug(f"First interaction: {agent.interactions[0]}")

        self.assertEqual(len(agent.interactions), 1)
        interaction = agent.interactions[0]
        self.assertIn("id", interaction)
        self.assertIn("name", interaction)
        self.assertIn("system", interaction)
        self.assertIn("task", interaction)
        self.assertIn("response", interaction)
        self.assertIn("timestamp", interaction)

if __name__ == '__main__':
    unittest.main()