import unittest
import logging
import asyncio
from uuid import uuid4

from market_agents.agents.cognitive_schemas import (
    ChainOfThoughtSchema, 
    ReActSchema, 
    PerceptionSchema, 
    ReflectionSchema,
    Action
)
from minference.enregistry import EntityRegistry
from minference.lite.models import (
    LLMConfig,
    LLMClient,
    ResponseFormat,
    StructuredTool
)

from market_agents.agents.base_agent.agent import Agent

# Set up test logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EntityRegistry()._logger = logger

class TestAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that should be shared across all tests."""
        cls.loop = asyncio.get_event_loop()

    def setUp(self):
        """Set up test fixtures."""
        # Default config for text responses
        self.text_config = LLMConfig(
            client=LLMClient.openai,
            model="gpt-4o",
            response_format=ResponseFormat.text,
            max_tokens=250
        )
        
        # Config for structured/schema responses
        self.structured_config = LLMConfig(
            client=LLMClient.openai,
            model="gpt-4o",
            response_format=ResponseFormat.auto_tools,
            max_tokens=1024
        )

        # Create structured tools from schema models
        self.cot_tool = StructuredTool(
            json_schema=ChainOfThoughtSchema.model_json_schema(),
            name="chain_of_thought",
            description="Generate step-by-step reasoning with final answer"
        )

        # Create structured tools from schema models
        self.cot_tool = StructuredTool(
            json_schema=ChainOfThoughtSchema.model_json_schema(),
            name="chain_of_thought",
            description="Generate step-by-step reasoning with final answer"
        )

        self.react_tool = StructuredTool(
            json_schema=ReActSchema.model_json_schema(),
            name="react_reasoning",
            description="Generate thought-action-observation cycle"
        )

        self.perception_tool = StructuredTool(
            json_schema=PerceptionSchema.model_json_schema(),
            name="perception",
            description="Analyze environment and generate perceptions"
        )

        self.reflection_tool = StructuredTool(
            json_schema=ReflectionSchema.model_json_schema(),
            name="reflection",
            description="Reflect on actions and generate insights"
        )
    def test_agent_text_response(self):
        """Test agent with plain text response."""
        agent = Agent(
            role="text_agent",
            task="Write a one sentence summary of what an LLM is.",
            llm_config=self.text_config,
            tools=[]
        )
        
        result = self.loop.run_until_complete(agent.execute())
        print(f"\nText Response: {result}")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_agent_chain_of_thought(self):
        """Test agent with chain of thought reasoning."""
        agent = Agent(
            role="reasoning_agent",
            task="Calculate the sum of numbers from 1 to 5 step by step.",
            llm_config=self.structured_config,
            tools=[self.cot_tool]
        )
        
        result = self.loop.run_until_complete(agent.execute())
        print(f"\nChain of Thought Response: {result}")
        cot_output = ChainOfThoughtSchema(**result)
        
        self.assertTrue(len(cot_output.thoughts) > 0)
        self.assertTrue(len(cot_output.final_answer) > 0)

    def test_agent_react_reasoning(self):
        """Test agent with ReAct reasoning pattern."""
        agent = Agent(
            role="react_agent",
            task="Find the sum of the first 3 prime numbers. Show your work.",
            llm_config=self.structured_config,
            tools=[self.react_tool]
        )
        
        result = self.loop.run_until_complete(agent.execute())
        print(f"\nReAct Response: {result}")
        react_output = ReActSchema(**result)
        
        self.assertTrue(len(react_output.thought) > 0)
        if react_output.action:
            self.assertIsInstance(react_output.action, Action)

    def test_agent_perception(self):
        """Test agent with perception schema."""
        agent = Agent(
            role="perceptive_agent",
            task="Analyze the current market conditions: high inflation, rising interest rates, and volatile stock market.",
            llm_config=self.structured_config,
            tools=[self.perception_tool]
        )
        
        result = self.loop.run_until_complete(agent.execute())
        print(f"\nPerception Response: {result}")
        perception_output = PerceptionSchema(**result)
        
        self.assertTrue(len(perception_output.monologue) > 0)
        self.assertTrue(len(perception_output.key_observations) > 0)
        self.assertTrue(len(perception_output.strategy) > 0)
        self.assertTrue(0 <= perception_output.confidence <= 1)

    def test_agent_reflection(self):
        """Test agent with reflection schema."""
        agent = Agent(
            role="reflective_agent",
            task="Reflect on a trading strategy that resulted in a 10% loss due to unexpected market volatility.",
            llm_config=self.structured_config,
            tools=[self.reflection_tool]
        )
        
        result = self.loop.run_until_complete(agent.execute())
        print(f"\nReflection Response: {result}")
        reflection_output = ReflectionSchema(**result)
        
        self.assertTrue(len(reflection_output.reflection) > 0)
        self.assertTrue(len(reflection_output.self_critique) > 0)
        self.assertTrue(isinstance(reflection_output.self_reward, float))
        self.assertTrue(len(reflection_output.strategy_update) > 0)

    def test_schema_validation(self):
        """Test schema validation for various cognitive schemas."""
        # Test ChainOfThoughtSchema
        cot_data = {
            "thoughts": [{"reasoning": "First step"}, {"reasoning": "Second step"}],
            "final_answer": "The answer is 42"
        }
        cot = ChainOfThoughtSchema(**cot_data)
        self.assertEqual(len(cot.thoughts), 2)
        
        # Test ReActSchema
        react_data = {
            "thought": "I should calculate this",
            "action": {"name": "calculate", "input": "1 + 1"},
            "observation": "The result is 2",
            "final_answer": "2"
        }
        react = ReActSchema(**react_data)
        self.assertEqual(react.action.name, "calculate")
        
        # Test PerceptionSchema
        perception_data = {
            "monologue": "Analyzing market conditions",
            "key_observations": ["High volatility", "Increasing rates"],
            "strategy": ["Reduce risk", "Increase cash position"],
            "confidence": 0.8
        }
        perception = PerceptionSchema(**perception_data)
        self.assertEqual(len(perception.key_observations), 2)
        
        # Test ReflectionSchema
        reflection_data = {
            "reflection": "The strategy was too aggressive",
            "self_critique": ["Ignored market signals", "Poor timing"],
            "self_reward": 0.3,
            "strategy_update": ["Implement stricter stop-loss", "Better risk management"]
        }
        reflection = ReflectionSchema(**reflection_data)
        self.assertEqual(len(reflection.strategy_update), 2)

if __name__ == '__main__':
    unittest.main()