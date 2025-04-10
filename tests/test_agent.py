import unittest
import logging
import asyncio
from uuid import uuid4
from minference.lite.inference import InferenceOrchestrator

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
from market_agents.agents.personas.persona import Persona

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EntityRegistry()._logger = logger

class TestAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that should be shared across all tests."""
        cls.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(cls.loop)

    def setUp(self):
        """Set up test fixtures."""
        # Create LLM orchestrator
        self.llm_orchestrator = InferenceOrchestrator()
        
        # Default config for text responses
        self.text_config = LLMConfig(
            client=LLMClient.openai,
            model="gpt-4",
            response_format=ResponseFormat.text,
            max_tokens=250
        )
        
        # Config for structured/schema responses
        self.structured_config = LLMConfig(
            client=LLMClient.openai,
            model="gpt-4",
            response_format=ResponseFormat.auto_tools,
            max_tokens=1024
        )

        # Create test persona
        self.test_persona = Persona(
            role="AI Test Assistant",
            persona="I am an AI test assistant focused on validating agent behaviors.",
            objectives=["Test agent functionality", "Validate responses"],
            skills=["Testing", "Debugging", "Schema Validation"]
        )

        # Create structured tools
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
                name="text-agent",
                persona="You are a helpful AI assistant",
                task="Write a one sentence summary of what an LLM is.",
                llm_config=self.text_config,
                llm_orchestrator=self.llm_orchestrator,
                tools=[]
            )
            
            result = self.loop.run_until_complete(agent.execute())
            print(f"\nText Response: {result}")
            self.assertIsInstance(result, str)
            self.assertTrue(len(result) > 0)

    def test_agent_chain_of_thought(self):
        """Test agent with chain of thought reasoning."""
        agent = Agent(
            name="reasoning-agent",
            persona=self.test_persona,  # Using Persona object
            task="Calculate the sum of numbers from 1 to 5 step by step.",
            llm_config=self.structured_config,
            llm_orchestrator=self.llm_orchestrator,
            tools=[self.cot_tool]
        )
        
        result = self.loop.run_until_complete(agent.execute())
        print(f"\nChain of Thought Response: {result}")
        cot_output = ChainOfThoughtSchema(**result)
        
        self.assertTrue(len(cot_output.thoughts) > 0)
        self.assertTrue(len(cot_output.final_answer) > 0)

    def test_agent_perception(self):
        """Test agent with perception schema."""
        agent = Agent(
            name="perception-agent",
            persona=self.test_persona,
            task="Analyze the current market conditions: high inflation, rising interest rates, and volatile stock market.",
            llm_config=self.structured_config,
            llm_orchestrator=self.llm_orchestrator,
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
            name="reflection-agent",
            persona=self.test_persona,
            task="Reflect on a trading strategy that resulted in a 10% loss due to unexpected market volatility.",
            llm_config=self.structured_config,
            llm_orchestrator=self.llm_orchestrator,
            tools=[self.reflection_tool]
        )
        
        result = self.loop.run_until_complete(agent.execute())
        print(f"\nReflection Response: {result}")
        reflection_output = ReflectionSchema(**result)
        
        self.assertTrue(len(reflection_output.reflection) > 0)
        self.assertTrue(len(reflection_output.self_critique) > 0)
        self.assertTrue(isinstance(reflection_output.self_reward, float))
        self.assertTrue(len(reflection_output.strategy_update) > 0)

    def test_persona_handling(self):
        """Test different types of persona inputs."""
        # Test with string persona
        string_agent = Agent(
            name="string-agent",
            persona="Simple string persona",
            llm_config=self.text_config,
            llm_orchestrator=self.llm_orchestrator
        )
        self.assertEqual(string_agent.persona_prompt, "Simple string persona")

        # Test with Persona object
        persona_agent = Agent(
            name="persona-agent",
            persona=self.test_persona,
            llm_config=self.text_config,
            llm_orchestrator=self.llm_orchestrator
        )
        self.assertIn("AI Test Assistant", persona_agent.persona_prompt)
        self.assertIn("Test agent functionality", persona_agent.persona_prompt)

if __name__ == '__main__':
    unittest.main()