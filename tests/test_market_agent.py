import asyncio
import logging
from pathlib import Path
from unittest import IsolatedAsyncioTestCase
import unittest

from market_agents.agents.cognitive_schemas import ReActSchema
from market_agents.agents.market_agent import MarketAgent
from market_agents.agents.personas.persona import Persona
from market_agents.environments.environment import MultiAgentEnvironment
from market_agents.environments.mechanisms.group_chat import GroupChat
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.config import load_config_from_yaml
from minference.lite.models import LLMConfig, ResponseFormat
from market_agents.agents.protocols.acl_message import ACLMessage
from market_agents.economics.econ_agent import EconomicAgent
from market_agents.agents.cognitive_steps import (
    PerceptionStep, ActionStep, ReflectionStep, CognitiveEpisode
)
from colorama import Fore, Style

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMarketAgent(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Initialize test environment
        self.chat_env = MultiAgentEnvironment(
            name="TestGroupChat",
            address="test_chat_1",
            max_steps=5,
            mechanism=GroupChat(
                max_rounds=5,
                sequential=False,
                topics={"1": "Market Analysis", "2": "Trading Strategy"}
            )
        )

        # Configure LLM
        llm_config = LLMConfig(
            client="openai",
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1024,
            response_format=ResponseFormat.auto_tools
        )

        # Initialize economic agent with holdings
        econ_agent = EconomicAgent(
            initial_holdings={"USDC": 1000.0, "ETH": 5},
            generate_wallet=True
        )

        # Create persona
        persona = Persona(
            name="Warren Buffet",
            role="market_analyst",
            persona="Well-known investor with a multi-billion dollar fund",
            objectives=["maximize stock returns", "investment influencer"],
            trader_type=["Risk Averse", "value stocks"],
            communication_style="diplomatic",
            routines=["breakfast at mcdonalds", "start trading day"],
            skills=["trading", "investment"]
        )

        # Load storage config and initialize agent
        storage_config_path = Path("market_agents/memory/storage_config.yaml")
        storage_config = load_config_from_yaml(str(storage_config_path))
        storage_utils = AgentStorageAPIUtils(config=storage_config)

        # Create MarketAgent with test environment
        self.agent = await MarketAgent.create(
            storage_utils=storage_utils,
            agent_id="agent_007",
            econ_agent=econ_agent,
            environments={"chat": self.chat_env},
            llm_config=llm_config,
            protocol=ACLMessage
        )

    async def test_perception_step(self):
        """Test running perception step individually"""
        self.chat_env.current_step = 2

        perception_result = await self.agent.run_step(
            step=PerceptionStep(
                agent_id=self.agent.id,
                environment_name="chat",
                environment_info=self.chat_env.get_global_state(),
                structured_tool=True
            )
        )
        
        print(f"{Fore.GREEN}Perception result: {perception_result}{Style.RESET_ALL}")
        self.assertIsNotNone(perception_result)
        self.assertIsInstance(perception_result, dict)
        self.assertIn("monologue", perception_result)
        self.assertIn("key_observations", perception_result)

    async def test_action_step_react(self):
        """Test running action step with ReAct schema"""
        self.chat_env.current_step = 2
        action_result = await self.agent.run_step(
            step=ActionStep(
                agent_id=self.agent.id,
                environment_name="chat",
                environment_info=self.chat_env.get_global_state(),
                structured_tool=True,
                action_schema=ReActSchema.model_json_schema()
            )
        )
        
        print(f"{Fore.BLUE}Action result (ReAct): {action_result}{Style.RESET_ALL}")
        self.assertIsNotNone(action_result)
        self.assertIsInstance(action_result, dict)
        self.assertIn("thought", action_result)
        self.assertIn("action", action_result)

    async def test_reflection_step(self):
        """Test running reflection step individually"""
        self.chat_env.current_step = 2

        reflection_result = await self.agent.run_step(
            step=ReflectionStep(
                agent_id=self.agent.id,
                environment_name="chat",
                environment_info=self.chat_env.get_global_state(),
                structured_tool=True
            )
        )
        
        print(f"{Fore.YELLOW}Reflection result: {reflection_result}{Style.RESET_ALL}")
        self.assertIsNotNone(reflection_result)
        self.assertIsInstance(reflection_result, dict)
        self.assertIn("reflection", reflection_result)
        self.assertIn("self_critique", reflection_result)

    async def test_default_episode(self):
        """Test running default cognitive episode"""
        self.chat_env.current_step = 2

        self.agent.task = "Discuss the current topic in group chat"
        self.agent._refresh_prompts()

        results = await self.agent.run_episode(environment_name="chat")
        
        print(f"{Fore.CYAN}Default Episode Results:{Style.RESET_ALL}")
        for i, result in enumerate(["Perception", "Action", "Reflection"]):
            print(f"{Fore.CYAN}{result}: {results[i]}{Style.RESET_ALL}")
        
        self.assertEqual(len(results), 3)
        self.assertTrue(all(result is not None for result in results))

    async def test_custom_schema_episode(self):
        """Test running episode with custom schemas"""
        self.chat_env.current_step = 2

        self.agent.task = "Discuss current topic in group chat"
        self.agent._refresh_prompts()

        custom_episode = CognitiveEpisode(
            steps=[PerceptionStep, ActionStep, ReflectionStep],
            environment_name="chat"
        )

        results = await self.agent.run_episode(episode=custom_episode)
        
        print(f"{Fore.MAGENTA}Custom Schema Episode Results:{Style.RESET_ALL}")
        for i, result in enumerate(results):
            print(f"{Fore.MAGENTA}Step {i+1}: {result}{Style.RESET_ALL}")
        
        self.assertEqual(len(results), 3)
        self.assertIsInstance(results[0], dict)
        self.assertIsInstance(results[1], dict)
        self.assertIsInstance(results[2], dict)

if __name__ == '__main__':
    asyncio.run(unittest.main())