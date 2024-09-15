import asyncio
import logging
from unittest import IsolatedAsyncioTestCase
import unittest
from market_agents.agents.market_agent import MarketAgent
from market_agents.environments.environment import MultiAgentEnvironment
from market_agents.environments.mechanisms.auction import DoubleAuction
from market_agents.orchestrator import MarketOrchestrator
from market_agents.inference.message_models import LLMConfig
from market_agents.agents.protocols.acl_message import ACLMessage
from market_agents.economics.econ_models import Endowment, Basket, Good
from market_agents.economics.econ_agent import BuyerPreferenceSchedule
from market_agents.inference.parallel_inference import ParallelAIUtilities
from colorama import Fore, Style

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMarketAgentBase(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Initialize the auction environment
        self.auction_env = MultiAgentEnvironment(
            name="TestAuction",
            address="test_auction_1",
            max_steps=5,
            mechanism=DoubleAuction(sequential=False, max_rounds=5, good_name="apple")
        )

        # Initialize the LLMConfig
        llm_config = LLMConfig(
            client="openai",
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1024
        )

        # Create initial basket and endowment
        initial_basket = Basket(
            cash=1000.0,
            goods=[Good(name="apple", quantity=5)]
        )
        endowment = Endowment(
            initial_basket=initial_basket,
            agent_id="0"
        )

        # Create preference schedules
        value_schedules = {
            "apple": BuyerPreferenceSchedule(
                num_units=20,
                base_value=10.0,
                noise_factor=0.1
            )
        }
        cost_schedules = {}  # Empty for buyer

        # Initialize the agent
        self.agent = MarketAgent(
            id="0",
            endowment=endowment,
            value_schedules=value_schedules,
            cost_schedules=cost_schedules,
            max_relative_spread=0.2,
            role="buyer",
            environments={"auction": self.auction_env},
            llm_config=llm_config,
            protocol=ACLMessage,  # Provide the ACLMessage protocol
            address="agent_0_address"
        )

        # Initialize ParallelAIUtilities (you may need to adjust this based on your actual implementation)
        self.ai_utils = ParallelAIUtilities()

        # Initialize the orchestrator
        self.orchestrator = MarketOrchestrator(
            agents=[self.agent],
            markets=[self.auction_env],
            ai_utils=self.ai_utils
        )

    async def test_generate_action(self):
        await self.asyncSetUp()  # Ensure setup is called before the test
        self.auction_env.current_step = 2
        action = await self.agent.generate_action("auction")
        print(f"{Fore.GREEN}Generated action: {action}{Style.RESET_ALL}")
        self.assertIsNotNone(action)
        self.assertIsInstance(action, dict)

    async def test_perceive(self):
        await self.asyncSetUp()  # Ensure setup is called before the test
        self.auction_env.current_step = 3
        perception = await self.agent.perceive("auction")
        print(f"{Fore.BLUE}Perception: {perception}{Style.RESET_ALL}")
        self.assertIsNotNone(perception)

    async def test_reflect(self):
        await self.asyncSetUp()  # Ensure setup is called before the test
        reflection = await self.agent.reflect("auction")
        print(f"{Fore.YELLOW}Reflection: {reflection}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Reflection message:{Style.RESET_ALL} In the current auction environment, there are no ongoing trades or bids, indicating that the market is inactive at this moment. As I have not taken any previous action, there is no surplus to consider. This state presents an opportunity to observe the market dynamics without any immediate pressure to act. It is essential to monitor the behavior of other participants and identify potential entry points for bidding when the market becomes active.")
        self.assertIsNotNone(reflection)

    async def test_orchestrator(self):
        await self.asyncSetUp()  # Ensure setup is called before the test
        good_name = "apple"
        step_result, surplus = await self.orchestrator.run_auction_step(good_name)
        print(f"{Fore.CYAN}Orchestrator step result:{Style.RESET_ALL} {step_result}")
        print(f"{Fore.MAGENTA}Orchestrator surplus:{Style.RESET_ALL} {surplus}")
        self.assertIsNotNone(step_result)
        self.assertIsNotNone(surplus)

class AsyncTestCase(IsolatedAsyncioTestCase):
    async def test_generate_action(self):
        test_base = TestMarketAgentBase()
        await test_base.asyncSetUp()
        await test_base.test_generate_action()

    async def test_perceive(self):
        test_base = TestMarketAgentBase()
        await test_base.asyncSetUp()
        await test_base.test_perceive()

    async def test_reflect(self):
        test_base = TestMarketAgentBase()
        await test_base.asyncSetUp()
        await test_base.test_reflect()

    async def test_orchestrator(self):
        test_base = TestMarketAgentBase()
        await test_base.asyncSetUp()
        await test_base.test_orchestrator()

if __name__ == '__main__':
    asyncio.run(unittest.main())