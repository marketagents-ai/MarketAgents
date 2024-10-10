import asyncio
from dotenv import load_dotenv
from market_agents.inference.parallel_inference import ParallelAIUtilities, RequestLimits
from market_agents.inference.message_models import LLMConfig
from market_agents.simple_agent import SimpleAgent, create_simple_agent
from market_agents.economics.econ_agent import ZiFactory, ZiParams
from market_agents.economics.econ_models import Good, Endowment, Basket
from market_agents.economics.scenario import Scenario
from market_agents.environments.mechanisms.auction import DoubleAuction, AuctionActionSpace, AuctionObservationSpace
from market_agents.environments.environment import MultiAgentEnvironment
from market_agents.market_orchestrator import MarketOrchestrator
import os


async def zi_exploration():
    load_dotenv()
    
    # Set up ParallelAIUtilities
    # Set up ParallelAIUtilities
    oai_request_limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=200000)
    anthropic_request_limits = RequestLimits(max_requests_per_minute=40, max_tokens_per_minute=10000)
    parallel_ai = ParallelAIUtilities(
        oai_request_limits=oai_request_limits,
        anthropic_request_limits=anthropic_request_limits
    )

    # Create a good
    apple = Good(name="apple", quantity=0)
    

    # Create LLM configs
    # buyer_llm_config = LLMConfig(client="anthropic", model="claude-3-5-sonnet-20240620", response_format="tool", max_tokens=50)
    # seller_llm_config = LLMConfig(client="anthropic", model="claude-3-5-sonnet-20240620", response_format="tool", max_tokens=50)
    
    #vllm config
    # if vllm_model is not None:
    #     buyer_llm_config = LLMConfig(client="vllm", model=vllm_model, response_format="structured_output", max_tokens=50)
    #     seller_llm_config = LLMConfig(client="vllm", model=vllm_model, response_format="structured_output", max_tokens=50)
    # # #litellm config
    # if litellm_model is not None:
    #     buyer_llm_config = LLMConfig(client="litellm", model=litellm_model, response_format="tool", max_tokens=50)
    #     seller_llm_config = LLMConfig(client="litellm", model=litellm_model, response_format="tool", max_tokens=50)
    buyer_params = ZiParams(
        id="buyer_template",
        initial_cash=10000.0,
        initial_goods={"apple": 0},
        base_values={"apple": 20.0},
        num_units=5,
        noise_factor=0.05,
        max_relative_spread=0.2,
        is_buyer=True
    )

    seller_params = ZiParams(
        id="seller_template",
        initial_cash=0,
        initial_goods={"apple": 20},
        base_values={"apple": 15.0},
        num_units=5,
        noise_factor=0.05,
        max_relative_spread=0.2,
        is_buyer=False
    )
    # Create DoubleAuction mechanism
    factories = [
        ZiFactory(
            id=f"factory_episode_{0}",
            goods=["apple"],
            num_buyers=25,  # Increase buyers by 1 each episode
            num_sellers=25,     # Keep sellers constant
            buyer_params=buyer_params,
            seller_params=seller_params
        )
    ]
    scenario = Scenario(
        name="Static Apple Market",
        goods=["apple"],
        factories=factories
    )

    orchestrator = MarketOrchestrator(llm_agents=[], goods=[apple.name], ai_utils=parallel_ai,max_rounds=1,scenario=scenario)

    # Run the market simulation
    await orchestrator.run_scenario()
    #plot the market results
    return orchestrator.state

if __name__ == "__main__":
    state = asyncio.run(zi_exploration())
    print(state)
