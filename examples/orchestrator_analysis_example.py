import asyncio
from dotenv import load_dotenv
from market_agents.inference.parallel_inference import ParallelAIUtilities, RequestLimits
from market_agents.inference.message_models import LLMConfig
from market_agents.simple_agent import SimpleAgent, create_simple_agent
from market_agents.economics.econ_models import Good, Endowment, Basket
from market_agents.environments.mechanisms.auction import DoubleAuction, AuctionActionSpace, AuctionObservationSpace
from market_agents.environments.environment import MultiAgentEnvironment
from market_agents.market_orchestrator import MarketOrchestrator
import os
import datetime
async def main():
    load_dotenv()

    start_time = datetime.datetime.now()
    # Set up ParallelAIUtilities
    oai_request_limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=200000)
    anthropic_request_limits = RequestLimits(max_requests_per_minute=40, max_tokens_per_minute=10000)
    litellm_request_limits = RequestLimits(max_requests_per_minute=1000000, max_tokens_per_minute=10000000000)
    parallel_ai = ParallelAIUtilities(
        oai_request_limits=oai_request_limits,
        anthropic_request_limits=anthropic_request_limits,
        litellm_request_limits=litellm_request_limits
    )

    # Create a good
    apple = Good(name="apple", quantity=0)
    num_buyers = 5000
    num_sellers = 5000
    # Create LLM configs
    # buyer_llm_config = LLMConfig(client="anthropic", model="claude-3-5-sonnet-20240620", response_format="tool", max_tokens=50)
    # seller_llm_config = LLMConfig(client="anthropic", model="claude-3-5-sonnet-20240620", response_format="tool", max_tokens=50)
    vllm_model = os.getenv("VLLM_MODEL")
    litellm_model = os.getenv("LITELLM_MODEL")
    # buyer_llm_config = LLMConfig(client="openai", model="gpt-4o-mini", response_format="tool", max_tokens=50)
    # seller_llm_config = LLMConfig(client="openai", model="gpt-4o-mini", response_format="tool", max_tokens=50)
    # #vllm config
    if vllm_model is not None:
        buyer_llm_config = LLMConfig(client="vllm", model=vllm_model, response_format="tool", max_tokens=50)
        seller_llm_config = LLMConfig(client="vllm", model=vllm_model, response_format="tool", max_tokens=50)
    # # #litellm config
    if litellm_model is not None:
        model_name_base = litellm_model[:-1]
        lite_llm_buyer_configs=[]
        lite_llm_seller_configs=[]
        num_models =6 
        
        for i in range(num_buyers):
            model_block = i%num_models
            model_name = model_name_base + str(model_block)
            lite_llm_buyer_configs.append(LLMConfig(client="litellm", model=model_name, response_format="tool", max_tokens=50))
        for i in range(num_sellers):
            model_block = i%num_models
            model_name = model_name_base + str(model_block)
            lite_llm_seller_configs.append(LLMConfig(client="litellm", model=model_name, response_format="tool", max_tokens=50))

    
    # Create simple agents
    buyers = [
        create_simple_agent(
            agent_id=f"buyer_{i}",
            llm_config=buyer_llm_config if litellm_model is None else lite_llm_buyer_configs[i],
            good=apple,
            is_buyer=True,
            endowment=Endowment(agent_id=f"buyer_{i}", initial_basket=Basket(cash=1000, goods=[Good(name="apple", quantity=0)])),
            starting_value=20.0,  # Slightly different values for each buyer
            num_units=10
        ) for i in range(num_buyers)  # Create 5 buyers
    ]

    sellers = [
        create_simple_agent(
            agent_id=f"seller_{i}",
            llm_config=seller_llm_config if litellm_model is None else lite_llm_seller_configs[i],
            good=apple,
            is_buyer=False,
            endowment=Endowment(agent_id=f"seller_{i}", initial_basket=Basket(cash=0, goods=[Good(name="apple", quantity=10)])),
            starting_value=10.0,  # Slightly different values for each seller
            num_units=10
        ) for i in range(num_sellers)  # Create 5 sellers
    ]

    agents = buyers + sellers

    # Create DoubleAuction mechanism


    orchestrator = MarketOrchestrator(llm_agents=agents, goods=[apple.name], ai_utils=parallel_ai,max_rounds=5)

    # Run the market simulation
    await orchestrator.run_scenario()
    #plot the market results
    print(orchestrator.state)
    end_time = datetime.datetime.now()
    print(f"Time taken: {end_time - start_time}")
    return orchestrator
if __name__ == "__main__":
    orchestrator = asyncio.run(main())
    sate = orchestrator.state
