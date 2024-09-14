import asyncio
from dotenv import load_dotenv
from market_agents.inference.parallel_inference import ParallelAIUtilities, RequestLimits
from market_agents.inference.message_models import LLMConfig
from market_agents.simple_agent import SimpleAgent, create_simple_agent
from market_agents.economics.econ_models import Good, Endowment, Basket
from market_agents.inference.message_models import LLMPromptContext
from typing import List

async def main():
    load_dotenv()
    
    # Set up ParallelAIUtilities
    oai_request_limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=200000)
    anthropic_request_limits = RequestLimits(max_requests_per_minute=40, max_tokens_per_minute=40000)
    parallel_ai = ParallelAIUtilities(oai_request_limits=oai_request_limits, anthropic_request_limits=anthropic_request_limits)

    # Create a good
    apple = Good(name="apple", quantity=0)  # Quantity is set to 0 as it will be defined in the endowment

    # Create LLM configs
    buyer_llm_config = LLMConfig(client="anthropic", model="claude-3-5-sonnet-20240620", response_format="tool")
    seller_llm_config = LLMConfig(client="anthropic", model="claude-3-5-sonnet-20240620", response_format="tool")

    # Create simple agents
    buyer = create_simple_agent(
        agent_id="buyer_1",
        llm_config=buyer_llm_config,
        good=apple,
        is_buyer=True,
        endowment=Endowment(agent_id="buyer_1", initial_basket=Basket(cash=1000, goods=[Good(name="apple", quantity=0)])),
        starting_value=20.0,
        num_units=10
    )

    seller = create_simple_agent(
        agent_id="seller_1",
        llm_config=seller_llm_config,
        good=apple,
        is_buyer=False,
        endowment=Endowment(agent_id="seller_1", initial_basket=Basket(cash=0, goods=[Good(name="apple", quantity=10)])),
        starting_value=15.0,
        num_units=10
    )

    # Prepare agents for parallel inference
    agents = [buyer, seller]

    # Run parallel completions
    print("Running parallel completions...")
    typed_agents : List[LLMPromptContext] = [agent for agent in agents if isinstance(agent, LLMPromptContext)]
    completion_results = await parallel_ai.run_parallel_ai_completion(typed_agents)

    # Print results
    for agent, result in zip(agents, completion_results):
        print(f"\nAgent ID: {agent.id}")
        print(f"Role: {'Buyer' if agent.is_buyer else 'Seller'}")
        print(f"Response: {result.str_content}")
        if result.json_object:
            print(f"Structured Output: {result.json_object}")
        print(f"Usage: {result.usage}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())