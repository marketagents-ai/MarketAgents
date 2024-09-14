import asyncio
from dotenv import load_dotenv
from market_agents.inference.parallel_inference import ParallelAIUtilities, RequestLimits
from market_agents.inference.message_models import LLMConfig
from market_agents.simple_agent import SimpleAgent, create_simple_agent
from market_agents.economics.econ_models import Good, Endowment, Basket
from market_agents.environments.mechanisms.auction import DoubleAuction, AuctionActionSpace, AuctionObservationSpace
from market_agents.environments.environment import MultiAgentEnvironment
from market_agents.orchestrator import MarketOrchestrator
from typing import List

async def main():
    load_dotenv()
    
    # Set up ParallelAIUtilities
    oai_request_limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=200000)
    anthropic_request_limits = RequestLimits(max_requests_per_minute=40, max_tokens_per_minute=40000)
    parallel_ai = ParallelAIUtilities(
        oai_request_limits=oai_request_limits,
        anthropic_request_limits=anthropic_request_limits
    )

    # Create a good
    apple = Good(name="apple", quantity=0)

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
        starting_value=10.0,
        num_units=10
    )

    agents = [buyer, seller]
    print(f"Seller cost schedule: {seller.cost_schedules}")
    print(f"Seller value schedule: {seller.value_schedules}")
    # Create DoubleAuction mechanism
    double_auction = DoubleAuction(
        max_rounds=100,
        good_name="apple"
    )
    # Print starting state
    print("\nStarting State:")
    for agent in agents:
        print(f"Printing numb agents {len(agents)}")
        print(f"\nAgent ID: {agent.id}")
        print(f"Role: {'Buyer' if agent.is_buyer('apple') else 'Seller'}")
        if agent.is_buyer('apple'):
            print(f"Current schedule: {agent.value_schedules['apple']}")
        elif agent.is_seller('apple'):
            print(f"Current schedule: {agent.cost_schedules['apple']}")
        else:
            raise ValueError(f"Agent {agent.id} is not a buyer or seller of apple")
        print(f"Current Endowment: {agent.endowment.current_basket}")
        print("Chat History:")
        for message in agent.messages:
            print(f"  {message['role'].capitalize()}: {message['content']}")
        print("-" * 50)
    # Create MultiAgentEnvironment
    environment = MultiAgentEnvironment(
        name="Apple Market",
        address="apple_market",
        max_steps=100,
        action_space=AuctionActionSpace(),
        observation_space=AuctionObservationSpace(),
        mechanism=double_auction
    )
    assert isinstance(environment.mechanism, DoubleAuction)

    # Create MarketOrchestrator
    orchestrator = MarketOrchestrator(agents=agents, markets=[environment], ai_utils=parallel_ai)

    # Run the simulation
    max_rounds = 10
    for round in range(max_rounds):
        print(f"\nRound {round + 1}")
        
        # Run one step of the orchestrator
        await orchestrator.run_auction_step("apple")

        # Check if any trades were executed
        trades = environment.mechanism.trades
        if trades:
            print(f"Trade executed at price: {trades[-1].price}")
            
    
    #Print final state
    print("\nFinal State:")
    for agent in agents:
        print(f"\nAgent ID: {agent.id}")
        print(f"Role: {'Buyer' if agent.is_buyer('apple') else 'Seller'}")
        print(f"Current Endowment: {agent.endowment.current_basket}")
        print("Chat History:")
        for message in agent.messages:
            print(f"  {message['role'].capitalize()}: {message['content']}")
        print("-" * 50)

    if not environment.mechanism.trades:
        print("No trade was executed within the maximum number of rounds.")
    print(f"Failed actions: {orchestrator.failed_actions}")
    for agent in agents:
        print(f"\nAgent ID: {agent.id}")
        print(f"Current Endowment: {agent.endowment.current_basket}")

if __name__ == "__main__":
    asyncio.run(main())