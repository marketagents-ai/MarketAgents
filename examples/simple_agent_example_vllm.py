import asyncio
from dotenv import load_dotenv
from market_agents.inference.parallel_inference import ParallelAIUtilities, RequestLimits
from market_agents.inference.message_models import LLMConfig
from market_agents.simple_agent import SimpleAgent, create_simple_agent
from market_agents.economics.econ_models import Good, Endowment, Basket, Bid, Ask, Trade
from market_agents.environments.mechanisms.auction import AuctionObservation, AuctionLocalObservation, AuctionGlobalObservation
from typing import List, Dict, Union
from market_agents.inference.message_models import LLMPromptContext
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
    buyer_llm_config = LLMConfig(client="vllm", model="Hermes-3", response_format="tool")
    seller_llm_config = LLMConfig(client="vllm", model="llama-3.1-70b", response_format="tool")

    #buyer_llm_config = LLMConfig(client="anthropic", model="claude-3-5-sonnet-20240620", response_format="tool")
    #seller_llm_config = LLMConfig(client="anthropic", model="claude-3-5-sonnet-20240620", response_format="tool")

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
    #Print sthe initial  state similar to the one we do at the end
    

    agents = [buyer, seller]


    
    max_rounds = 100
    trade_executed = False
    trade_id = 0
    good_name = "apple"

    print("Initial State:")

    for agent in agents:
        print(f"\nAgent ID: {agent.id}")
        print(f"Role: {'Buyer' if agent.is_buyer(good_name) else 'Seller'}")
        print(f"Current Endowment: {agent.endowment.current_basket}")
        print("Chat History:")
        for message in agent.messages:
            print(f"  {message['role'].capitalize()}: {message['content']}")
        print("-" * 50)

    for round in range(max_rounds):
        print(f"\nRound {round + 1}")
        
        # Run parallel completions
        typed_agents : List[LLMPromptContext] = [p for p in agents if isinstance(p, LLMPromptContext)]
        completion_results = await parallel_ai.run_parallel_ai_completion(typed_agents)

        # Process results and create bids/asks
        actions: Dict[str, Union[Bid, Ask]] = {}
        for agent, result in zip(agents, completion_results):
            if result.json_object:
                if agent.is_buyer(good_name):
                    actions[agent.id] = Bid.model_validate(result.json_object.object)
                    agent.pending_orders.setdefault(good_name, []).append(actions[agent.id])
                else:
                    actions[agent.id] = Ask.model_validate(result.json_object.object)
                    agent.pending_orders.setdefault(good_name, []).append(actions[agent.id])
        # Try to match orders
        if len(actions) == 2:
            buyer_action = actions[buyer.id]
            seller_action = actions[seller.id]
            
            if isinstance(buyer_action, Bid) and isinstance(seller_action, Ask):
                if buyer_action.price >= seller_action.price:
                    trade_price = (buyer_action.price + seller_action.price) / 2
                    trade = Trade(
                        trade_id=trade_id,
                        buyer_id=buyer.id,
                        seller_id=seller.id,
                        price=trade_price,
                        quantity=1,
                        good_name="apple",
                        bid_price=buyer_action.price,
                        ask_price=seller_action.price
                    )
                    
                    # Execute trade
                    buyer.process_trade(trade)
                    seller.process_trade(trade)
                    trade_executed = True

                    # Create AuctionGlobalObservation
                    global_observation = AuctionGlobalObservation(
                        observations={
                            buyer.id: AuctionLocalObservation(
                                agent_id=buyer.id,
                                observation=AuctionObservation(
                                    trades=[trade],
                                    market_summary={"average_price": trade_price},
                                    waiting_orders=[]
                                )
                            ),
                            seller.id: AuctionLocalObservation(
                                agent_id=seller.id,
                                observation=AuctionObservation(
                                    trades=[trade],
                                    market_summary={"average_price": trade_price},
                                    waiting_orders=[]
                                )
                            )
                        },
                        all_trades=[trade],
                        market_summary={"average_price": trade_price}
                    )

                    # Update agents with local observations
                    buyer.update_state(global_observation.observations[buyer.id])
                    seller.update_state(global_observation.observations[seller.id])

                    print(f"Trade executed at price: {trade_price}")
                    break
                else:
                    print("No match: Bid price lower than Ask price")
            else:
                print("Invalid actions received")
        else:
            print("Not enough actions received")

        # Reset outstanding orders if no match
        buyer.reset_pending_orders("apple")
        seller.reset_pending_orders("apple")


    # Print final state
    print("\nFinal State:")
    for agent in agents:
        print(f"\nAgent ID: {agent.id}")
        print(f"Role: {'Buyer' if agent.is_buyer(good_name) else 'Seller'}")
        print(f"Current Endowment: {agent.endowment.current_basket}")
        print("Chat History:")
        for message in agent.messages:
            print(f"  {message['role'].capitalize()}: {message['content']}")
        print("-" * 50)

    if not trade_executed:
        print("No trade was executed within the maximum number of rounds.")

if __name__ == "__main__":
    asyncio.run(main())