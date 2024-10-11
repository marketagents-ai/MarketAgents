import asyncio
from dotenv import load_dotenv
from market_agents.economics.econ_agent import ZiFactory, ZiParams
from market_agents.economics.econ_models import Good, Endowment, Basket
from market_agents.economics.scenario import Scenario

from market_agents.market_orchestrator import MarketOrchestrator, MarketOrchestratorState
import os


async def zi_exploration(buyer_params: ZiParams, seller_params: ZiParams,max_rounds: int = 1,num_agents: int = 25):
    load_dotenv()
        # Create a good
    apple = Good(name="apple", quantity=0)
    # Create DoubleAuction mechanism
    factories = [
        ZiFactory(
            id=f"factory_episode_{0}",
            goods=["apple"],
            num_buyers=num_agents,  # Increase buyers by 1 each episode
            num_sellers=num_agents,     # Keep sellers constant
            buyer_params=buyer_params,
            seller_params=seller_params
        )
    ]
    scenario = Scenario(
        name="Static Apple Market",
        goods=["apple"],
        factories=factories
    )

    orchestrator = MarketOrchestrator(llm_agents=[], goods=[apple.name], max_rounds=max_rounds,scenario=scenario)

    # Run the market simulation
    await orchestrator.run_scenario()
    #plot the market results
    return scenario,orchestrator.state

if __name__ == "__main__":
    
    buyer_params = ZiParams(
        id="buyer_template",
        initial_cash=10000.0,
        initial_goods={"apple": 0},
        base_values={"apple": 20.0},
        num_units=20,
        noise_factor=0.05,
        max_relative_spread=0.2,
        is_buyer=True
    )

    seller_params = ZiParams(
        id="seller_template",
        initial_cash=0,
        initial_goods={"apple": 20},
        base_values={"apple": 10.0},
        num_units=20,
        noise_factor=0.05,
        max_relative_spread=0.2,
        is_buyer=False
    )
    scenario,state = asyncio.run(zi_exploration(buyer_params, seller_params,max_rounds=10))
    file_path_state = state.save_to_json(r"C:\Users\Tommaso\Documents\Dev\MarketAgents\outputs\econ_results")
    loaded_state = MarketOrchestratorState.load_from_json(file_path_state)
    file_path_scenario = scenario.save_to_json(r"C:\Users\Tommaso\Documents\Dev\MarketAgents\outputs\econ_results")
    loaded_scenario = Scenario.load_from_json(file_path_scenario)

    if state == loaded_state:
        print("The original state and the loaded state are the same.")
    else:
        print("Differences found:")
        from deepdiff import DeepDiff
        diff = DeepDiff(state.model_dump(), loaded_state.model_dump(), significant_digits=5, ignore_order=True)
        print(diff)
    if scenario == loaded_scenario:
        print("The original scenario and the loaded scenario are the same.")
    else:
        print("Differences found:")
        diff = DeepDiff(scenario.model_dump(), loaded_scenario.model_dump(), significant_digits=5, ignore_order=True)
        print(diff)
