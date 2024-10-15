import asyncio
from dotenv import load_dotenv
from market_agents.economics.econ_agent import ZiFactory, ZiParams
from market_agents.economics.econ_models import Good, Endowment, Basket
from market_agents.economics.scenario import Scenario

from market_agents.market_orchestrator import MarketOrchestrator, MarketOrchestratorState, zi_scenario
import os




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
    scenario,state = asyncio.run(zi_scenario(buyer_params, seller_params,max_rounds=10,num_buyers=25,num_sellers=25))
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
