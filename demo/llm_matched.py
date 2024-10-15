import asyncio
import os
from dotenv import load_dotenv
from market_agents.economics.econ_agent import ZiFactory, ZiParams
from market_agents.economics.econ_models import Good, Endowment, Basket
from market_agents.economics.scenario import Scenario
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
from market_agents.market_orchestrator import MarketOrchestrator, MarketOrchestratorState, run_zi_scenario, run_llm_matched_scenario
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from market_agents.inference.parallel_inference import ParallelAIUtilities, RequestLimits
from market_agents.inference.message_models import LLMConfig

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        return super().default(obj)

class SimulationConfig(BaseModel):
    num_units: int
    noise_factor: float
    max_relative_spread: float
    seller_base_value: float
    num_replicas: int
    max_rounds: int
    num_buyers: int
    num_sellers: int
    cost_spread: float

class SimulationResult(BaseModel):
    state: MarketOrchestratorState
    execution_time: float
    price_delta: Optional[float] = None
    
class TwoWaySimulationResult(BaseModel):
    zi_result: SimulationResult
    llm_result: SimulationResult

class ExperimentResults(BaseModel):
    results: Dict[str, List[SimulationResult]]
    config: SimulationConfig
    primary_variable: str
    secondary_variable: str

class OneWayLLMExperiment(BaseModel):
    primary_variable: str
    primary_values: List[Any]
    llm_clone_config: LLMConfig

async def run_single_simulation(config: SimulationConfig, experiment: OneWayLLMExperiment, primary_value: Any, replica: int) -> Tuple[str,  TwoWaySimulationResult]:
    key = f"{primary_value},{experiment.llm_clone_config}"
    print(f"{experiment.primary_variable}: {primary_value}, llm_config: {experiment.llm_clone_config}, Replica {replica + 1}")
    

    # Create a copy of the config and update it with the experiment values
    current_config = config.model_copy()
    setattr(current_config, experiment.primary_variable, primary_value)
    zi_start_time = time.time()
    zi_scenario, zi_state = await run_zi_scenario(
        buyer_params=ZiParams(
            id="buyer",
            initial_cash=1000,
            initial_goods={},
            base_values={"apple": current_config.seller_base_value + current_config.cost_spread / 2},
            num_units=current_config.num_units,
            noise_factor=current_config.noise_factor,
            max_relative_spread=current_config.max_relative_spread,
            is_buyer=True
        ),
        seller_params=ZiParams(
            id="seller",
            initial_cash=0,
            initial_goods={"apple": current_config.num_units},
            base_values={"apple": current_config.seller_base_value - current_config.cost_spread / 2},
            num_units=current_config.num_units,
            noise_factor=current_config.noise_factor,
            max_relative_spread=current_config.max_relative_spread,
            is_buyer=False
        ),
        max_rounds=current_config.max_rounds,
        num_buyers=current_config.num_buyers,
        num_sellers=current_config.num_sellers
    )
    zi_execution_time = time.time() - zi_start_time

    # Calculate price delta using equilibrium price
    zi_last_episode = max(zi_state.equilibrium_price.keys())
    zi_equilibrium_price = zi_state.equilibrium_price[zi_last_episode]
    zi_average_price = zi_state.average_price[zi_last_episode]
    zi_price_delta = zi_average_price - zi_equilibrium_price if zi_average_price > 0 else None

    llm_start_time = time.time()
    llm_orchestrator = await run_llm_matched_scenario(zi_scenario, clones_config=experiment.llm_clone_config)
    llm_execution_time = time.time() - llm_start_time
    llm_state = llm_orchestrator.state

    llm_last_episode = max(llm_state.equilibrium_price.keys())
    llm_equilibrium_price = llm_state.equilibrium_price[llm_last_episode]   
    llm_average_price = llm_state.average_price[llm_last_episode]
    llm_price_delta = llm_average_price - llm_equilibrium_price if llm_average_price > 0 else None
    zi_result = SimulationResult(state=zi_state, execution_time=zi_execution_time, price_delta=zi_price_delta)
    llm_result = SimulationResult(state=llm_state, execution_time=llm_execution_time, price_delta=llm_price_delta)

    return key, TwoWaySimulationResult(zi_result=zi_result, llm_result=llm_result)


async def run_simulations(config: SimulationConfig, experiment: OneWayLLMExperiment) -> Dict[str, List[SimulationResult]]:
    tasks = []
    for primary_value in experiment.primary_values:
        for i in range(config.num_replicas):
            tasks.append(run_single_simulation(config, experiment, primary_value, i))
    
    results = await asyncio.gather(*tasks)
    
    grouped_results = {}
    for key, result in results:
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(result)
    
    return grouped_results



if __name__ == "__main__":
    apple = Good(name="apple", quantity=0)
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
    num_buyers = 10
    num_sellers = 10
    # Create DoubleAuction mechanism
    factories = [
        ZiFactory(
            id=f"factory_episode_{0}",
            goods=["apple"],
            num_buyers=num_buyers,  # Increase buyers by 1 each episode
            num_sellers=num_sellers,     # Keep sellers constant
            buyer_params=buyer_params,
            seller_params=seller_params
        )
    ]
    scenario = Scenario(
        name="Static Apple Market",
        goods=["apple"],
        factories=factories
    )
    orchestrator = asyncio.run(run_llm_matched_scenario(scenario, max_rounds=10))
    state = orchestrator.state
