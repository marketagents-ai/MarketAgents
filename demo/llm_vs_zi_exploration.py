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
    match_buyers_sellers: bool
class SimulationResult(BaseModel):
    state: MarketOrchestratorState
    execution_time: float
    price_delta: Optional[float] = None

class TwoWaySimulationResult(BaseModel):
    zi_result: SimulationResult
    llm_result: SimulationResult

class ExperimentResults(BaseModel):
    results: Dict[str, List[TwoWaySimulationResult]]
    config: SimulationConfig
    primary_variable: str
    llm_config: LLMConfig

class OneWayLLMExperiment(BaseModel):
    primary_variable: str
    primary_values: List[Any]
    llm_clone_config: LLMConfig

async def run_single_simulation(config: SimulationConfig, experiment: OneWayLLMExperiment, primary_value: Any, replica: int, parallel_inference: ParallelAIUtilities) -> Tuple[str, TwoWaySimulationResult]:
    key = f"{primary_value}"
    print(f"{experiment.primary_variable}: {primary_value}, LLM: {experiment.llm_clone_config.model}, Replica {replica + 1}")

    current_config = config.model_copy()
    setattr(current_config, experiment.primary_variable, primary_value)

    zi_start_time = time.time()
    num_buyers = max(current_config.num_buyers, current_config.num_sellers) if current_config.match_buyers_sellers else current_config.num_buyers
    num_sellers = max(current_config.num_buyers, current_config.num_sellers) if current_config.match_buyers_sellers else current_config.num_sellers
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
        
        num_buyers=num_buyers,
        num_sellers=num_sellers
    )
    zi_execution_time = time.time() - zi_start_time

    zi_last_episode = max(zi_state.equilibrium_price.keys())
    zi_equilibrium_price = zi_state.equilibrium_price[zi_last_episode]
    zi_average_price = zi_state.average_price[zi_last_episode]
    zi_price_delta = zi_average_price - zi_equilibrium_price if zi_average_price > 0 else None

    llm_start_time = time.time()
    llm_orchestrator = await run_llm_matched_scenario(zi_scenario, clones_config=experiment.llm_clone_config, max_rounds=current_config.max_rounds, ai_utils=parallel_inference)
    print(f"LLM orchestrator created!!!")
    llm_execution_time = time.time() - llm_start_time
    llm_state = llm_orchestrator.state

    llm_last_episode = max(llm_state.equilibrium_price.keys())
    llm_equilibrium_price = llm_state.equilibrium_price[llm_last_episode]   
    llm_average_price = llm_state.average_price[llm_last_episode]
    llm_price_delta = llm_average_price - llm_equilibrium_price if llm_average_price > 0 else None

    zi_result = SimulationResult(state=zi_state, execution_time=zi_execution_time, price_delta=zi_price_delta)
    llm_result = SimulationResult(state=llm_state, execution_time=llm_execution_time, price_delta=llm_price_delta)

    return key, TwoWaySimulationResult(zi_result=zi_result, llm_result=llm_result)

async def run_simulations(config: SimulationConfig, experiment: OneWayLLMExperiment, parallel_inference: ParallelAIUtilities) -> Dict[str, List[TwoWaySimulationResult]]:
    tasks = []
    results = []
    for primary_value in experiment.primary_values:
        for i in range(config.num_replicas):
            print(f"CURRENT EXPERIMENT: \n {experiment.primary_variable}: {primary_value}, running replica {i + 1}")
            results.append(await run_single_simulation(config, experiment, primary_value, i, parallel_inference))
    
    
    
    grouped_results = {}
    for key, result in results:
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(result)
    
    return grouped_results

def create_dataframe(experiment_results: ExperimentResults) -> pd.DataFrame:
    data = []
    for key, sim_results in experiment_results.results.items():
        primary_value = float(key)
        for sim_result in sim_results:
            zi_last_episode = max(sim_result.zi_result.state.summary.keys())
            zi_summary = sim_result.zi_result.state.summary[zi_last_episode]
            data.append({
                'agent_type': 'ZI',
                'primary_variable_name': experiment_results.primary_variable,
                'primary_variable_value': primary_value,
                'efficiency': zi_summary.efficiency,
                'quantity': zi_summary.quantity,
                'price_delta': sim_result.zi_result.price_delta,
                'execution_time': sim_result.zi_result.execution_time,
                'episode': zi_last_episode
            })
            
            llm_last_episode = max(sim_result.llm_result.state.summary.keys())
            llm_summary = sim_result.llm_result.state.summary[llm_last_episode]
            data.append({
                'agent_type': experiment_results.llm_config.model,
                'primary_variable_name': experiment_results.primary_variable,
                'primary_variable_value': primary_value,
                'efficiency': llm_summary.efficiency,
                'quantity': llm_summary.quantity,
                'price_delta': sim_result.llm_result.price_delta,
                'execution_time': sim_result.llm_result.execution_time,
                'episode': llm_last_episode
            })
    
    df = pd.DataFrame(data)
    df['primary_variable_value'] = pd.to_numeric(df['primary_variable_value'])
    return df

def create_combined_boxplots(df: pd.DataFrame, experiment: OneWayLLMExperiment, config: SimulationConfig):
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle(f"Impact of {experiment.primary_variable} on Market Outcomes: {experiment.llm_clone_config.model} vs ZI", fontsize=20)

    # Create subtitle with hyperparameters
    subtitle = ", ".join([
        f"Units: {config.num_units}",
        f"Noise: {config.noise_factor}",
        f"Base Value: {config.seller_base_value}",
        f"Replicas: {config.num_replicas}",
        f"Rounds: {config.max_rounds}",
        f"Buyers: {config.num_buyers}",
        f"Sellers: {config.num_sellers}",
        f"Max Spread: {config.max_relative_spread}",
        f"Cost Spread: {config.cost_spread}"
    ])
    fig.text(0.5, 0.95, subtitle, ha='center', va='center', fontsize=12)

    metrics = ['efficiency', 'quantity', 'price_delta', 'execution_time']
    for i, metric in enumerate(metrics):
        ax = axs[i // 2, i % 2]
        
        sns.boxplot(x='primary_variable_value', y=metric, hue='agent_type', data=df, ax=ax)
        ax.set_title(f"{metric.capitalize().replace('_', ' ')} vs {experiment.primary_variable}")
        ax.set_xlabel(experiment.primary_variable)
        ax.set_ylabel(metric.capitalize().replace('_', ' '))
        if metric == 'execution_time':
            ax.set_ylabel("Execution Time (seconds)")
        elif metric == 'price_delta':
            ax.set_ylabel("Price Delta (relative to equilibrium)")

    plt.tight_layout(rect=(0, 0, 1, 0.93))  # Adjust the top margin to accommodate the subtitle
    return fig

async def run_experiment(config: SimulationConfig, experiment: OneWayLLMExperiment, parallel_inference: ParallelAIUtilities) -> ExperimentResults:
    results = await run_simulations(config, experiment, parallel_inference)
    return ExperimentResults(
        results=results,
        config=config,
        primary_variable=experiment.primary_variable,
        llm_config=experiment.llm_clone_config
    )

def get_experiment_base_dir(config: SimulationConfig, experiment: OneWayLLMExperiment) -> str:
    static_vars = [
        f"units_{config.num_units}",
        f"noise_{config.noise_factor}",
        f"base_value_{config.seller_base_value}",
        f"replicas_{config.num_replicas}",
        f"rounds_{config.max_rounds}",
        f"buyers_{config.num_buyers}",
        f"sellers_{config.num_sellers}",
        f"max_spread_{config.max_relative_spread}",
        f"cost_spread_{config.cost_spread}"
    ]
    experiment_vars = [
        f"{experiment.primary_variable}_{min(experiment.primary_values)}-{max(experiment.primary_values)}-{len(experiment.primary_values)}",
        f"llm_{experiment.llm_clone_config.model}"
    ]
    folder_name = "_".join(static_vars + experiment_vars)
    return os.path.join("outputs", "llm_zi_results", folder_name)

def save_results(experiment_results: ExperimentResults, config: SimulationConfig, experiment: OneWayLLMExperiment):
    base_dir = get_experiment_base_dir(config, experiment)
    os.makedirs(base_dir, exist_ok=True)
    
    results_path = os.path.join(base_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(experiment_results.model_dump(), f, cls=DateTimeEncoder, indent=2)
    
    df = create_dataframe(experiment_results)
    fig = create_combined_boxplots(df, experiment, config)
    plot_path = os.path.join(base_dir, f"{experiment.primary_variable}_comparison.png")
    fig.savefig(plot_path)
    plt.close(fig)
    
    print(f"Results saved to {base_dir}")

async def main():
    load_dotenv()

    base_config = SimulationConfig(
        num_units=20,
        noise_factor=0.1,
        max_relative_spread=0.2,
        seller_base_value=50.0,
        num_replicas=10,
        max_rounds=10,
        num_buyers=25,
        num_sellers=25,
        cost_spread=25,
        match_buyers_sellers=True
    )
    clones_config = LLMConfig(
        model="gpt-4o-mini",
        temperature=0.0,
        client="openai",
        response_format="tool",
        max_tokens=250
    )  
    experiments = [
        OneWayLLMExperiment(
            primary_variable="max_rounds",
            primary_values=[1,2,3,4,5,8,10],
            llm_clone_config=clones_config
        )
    ]

    # Initialize ParallelAIUtilities
    parallel_inference = ParallelAIUtilities(
        oai_request_limits=RequestLimits(
            max_requests_per_minute=2000,
            max_tokens_per_minute=1000000,
           
        ),
        vllm_request_limits=RequestLimits(
            max_requests_per_minute=2000,
            max_tokens_per_minute=1000000,
        )
    )

    for experiment in experiments:
        results = await run_experiment(base_config, experiment, parallel_inference)
        save_results(results, base_config, experiment)

if __name__ == "__main__":
    asyncio.run(main())