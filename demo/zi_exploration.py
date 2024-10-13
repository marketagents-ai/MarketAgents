import asyncio
import os
from dotenv import load_dotenv
from market_agents.economics.econ_agent import ZiFactory, ZiParams
from market_agents.economics.econ_models import Good, Endowment, Basket
from market_agents.economics.scenario import Scenario
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
from market_agents.market_orchestrator import MarketOrchestrator, MarketOrchestratorState, run_zi_scenario
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime

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

class ExperimentResults(BaseModel):
    results: Dict[str, List[SimulationResult]]
    config: SimulationConfig
    primary_variable: str
    secondary_variable: str

class TwoWayExperiment(BaseModel):
    primary_variable: str
    primary_values: List[Any]
    secondary_variable: str
    secondary_values: List[Any]

async def run_simulations(config: SimulationConfig, experiment: TwoWayExperiment) -> Dict[str, List[SimulationResult]]:
    tasks = []
    for primary_value in experiment.primary_values:
        for secondary_value in experiment.secondary_values:
            for i in range(config.num_replicas):
                tasks.append(run_single_simulation(config, experiment, primary_value, secondary_value, i))
    
    results = await asyncio.gather(*tasks)
    
    grouped_results = {}
    for key, result in results:
        if key not in grouped_results:
            grouped_results[key] = []
        grouped_results[key].append(result)
    
    return grouped_results

async def run_single_simulation(config: SimulationConfig, experiment: TwoWayExperiment, primary_value: Any, secondary_value: Any, replica: int) -> Tuple[str, SimulationResult]:
    key = f"{primary_value},{secondary_value}"
    print(f"{experiment.primary_variable}: {primary_value}, {experiment.secondary_variable}: {secondary_value}, Replica {replica + 1}")
    start_time = time.time()

    # Create a copy of the config and update it with the experiment values
    current_config = config.model_copy()
    setattr(current_config, experiment.primary_variable, primary_value)
    setattr(current_config, experiment.secondary_variable, secondary_value)

    scenario, state = await run_zi_scenario(
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
    execution_time = time.time() - start_time

    # Calculate price delta using equilibrium price
    last_episode = max(state.equilibrium_price.keys())
    equilibrium_price = state.equilibrium_price[last_episode]
    average_price = state.average_price[last_episode]
    price_delta = average_price - equilibrium_price if average_price > 0 else None

    return key, SimulationResult(state=state, execution_time=execution_time, price_delta=price_delta)

def get_experiment_base_dir(config: SimulationConfig, experiment: TwoWayExperiment) -> str:
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
        f"{experiment.secondary_variable}_{min(experiment.secondary_values)}-{max(experiment.secondary_values)}-{len(experiment.secondary_values)}"
    ]
    folder_name = "_".join(static_vars + experiment_vars)
    return os.path.join("outputs", "zi_results", folder_name)

def save_dict_results(experiment_results: ExperimentResults, config: SimulationConfig, experiment: TwoWayExperiment):
    base_dir = get_experiment_base_dir(config, experiment)
    os.makedirs(base_dir, exist_ok=True)
    file_path = os.path.join(base_dir, "results.json")
    try:
        serializable_results = experiment_results.model_dump()
        with open(file_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, cls=DateTimeEncoder)
        print(f"Results saved to {file_path}")
    except Exception as e:
        print(f"Error saving results to {file_path}: {str(e)}")

def load_dict_results(config: SimulationConfig, experiment: TwoWayExperiment) -> Optional[ExperimentResults]:
    base_dir = get_experiment_base_dir(config, experiment)
    file_path = os.path.join(base_dir, "results.json")
    if not os.path.exists(file_path):
        print(f"No saved results found for experiment: {experiment.primary_variable}, {experiment.secondary_variable}")
        return None
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return ExperimentResults.model_validate(data)

def create_dataframe(experiment_results: ExperimentResults) -> pd.DataFrame:
    data = []
    for key, sim_results in experiment_results.results.items():
        primary_value, secondary_value = map(float, key.split(','))
        for sim_result in sim_results:
            last_episode = max(sim_result.state.summary.keys())
            summary = sim_result.state.summary[last_episode]
            data.append({
                experiment_results.primary_variable: primary_value,
                experiment_results.secondary_variable: secondary_value,
                'efficiency': summary.efficiency,
                'quantity': summary.quantity,
                'price_delta': sim_result.price_delta,
                'time': sim_result.execution_time,
                'episode': last_episode
            })
    
    return pd.DataFrame(data)

async def run_experiment_async(config: SimulationConfig, experiment: TwoWayExperiment, use_cached_values: bool = True) -> ExperimentResults:
    if use_cached_values:
        cached_results = load_dict_results(config, experiment)
        if cached_results:
            print(f"Using cached results for {experiment.primary_variable} and {experiment.secondary_variable}")
            return cached_results
    
    print(f"Running new experiment for {experiment.primary_variable} and {experiment.secondary_variable}")
    results = await run_simulations(config, experiment)
    
    experiment_results = ExperimentResults(
        results=results, 
        config=config, 
        primary_variable=experiment.primary_variable, 
        secondary_variable=experiment.secondary_variable
    )
    save_dict_results(experiment_results, config, experiment)
    return experiment_results

async def run_variable_experiment(base_config: SimulationConfig, experiment: TwoWayExperiment, use_cached_values: bool = True) -> pd.DataFrame:
    print(f"Running experiment with {experiment.primary_variable} and {experiment.secondary_variable}")
    experiment_results = await run_experiment_async(base_config, experiment, use_cached_values=use_cached_values)
    return create_dataframe(experiment_results)

def create_combined_boxplots(df: pd.DataFrame, experiment: TwoWayExperiment, config: SimulationConfig):
    base_dir = get_experiment_base_dir(config, experiment)
    os.makedirs(base_dir, exist_ok=True)
    
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle(f"Impact of {experiment.primary_variable} and {experiment.secondary_variable} on Market Outcomes", fontsize=20)

    static_vars = [
        f"Units: {config.num_units}",
        f"Noise: {config.noise_factor}",
        f"Base Value: {config.seller_base_value}",
        f"Replicas: {config.num_replicas}",
        f"Rounds: {config.max_rounds}",
        f"Buyers: {config.num_buyers}",
        f"Sellers: {config.num_sellers}",
        f"Max Spread: {config.max_relative_spread}"
    ]
    subtitle = ", ".join(static_vars)
    fig.text(0.5, 0.95, subtitle, ha='center', va='center', fontsize=12)

    metrics = ['efficiency', 'quantity', 'price_delta', 'time']
    for i, metric in enumerate(metrics):
        ax = axs[i // 2, i % 2]
        
        # Filter out None values for the price_delta plot
        if metric == 'price_delta':
            df_filtered = df[df['price_delta'].notna()]
        else:
            df_filtered = df
        
        sns.boxplot(x=experiment.secondary_variable, y=metric, hue=experiment.primary_variable, data=df_filtered, ax=ax)
        ax.set_title(f"{metric.capitalize().replace('_', ' ')} vs {experiment.secondary_variable}")
        ax.set_xlabel(experiment.secondary_variable)
        ax.set_ylabel(metric.capitalize().replace('_', ' '))
        if metric == 'time':
            ax.set_ylabel("Execution Time (seconds)")
        elif metric == 'price_delta':
            ax.set_ylabel("Price Delta (relative to mid-price)")

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plot_path = os.path.join(base_dir, f"{experiment.primary_variable}_{experiment.secondary_variable}_impact.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")

async def main():
    base_config = SimulationConfig(
        num_units=20,
        noise_factor=0.1,
        max_relative_spread=0.1,
        seller_base_value=50.0,
        num_replicas=100,
        max_rounds=10,
        num_buyers=15,
        num_sellers=15,
        cost_spread=25
    )
    cost_spread_list = [15]#,25, 30.0, 40.0, 50.0]
    use_cached_values = True  # Set this to False if you want to rerun all experiments
    experiments = [
        # TwoWayExperiment(
        #     primary_variable="max_relative_spread",
        #     primary_values=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        #     secondary_variable="cost_spread",
        #     secondary_values=cost_spread_list
        # ),
        # TwoWayExperiment(
        #     primary_variable="noise_factor",
        #     primary_values=[0.01, 0.05, 0.1, 0.2, 0.35, 0.5],
        #     secondary_variable="cost_spread",
        #     secondary_values=cost_spread_list
        # ),
        # TwoWayExperiment(
        #     primary_variable="num_agents",
        #     primary_values=[1, 2, 5, 10, 25, 50, 100],
        #     secondary_variable="cost_spread",
        #     secondary_values=cost_spread_list
        # ),
        TwoWayExperiment(
            primary_variable="num_buyers",
            primary_values=[5, 10, 15, 25, 30],
            secondary_variable="max_rounds",
            secondary_values=[1, 2, 3, 4, 5,6,7,8,9,10]
        )
    ]

    # Run experiments
    print("Running experiments...")
    results = {}
    for experiment in experiments:
        results[experiment.primary_variable] = await run_variable_experiment(base_config, experiment, use_cached_values)

    # Create plots
    print("Creating plots...")
    for experiment in experiments:
        create_combined_boxplots(results[experiment.primary_variable], experiment, base_config)

if __name__ == "__main__":
    asyncio.run(main())
