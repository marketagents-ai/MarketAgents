import asyncio
import os
from dotenv import load_dotenv
from market_agents.economics.econ_agent import ZiFactory, ZiParams
from market_agents.economics.econ_models import Good, Endowment, Basket
from market_agents.economics.scenario import Scenario
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from market_agents.market_orchestrator import MarketOrchestrator, MarketOrchestratorState, zi_scenario
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
        return super().default(obj)

class SimulationConfig(BaseModel):
    num_units: int
    noise_factor: float
    max_relative_spread: float
    seller_base_value: float
    num_replicas: int
    max_rounds: int
    num_agents: int
    cost_spread_list: List[float]

class AgentParams(BaseModel):
    seller_params: ZiParams
    buyer_params: ZiParams

class SimulationResult(BaseModel):
    state: MarketOrchestratorState
    execution_time: float

class ExperimentResults(BaseModel):
    results: Dict[float, List[SimulationResult]]
    config: SimulationConfig
    variable_name: str
    variable_value: Any

def cost_spread_experiment(config: SimulationConfig) -> Dict[float, List[AgentParams]]:
    result = {}
    
    for cost_spread in config.cost_spread_list:
        replica_params = []
        for _ in range(config.num_replicas):
            seller_params = ZiParams(
                id="seller_template",
                initial_cash=0,
                initial_goods={"apple": config.num_units},
                base_values={"apple": config.seller_base_value},
                num_units=config.num_units,
                noise_factor=config.noise_factor,
                max_relative_spread=config.max_relative_spread,
                is_buyer=False
            )
            
            buyer_params = ZiParams(
                id="buyer_template",
                initial_cash=10000.0,  # Assuming a large initial cash value
                initial_goods={"apple": 0},
                base_values={"apple": config.seller_base_value + cost_spread},
                num_units=config.num_units,
                noise_factor=config.noise_factor,
                max_relative_spread=config.max_relative_spread,
                is_buyer=True
            )
            
            replica_params.append(AgentParams(seller_params=seller_params, buyer_params=buyer_params))
        
        result[cost_spread] = replica_params
    
    return result

def run_simulations(config: SimulationConfig) -> Dict[float, List[SimulationResult]]:
    results = {}
    for cost_spread in config.cost_spread_list:
        print(f"Cost spread: {cost_spread}")
        results[cost_spread] = []
        for i in range(config.num_replicas):
            print(f"  Replica {i + 1}:")
            start_time = time.time()
            scenario, state = asyncio.run(zi_scenario(
                buyer_params=ZiParams(
                    id="buyer",
                    initial_cash=1000,
                    initial_goods={},
                    base_values={"apple": config.seller_base_value + cost_spread / 2},
                    num_units=config.num_units,
                    noise_factor=config.noise_factor,
                    max_relative_spread=config.max_relative_spread,
                    is_buyer=True
                ),
                seller_params=ZiParams(
                    id="seller",
                    initial_cash=0,
                    initial_goods={"apple": config.num_units},
                    base_values={"apple": config.seller_base_value - cost_spread / 2},
                    num_units=config.num_units,
                    noise_factor=config.noise_factor,
                    max_relative_spread=config.max_relative_spread,
                    is_buyer=False
                ),
                max_rounds=config.max_rounds,
                num_buyers=config.num_agents,
                num_sellers=config.num_agents
            ))
            execution_time = time.time() - start_time
            results[cost_spread].append(SimulationResult(state=state, execution_time=execution_time))
    return results

def get_experiment_base_dir(variable_name: str) -> str:
    return f"outputs/zi_results/{variable_name}_experiment"

def get_sub_experiment_dir(base_dir: str, value: Any) -> str:
    return os.path.join(base_dir, f"{value}")

def datetime_to_str(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def save_dict_results(experiment_results: ExperimentResults, variable_name: str, value: Any):
    base_dir = get_experiment_base_dir(variable_name)
    sub_dir = get_sub_experiment_dir(base_dir, value)
    os.makedirs(sub_dir, exist_ok=True)
    file_path = os.path.join(sub_dir, "results.json")
    try:
        with open(file_path, 'w') as f:
            json.dump(experiment_results.model_dump(), f, indent=2, default=datetime_to_str)
        print(f"Results saved to {file_path}")
    except Exception as e:
        print(f"Error saving results to {file_path}: {str(e)}")

def load_dict_results(variable_name: str, value: Any) -> Optional[ExperimentResults]:
    base_dir = get_experiment_base_dir(variable_name)
    sub_dir = get_sub_experiment_dir(base_dir, value)
    file_path = os.path.join(sub_dir, "results.json")
    if not os.path.exists(file_path):
        print(f"No saved results found for experiment: {variable_name}, value: {value}")
        return None
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    return ExperimentResults.model_validate(data)

def create_boxplots(dict_results: Dict[float, List[MarketOrchestratorState]], config: SimulationConfig, show_plot: bool = True) -> pd.DataFrame:
    data = []
    for cost_spread, episodes in dict_results.items():
        for episode in episodes:
            summary = episode.summary[0]  # Accessing the first (and only) episode summary
            data.append({
                'cost_spread': cost_spread,
                'surplus': summary.surplus,
                'efficiency': summary.efficiency,
                'quantity': summary.quantity,
                'price': summary.average_price
            })
    
    df = pd.DataFrame(data)
    
    if show_plot:
        # Set up the plot style
        sns.set_style("whitegrid")
        sns.set_palette("Set2")
        
        # Create subplots
        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        
        metrics = ['surplus', 'efficiency', 'quantity', 'price']
        titles = ['Surplus', 'Efficiency', 'Quantity', 'Average Price']
        
        for i, (ax, metric, title) in enumerate(zip(axs.flatten(), metrics, titles)):
            # Create boxplot with individual points
            sns.boxplot(x='cost_spread', y=metric, data=df, ax=ax)
            sns.stripplot(x='cost_spread', y=metric, data=df, ax=ax, color='black', alpha=0.5, jitter=True)
            
            # Customize the plot
            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Cost Spread', fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            
            # Rotate x-axis labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()
    
    return df

def create_dataframe(experiment_results: ExperimentResults) -> pd.DataFrame:
    data = []
    for cost_spread, sim_results in experiment_results.results.items():
        for sim_result in sim_results:
            for episode, summary in sim_result.state.summary.items():
                data.append({
                    'cost_spread': cost_spread,
                    experiment_results.variable_name: experiment_results.variable_value,
                    'efficiency': summary.efficiency,
                    'quantity': summary.quantity,
                    'price': summary.average_price,
                    'time': sim_result.execution_time,
                    'episode': episode
                })
    
    return pd.DataFrame(data)

def run_experiment(config: SimulationConfig, variable_name: str, value: Any, show_plot: bool = True, use_cached_values: bool = True) -> ExperimentResults:
    if use_cached_values:
        cached_results = load_dict_results(variable_name, value)
        if cached_results:
            print(f"Using cached results for {variable_name}: {value}")
            return cached_results
    
    print(f"Running new experiment for {variable_name}: {value}")
    results = run_simulations(config)
    experiment_results = ExperimentResults(results=results, config=config, variable_name=variable_name, variable_value=value)
    save_dict_results(experiment_results, variable_name, value)
    return experiment_results

def run_variable_experiment(base_config: SimulationConfig, variable_name: str, variable_values: List[Any], use_cached_values: bool = True):
    all_results = []
    for value in variable_values:
        print(f"Running experiment with {variable_name}: {value}")
        config = base_config.copy(update={variable_name: value})
        experiment_results = run_experiment(config, variable_name, value, show_plot=False, use_cached_values=use_cached_values)
        df = create_dataframe(experiment_results)
        all_results.append(df)
    return pd.concat(all_results, ignore_index=True)

def create_combined_boxplots(df: pd.DataFrame, variable_name: str, config: SimulationConfig):
    base_dir = get_experiment_base_dir(variable_name)
    os.makedirs(base_dir, exist_ok=True)
    
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle(f"Impact of {variable_name} on Market Outcomes", fontsize=20)

    # Create subtitle with static variables
    static_vars = [
        f"Units: {config.num_units}",
        f"Noise: {config.noise_factor}",
        f"Base Value: {config.seller_base_value}",
        f"Replicas: {config.num_replicas}",
        f"Rounds: {config.max_rounds}",
        f"Agents: {config.num_agents}"
    ]
    subtitle = ", ".join(static_vars)
    fig.text(0.5, 0.95, subtitle, ha='center', va='center', fontsize=12)

    metrics = ['efficiency', 'quantity', 'price', 'time']
    for i, metric in enumerate(metrics):
        ax = axs[i // 2, i % 2]
        sns.boxplot(x='cost_spread', y=metric, hue=variable_name, data=df, ax=ax)
        ax.set_title(f"{metric.capitalize()} vs Cost Spread")
        ax.set_xlabel("Cost Spread")
        ax.set_ylabel(metric.capitalize())
        if metric == 'time':
            ax.set_ylabel("Execution Time (seconds)")

    plt.tight_layout(rect=(0, 0, 1, 0.96))  # Adjust layout to accommodate subtitle
    plot_path = os.path.join(base_dir, f"{variable_name}_impact.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")

def create_combined_boxplots_test(base_config: SimulationConfig, variable_name: str, variable_values: List[Any]):
    print(f"Testing combined boxplots for {variable_name}")
    
    # Run fresh simulations
    fresh_results = run_variable_experiment(base_config, variable_name, variable_values, use_cached_values=True)
    
    # Run with cached results
    cached_results = run_variable_experiment(base_config, variable_name, variable_values, use_cached_values=True)
    
    metrics = ['surplus', 'efficiency', 'quantity', 'price']
    
    for metric in metrics:
        print(f"\nChecking {metric}:")
        for value in variable_values:
            fresh_data = fresh_results[fresh_results[variable_name] == value][metric]
            cached_data = cached_results[cached_results[variable_name] == value][metric]
            
            print(f"  {variable_name} = {value}:")
            print(f"    Fresh data points: {len(fresh_data)}")
            print(f"    Cached data points: {len(cached_data)}")
            
            if len(fresh_data) != len(cached_data):
                print("    WARNING: Number of data points differs!")
                print(f"    Fresh data: {fresh_data.tolist()}")
                print(f"    Cached data: {cached_data.tolist()}")
            else:
                if not np.allclose(fresh_data, cached_data, equal_nan=True):
                    print("    WARNING: Data values differ!")
                    print(f"    Fresh data: {fresh_data.tolist()}")
                    print(f"    Cached data: {cached_data.tolist()}")
                else:
                    print("    Data values match.")
    
    # Create plots for visual comparison
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle(f"Comparison of Fresh vs Cached Results for {variable_name}", fontsize=16)
    
    for i, metric in enumerate(metrics):
        ax = axs[i // 2, i % 2]
        sns.boxplot(x=variable_name, y=metric, data=fresh_results, ax=ax, color='lightblue', label='Fresh')
        sns.boxplot(x=variable_name, y=metric, data=cached_results, ax=ax, color='lightgreen', label='Cached')
        ax.set_title(metric)
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def compare_dict_results(original: Dict[float, List[MarketOrchestratorState]], 
                         loaded: Dict[float, List[MarketOrchestratorState]]) -> bool:
    if original.keys() != loaded.keys():
        print("Cost spreads don't match")
        print(f"Original cost spreads: {original.keys()}")
        print(f"Loaded cost spreads: {loaded.keys()}")
        return False
    
    for cost_spread in original.keys():
        print(f"Comparing cost_spread {cost_spread}")
        print(f"  Original states: {len(original[cost_spread])}")
        print(f"  Loaded states: {len(loaded[cost_spread])}")
        if len(original[cost_spread]) != len(loaded[cost_spread]):
            print(f"Number of states for cost_spread {cost_spread} doesn't match")
            return False
        
        for i, (orig_state, loaded_state) in enumerate(zip(original[cost_spread], loaded[cost_spread])):
            if orig_state.model_dump() != loaded_state.model_dump():
                print(f"State mismatch for cost_spread {cost_spread}, state {i}")
                print("Original state:")
                print(json.dumps(orig_state.model_dump(), indent=2))
                print("Loaded state:")
                print(json.dumps(loaded_state.model_dump(), indent=2))
                return False
    
    return True


    return df

if __name__ == "__main__":
    cost_spread_list = list(range(10, 51, 10))+[75,100]
    cost_spread_list = [float(x) for x in cost_spread_list]
    base_config = SimulationConfig(
        num_units=10,
        noise_factor=0.05,
        max_relative_spread=0.05,
        seller_base_value=50.0,
        num_replicas=100,
        max_rounds=10,
        num_agents=25,
        cost_spread_list=cost_spread_list
    )

    use_cached_values = True  # Set this to False if you want to rerun all experiments

    # Run experiments (this will use cached values if available and use_cached_values is True)
    print("Running experiments...")
    combined_results_max_relative_spread = run_variable_experiment(base_config, "max_relative_spread", [0.05, 0.1, 0.2, 0.3, 0.4, 0.5], use_cached_values)
    combined_results_noise_factor = run_variable_experiment(base_config, "noise_factor", [0.01, 0.05, 0.1, 0.2, 0.35, 0.5], use_cached_values)
    combined_results_num_agents = run_variable_experiment(base_config, "num_agents", [1,2,5, 10, 25, 50, 100], use_cached_values)
    combined_results_max_rounds = run_variable_experiment(base_config, "max_rounds", [1, 2, 3, 4,5,6,7,8,9, 10], use_cached_values)

    # Create plots using the combined results
    print("Creating plots...")
    create_combined_boxplots(combined_results_max_relative_spread, "max_relative_spread", base_config)
    create_combined_boxplots(combined_results_noise_factor, "noise_factor", base_config)
    create_combined_boxplots(combined_results_num_agents, "num_agents", base_config)

