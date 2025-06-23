import csv
import os
import json
import argparse
from typing import Any, Dict, List
import networkx as nx
from datetime import datetime

from itertools import islice
import concurrent.futures

import importlib
from src.tasks import load_tasks
from src.resources import Resource
from src.agents import Agent
from src.utils import setup_logger, get_source_dir, create_log_directories

from matplotlib import pyplot as plt

class AgentOrchestrator:
    def __init__(
            self,
            generation_type: str, 
            agent_config: str = "default.json",
            verbose: bool = False, 
            log_file: str = "orchestrator_log.log",
            local_embeddings: bool = False
        ):
        self.generation_type = generation_type
        self.agent_logger = setup_logger(log_file)
        self.agent_config = agent_config
        self.verbose = verbose 
        self.log_file = log_file
        self.log_data = []
        self.llama_logs = []
        self.local_embeddings = local_embeddings

    def run(
            self, 
            task: Dict[str, Any],
            
        ) -> str:

        agents_metadata = self.load_or_generate_graph(agents_file=self.agent_config)

        G = nx.DiGraph()
        for agent_data in agents_metadata:
            agent_role = agent_data["name"]
            G.add_node(agent_role, **agent_data)

        # Add edges to the graph based on the dependencies
        for agent_data in agents_metadata:
            agent_role = agent_data["name"]
            dependencies = agent_data.get("dependencies", [])
            for dependency in dependencies:
                G.add_edge(dependency, agent_role)

        # Create a dictionary to store the output of each agent
        agent_outputs = {}

        # Execute agents in topological order (respecting dependencies)
        for agent_role in nx.topological_sort(G):
            agent_data = G.nodes[agent_role]
            agent = Agent(task=task, local_embeddings=self.local_embeddings, generation_type=self.generation_type, **agent_data)

            self.agent_logger.info(f"Starting Agent: {agent.name}")
            if agent.verbose:
                self.agent_logger.info(f"Agent Task: {agent.task}")

            # Prepare the input messages for the agent
            input_messages = []
            for predecessor in G.predecessors(agent_role):
                if predecessor in agent_outputs:
                    input_messages.append({"role": predecessor, "content": agent_outputs[predecessor]})

            agent.input_messages = input_messages

            # Execute the agent
            output = agent.execute()

            if agent.verbose:
                self.agent_logger.info(f"Agent Output:\n{output}")

            agent_outputs[agent_role] = output
            self.llama_logs.extend(agent.interactions)

        # Collect the final output from all the agents
        final_output = "\n".join([f"Agent: {name}\nOutput:\n{output}\n" for name, output in agent_outputs.items()])

        self.save_llama_logs()

        return final_output

    
    def load_or_generate_graph(
            self,
            config_dir: str = "configs",
            agents_dir: str = "agents",
            agents_file: str = "default.json",
        ):
        # Construct the path to the curriculum CSV file
        data_genie_agents_path = get_source_dir()
        agents_config_dir = os.path.join(config_dir, agents_dir)
        agent_metadata_file = os.path.join(agents_config_dir, agents_file)
        total_path = os.path.join(data_genie_agents_path, agent_metadata_file)
        print(f"Total path: {total_path}")

        if os.path.exists(total_path):
            print("Loading agents metadata from file...")

            with open(total_path, "r") as file:
                agents_metadata = json.load(file)

        return agents_metadata

    def save_llama_logs(self):
        log_path = "logs"
        qa_interactions_path = os.path.join(log_path, "qa_interactions")
        qa_interaction_path = os.path.join(qa_interactions_path, "qa_interactions" + datetime.now().strftime("%Y%m%d%H%M%S") + ".json")
        with open(qa_interaction_path, "w") as file:
            json.dump(self.llama_logs, file, indent=2)

def parse_args():
    parser = argparse.ArgumentParser(description="Run the agent orchestrator with dynamic configurations.")
    #parser.add_argument('-q', '--query', type=str, help="user query for agents to assist with", required=True)
    parser.add_argument('--generation_type', type=str, help="type of data generation", required=True)
    parser.add_argument('--num_tasks', type=int, help="number of tasks to generate", required=True)
    parser.add_argument('--agent_config', type=str, help="agent configuration file", required=True)
    parser.add_argument('--local_embeddings', type=bool, help="use local embeddings via Ollama", default=False)
    return parser.parse_args()

def mainflow(generation_type: str, num_tasks: int, agent_config: str, local_embeddings: bool = False):
    """Main flow of the agent orchestrator for data generation."""

    # load tasks from the curriculum CSV file
    tasks = load_tasks(generation_type, num_tasks)
    
    # Run the orchestrator for each task
    log_path = "logs" 
    orchestrator_log_path = os.path.join(log_path, "orchestrator_logs")
    create_log_directories()
    
    orchestrator = AgentOrchestrator(
        generation_type=generation_type,
        agent_config=agent_config,
        verbose=True,
        log_file=os.path.join(orchestrator_log_path, "orchestrator_log" + datetime.now().strftime("%Y%m%d%H%M%S") + ".log"),
        local_embeddings=local_embeddings
    )
    
    for task in tasks:
        orchestrator.run(task)

   #with concurrent.futures.ThreadPoolExecutor() as executor:
   #     executor.map(orchestrator.run, tasks)
    
if __name__ == "__main__":
    args = parse_args()
    mainflow(args.generation_type, args.num_tasks, args.agent_config, args.local_embeddings)
