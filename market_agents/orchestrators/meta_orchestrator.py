# meta_orchestrator.py

import asyncio
import logging
import os
import random
import uuid
from pathlib import Path
from typing import List, Dict, Union
import warnings

import yaml
from auction_orchestrator import AuctionOrchestrator
from groupchat_orchestrator import GroupChatOrchestrator
from market_agents.agents.market_agent import MarketAgent
from market_agents.agents.personas.persona import Persona, generate_persona, save_persona_to_file
from market_agents.agents.protocols.acl_message import ACLMessage
from market_agents.economics.econ_agent import EconomicAgent
from market_agents.economics.econ_models import (
    Ask,
    Basket,
    Bid,
    BuyerPreferenceSchedule,
    Endowment,
    Good,
    SellerPreferenceSchedule,
)
from market_agents.inference.parallel_inference import ParallelAIUtilities, RequestLimits
from market_agents.orchestrators.base_orchestrator import BaseEnvironmentOrchestrator
from market_agents.orchestrators.config import OrchestratorConfig, load_config
from market_agents.orchestrators.insert_simulation_data import SimulationDataInserter
from market_agents.orchestrators.logger_utils import (
    log_section,
    log_environment_setup,
    log_agent_init,
    log_round,
    print_ascii_art,
    orchestartion_logger,
    log_completion
)

warnings.filterwarnings("ignore", module="pydantic")

class MetaOrchestrator:
    def __init__(self, config: OrchestratorConfig, environment_order: List[str] = None):
        self.config = config
        self.agents: List[MarketAgent] = []
        self.ai_utils = self._initialize_ai_utils()
        self.data_inserter = self._initialize_data_inserter()
        self.logger = orchestartion_logger
        self.environment_order = environment_order or config.environment_order
        self.environment_orchestrators = {}

    def _initialize_ai_utils(self):
        # Initialize AI utilities
        oai_request_limits = RequestLimits(max_requests_per_minute=20000, max_tokens_per_minute=2000000)
        anthropic_request_limits = RequestLimits(max_requests_per_minute=20000, max_tokens_per_minute=2000000)
        ai_utils = ParallelAIUtilities(
            oai_request_limits=oai_request_limits,
            anthropic_request_limits=anthropic_request_limits
        )
        return ai_utils

    def _initialize_data_inserter(self):
        db_config = self.config.database_config
        db_params = {
            'dbname': db_config.db_name,
            'user': db_config.db_user,
            'password': db_config.db_password,
            'host': db_config.db_host,
            'port': db_config.db_port
        }
        data_inserter = SimulationDataInserter(db_params)
        return data_inserter

    def load_or_generate_personas(self) -> List[Persona]:
        personas_dir = Path("./market_agents/agents/personas/generated_personas")
        existing_personas = []

        if os.path.exists(personas_dir):
            for filename in os.listdir(personas_dir):
                if filename.endswith(".yaml"):
                    with open(os.path.join(personas_dir, filename), 'r') as file:
                        persona_data = yaml.safe_load(file)
                        existing_personas.append(Persona(**persona_data))

        while len(existing_personas) < self.config.num_agents:
            new_persona = generate_persona()
            existing_personas.append(new_persona)
            save_persona_to_file(new_persona, personas_dir)

        return existing_personas[:self.config.num_agents]

    def generate_agents(self):
        log_section(self.logger, "Generating Agents")
        personas = self.load_or_generate_personas()
        num_agents = len(personas)
        num_buyers = num_agents // 2
        num_sellers = num_agents - num_buyers

        for i, persona in enumerate(personas):
            agent_uuid = str(uuid.uuid4())
            # Randomly assign an LLM config if there are multiple configs
            llm_config = random.choice(self.config.llm_configs).dict() if len(self.config.llm_configs) > 1 else self.config.llm_configs[0].dict()
            # Assign roles explicitly based on index
            if i < num_buyers:
                is_buyer = True
                persona.role = "buyer"
            else:
                is_buyer = False
                persona.role = "seller"

            agent_config = self.config.agent_config.dict()
            if is_buyer:
                initial_cash = agent_config.get('buyer_initial_cash', 1000)
                initial_goods_quantity = agent_config.get('buyer_initial_goods', 0)
                base_value = agent_config.get('buyer_base_value', 120.0)
            else:
                initial_cash = agent_config.get('seller_initial_cash', 0)
                initial_goods_quantity = agent_config.get('seller_initial_goods', 10)
                base_value = agent_config.get('seller_base_value', 80.0)

            good_name = agent_config.get('good_name', 'apple')
            initial_goods = {good_name: initial_goods_quantity}

            # Create initial basket and endowment
            initial_basket = Basket(
                cash=initial_cash,
                goods=[Good(name=good_name, quantity=initial_goods_quantity)]
            )
            endowment = Endowment(
                initial_basket=initial_basket,
                agent_id=agent_uuid
            )

            # Create preference schedules
            if is_buyer:
                value_schedules = {
                    good_name: BuyerPreferenceSchedule(
                        num_units=agent_config.get('num_units', 10),
                        base_value=base_value,
                        noise_factor=agent_config.get('noise_factor', 0.05)
                    )
                }
                cost_schedules = {}
            else:
                value_schedules = {}
                cost_schedules = {
                    good_name: SellerPreferenceSchedule(
                        num_units=agent_config.get('num_units', 10),
                        base_value=base_value,
                        noise_factor=agent_config.get('noise_factor', 0.05)
                    )
                }

            economic_agent = EconomicAgent(
                id=agent_uuid,
                endowment=endowment,
                value_schedules=value_schedules,
                cost_schedules=cost_schedules,
                max_relative_spread=agent_config.get('max_relative_spread', 0.2)
            )

            agent = MarketAgent.create(
                agent_id=agent_uuid,
                use_llm=agent_config.get('use_llm', True),
                llm_config=llm_config,
                environments={},  # Environments will be assigned later
                protocol=ACLMessage,
                persona=persona,
                econ_agent=economic_agent
            )

            # Initialize last_perception and last_observation
            agent.last_perception = None
            agent.last_observation = None
            agent.last_step = None
            agent.index = i
            self.agents.append(agent)
            log_agent_init(self.logger, agent.index, is_buyer, persona)

    def _initialize_environment_orchestrators(self) -> Dict[str, BaseEnvironmentOrchestrator]:
        orchestrators = {}
        for env_name in self.environment_order:
            env_config = self.config.environment_configs.get(env_name)
            if not env_config:
                self.logger.warning(f"Configuration for environment '{env_name}' not found.")
                continue
            if env_name == 'auction':
                orchestrator = AuctionOrchestrator(
                    config=env_config,
                    orchestrator_config=self.config,
                    agents=self.agents,
                    ai_utils=self.ai_utils,
                    data_inserter=self.data_inserter,
                    logger=self.logger
                )
            elif env_name == 'group_chat':
                orchestrator = GroupChatOrchestrator(
                    config=env_config,
                    agents=self.agents,
                    ai_utils=self.ai_utils,
                    data_inserter=self.data_inserter,
                    logger=self.logger
                )
            else:
                self.logger.warning(f"Unknown environment: {env_name}")
                continue
            orchestrators[env_name] = orchestrator
        return orchestrators

    async def run_simulation(self):
        self.generate_agents()
        # Initialize environment orchestrators after agents are generated
        self.environment_orchestrators = self._initialize_environment_orchestrators()
        for orchestrator in self.environment_orchestrators.values():
            orchestrator.setup_environment()

        for round_num in range(1, self.config.max_rounds + 1):
            log_round(self.logger, round_num)
            for env_name in self.environment_order:
                orchestrator = self.environment_orchestrators.get(env_name)
                if orchestrator is None:
                    self.logger.warning(f"No orchestrator found for environment '{env_name}'. Skipping.")
                    continue
                log_environment_setup(self.logger, env_name)
                await orchestrator.run_environment(round_num)
                # Optionally process and store results
                await orchestrator.process_round_results(round_num)
                # Optionally, implement logic to generate new topics or adjust agent states between environments
            # If you have any per-round logic that depends on multiple environments, handle it here
        # Print summaries for each environment
        for orchestrator in self.environment_orchestrators.values():
            orchestrator.print_summary()

    async def start(self):
        print_ascii_art()
        log_section(self.logger, "Simulation Starting")
        await self.run_simulation()
        log_completion(self.logger, "Simulation completed successfully")

if __name__ == "__main__":
    import sys
    import argparse
    from pathlib import Path

    async def main():
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description='Run the market simulation.')
        parser.add_argument('--environments', nargs='+', help='List of environments to run (e.g., group_chat auction)')
        args = parser.parse_args()

        # Load configuration from orchestrator_config.yaml
        config_path = Path("market_agents/orchestrators/orchestrator_config.yaml")
        config = load_config(config_path=config_path)

        # If environments are specified in command-line arguments, use them
        if args.environments:
            environment_order = args.environments
        else:
            environment_order = config.environment_order

        # Initialize MetaOrchestrator with the loaded config and specified environments
        orchestrator = MetaOrchestrator(config, environment_order=environment_order)
        # Start the simulation
        await orchestrator.start()

    asyncio.run(main())