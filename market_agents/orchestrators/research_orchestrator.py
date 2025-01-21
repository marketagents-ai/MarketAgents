# research_orchestrator.py

import asyncio
import importlib
import logging
from typing import List, Dict, Any, Optional, Type
from datetime import datetime

from market_agents.orchestrators.logger_utils import log_action, log_perception, log_persona, log_reflection
from pydantic import BaseModel, Field

from market_agents.orchestrators.base_orchestrator import BaseEnvironmentOrchestrator
from market_agents.orchestrators.config import OrchestratorConfig, ResearchConfig
from market_agents.agents.market_agent import MarketAgent
from market_agents.inference.parallel_inference import ParallelAIUtilities
from market_agents.orchestrators.insert_simulation_data import SimulationDataInserter
from market_agents.orchestrators.agent_cognitive import AgentCognitiveProcessor
from market_agents.environments.environment import MultiAgentEnvironment, EnvironmentStep
from market_agents.environments.mechanisms.research import (
    ResearchEnvironment,
    ResearchGlobalAction,
    ResearchAction
)


class ResearchOrchestrator(BaseEnvironmentOrchestrator):
    """
    Orchestrator for the Research Environment.
    """
    config: ResearchConfig
    orchestrator_config: OrchestratorConfig
    environment: ResearchEnvironment = Field(default=None)
    cognitive_processor: Optional[AgentCognitiveProcessor] = None
    summary_model: Type[BaseModel] = Field(default=None)

    def __init__(
        self,
        config: ResearchConfig,
        agents: List[MarketAgent],
        ai_utils: ParallelAIUtilities,
        data_inserter: SimulationDataInserter,
        orchestrator_config: OrchestratorConfig,
        logger=None,
        **kwargs
    ):
        super().__init__(
            config=config,
            orchestrator_config=orchestrator_config,
            agents=agents,
            ai_utils=ai_utils,
            data_inserter=data_inserter,
            logger=logger,
            environment_name=config.name
        )
        
        self.orchestrator_config = orchestrator_config

        # Get schema model directly from config
        self.summary_model = self.get_schema_model(self.config.schema_model)

        # Initialize environment with the schema model
        self.environment = ResearchEnvironment(
            summary_model=self.summary_model,
            name=self.config.name,
            address=self.config.address,
            max_steps=self.config.max_rounds
        )

        # Attach environment to all agents
        for agent in self.agents:
            agent.environments[self.config.name] = self.environment

        # Initialize a cognitive processor for parallel agent perceptions/actions
        self.cognitive_processor = AgentCognitiveProcessor(
            ai_utils=self.ai_utils,
            data_inserter=self.data_inserter,
            logger=self.logger,
            tool_mode=self.orchestrator_config.tool_mode
        )

        self.logger.info(f"Initialized ResearchOrchestrator for environment: {self.config.name}")

    def get_schema_model(self, schema_name: str) -> Type[BaseModel]:
        try:
            # Import the research_schemas module
            schemas_module = importlib.import_module('market_agents.orchestrators.research_schemas')
            # Get the specified model class
            model_class = getattr(schemas_module, schema_name)
            return model_class
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Could not load schema model '{schema_name}': {e}")
        
    async def setup_environment(self):
        """
        Setup or reset the research environment.
        """
        self.logger.info("Setting up the Research Environment...")
        self.environment.reset()
        self.logger.info("Research Environment setup complete.")

    async def run_environment(self, round_num: int = None):
        """
        Run the environment for N rounds. If round_num is provided, run only that round.
        """
        if round_num is not None:
            await self.run_round(round_num)
        else:
            for r in range(1, self.config.max_rounds + 1):
                await self.run_round(r)

    async def run_round(self, round_num: int):
        """
        Orchestrates a single round of research steps among all agents.
        """
        self.logger.info(f"=== Running Research Round {round_num} ===")

        try:
            # 1) Agents perceive environment state
            self.logger.info(f"Round {round_num}: Agents perceiving environment...")
            perceptions = await self.cognitive_processor.run_parallel_perceive(self.agents, self.config.name)
            
            # Log perceptions
            for agent, perception in zip(self.agents, perceptions or []):
                log_persona(self.logger, agent.index, agent.persona)
                log_perception(
                    self.logger, 
                    agent.index, 
                    perception.json_object.object if perception and perception.json_object else None
                )

            # 2) Agents produce summary actions
            self.logger.info(f"Round {round_num}: Gathering agent research summaries...")
            actions = await self.cognitive_processor.run_parallel_action(self.agents, self.config.name)
            
            # Log actions
            for agent, action in zip(self.agents, actions or []):
                if action and action.json_object and action.json_object.object:
                    log_action(self.logger, agent.index, str(action.json_object.object))
                else:
                    log_action(self.logger, agent.index, action.str_content if action else "No action")

            # 3) Construct a GlobalAction and step the environment
            global_actions = {}
            for agent, action_response in zip(self.agents, actions or []):
                try:
                    # Parse the action response into our summary model
                    if action_response and action_response.json_object and action_response.json_object.object:
                        summary_dict = action_response.json_object.object
                        summary_instance = self.summary_model.model_validate(summary_dict)
                    else:
                        # Create an empty instance if no valid response
                        summary_instance = self.summary_model.model_construct()
                    
                    local_action = ResearchAction(agent_id=agent.id, action=summary_instance)
                    global_actions[agent.id] = local_action
                except Exception as e:
                    self.logger.error(f"Error creating action for agent {agent.id}: {e}")
                    # Create empty instance as fallback
                    summary_instance = self.summary_model.model_construct()
                    local_action = ResearchAction(agent_id=agent.id, action=summary_instance)
                    global_actions[agent.id] = local_action

            research_global_action = ResearchGlobalAction(actions=global_actions)
            step_result = self.environment.step(research_global_action)

            # 4) Process results and store in database
            await self.process_round_results(round_num, step_result)

            # 5) Agents reflect on new environment observation
            self.logger.info(f"Round {round_num}: Agents reflecting on environment changes...")
            try:
                reflections = await self.cognitive_processor.run_parallel_reflect(self.agents, self.config.name)
                
                # Log reflections
                if reflections:
                    for agent, reflection in zip(self.agents, reflections):
                        log_reflection(
                            self.logger, 
                            agent.index, 
                            reflection.json_object.object if reflection and reflection.json_object else None
                        )
                else:
                    self.logger.warning("No reflections received from agents")
            except Exception as e:
                self.logger.error(f"Error during reflection step: {e}")
                self.logger.exception("Reflection step failed but continuing...")

            self.logger.info(f"Round {round_num} complete.\n")
            
        except Exception as e:
            self.logger.error(f"Error in round {round_num}: {e}")
            self.logger.exception("Round failed")
            raise

    async def process_round_results(self, round_num: int, step_result: EnvironmentStep = None):
        """
        Optional: Insert step_result in DB, or process any environment info.
        You can also store partial aggregator results or finalize them at the end.
        """
        self.logger.info(f"Processing results for round {round_num}...")

        # Insert environment step data
        self.data_inserter.insert_round_data(
            round_num=round_num,
            agents=self.agents,
            environment=self.environment,
            config=self.orchestrator_config,
            tracker=None,
            environment_name=self.config.name
        )
        self.logger.info(f"Results for round {round_num} saved.")

    def process_environment_state(self, env_state):
        """
        A required abstract method from BaseEnvironmentOrchestrator.
        You could do something with the env_state (like partial aggregator logic).
        """
        self.logger.info(f"Processing environment state: {env_state}")

    async def get_round_summary(self, round_num: int) -> Dict[str, Any]:
        """
        Return a summary of a particular round. 
        For example, you can gather each agent's last action or environment observation.
        """
        summary = {
            "round": round_num,
            "agent_states": [],
        }
        for agent in self.agents:
            # e.g. gather agent last action, last observation
            agent_info = {
                "id": agent.id,
                "index": agent.index,
                "last_action": agent.last_action,
                "last_observation": agent.last_observation
            }
            summary["agent_states"].append(agent_info)
        return summary

    async def run(self):
        """
        A main entrypoint if you want a single call to run everything:
        - Setup environment
        - Run all rounds
        - Print or store final results
        """
        self.setup_environment()
        await self.run_environment()
        self.logger.info("All rounds complete. Printing final summary...\n")
        await self.print_summary()

    async def print_summary(self):
        """
        Print or store any final summary, aggregator results, etc.
        We'll just fetch the environment's aggregator state if final.
        """
        self.logger.info("=== RESEARCH SIMULATION SUMMARY ===")

        # Possibly the environment may have final aggregator data in last step
        global_state = self.environment.get_global_state()
        self.logger.info(f"Final Environment State: {global_state}")

        # Or you might fetch from environment.history, or from the Mechanism if you prefer
        # E.g. final aggregator summaries from the last round
        # ...

        print("\nFinal Agent States:")
        for agent in self.agents:
            print(f"Agent {agent.index}: last action = {agent.last_action}")
        print()

        self.logger.info("Finished printing summary.")