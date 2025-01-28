# research_orchestrator.py

import asyncio
import importlib
import json
import logging
from typing import List, Dict, Any, Type


from market_agents.orchestrators.logger_utils import log_action, log_perception, log_persona, log_reflection
from pydantic import BaseModel

from market_agents.orchestrators.base_orchestrator import BaseEnvironmentOrchestrator
from market_agents.orchestrators.config import OrchestratorConfig, ResearchConfig
from market_agents.agents.market_agent import MarketAgent
from market_agents.inference.parallel_inference import ParallelAIUtilities
from market_agents.orchestrators.orchestration_data_inserter import OrchestrationDataInserter
from market_agents.orchestrators.agent_cognitive import AgentCognitiveProcessor
from market_agents.environments.environment import EnvironmentStep
from market_agents.environments.mechanisms.research import (
    ResearchEnvironment,
    ResearchGlobalAction,
    ResearchAction
)

from market_agents.memory.agent_storage.storage_service import StorageService

class ResearchOrchestrator(BaseEnvironmentOrchestrator):
    def __init__(
        self,
        config: ResearchConfig,
        agents: List[MarketAgent],
        ai_utils: ParallelAIUtilities,
        storage_service: StorageService,
        orchestrator_config: OrchestratorConfig,
        logger=None,
        **kwargs
    ):
        super().__init__(
            config=config,
            orchestrator_config=orchestrator_config,
            agents=agents,
            ai_utils=ai_utils,
            storage_service=storage_service,
            logger=logger,
            environment_name=config.name
        )
        
        self.data_inserter = OrchestrationDataInserter(storage_service=storage_service)
        self.cognitive_processor = AgentCognitiveProcessor(
            ai_utils=self.ai_utils,
            storage_service=storage_service,
            logger=self.logger,
            tool_mode=self.orchestrator_config.tool_mode
        )
        
        self.summary_model = self.get_schema_model(self.config.schema_model)
        self.environment = ResearchEnvironment(
            summary_model=self.summary_model,
            name=self.config.name,
            api_url=self.config.api_url,
            max_steps=self.config.max_rounds
        )

        for agent in self.agents:
            agent.environments[self.config.name] = self.environment

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
        """Orchestrates a single round of research steps among all agents."""
        self.logger.info(f"=== Running Research Round {round_num} ===")

        try:
            # Agents perceive environment state
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

            # Agents produce summary actions
            self.logger.info(f"Round {round_num}: Gathering agent research summaries...")
            actions = await self.cognitive_processor.run_parallel_action(self.agents, self.config.name)
            
            # Log actions and store them in agent state
            for agent, action in zip(self.agents, actions or []):
                try:
                    content = None
                    if action and action.json_object and action.json_object.object:
                        content = action.json_object.object
                    elif action and hasattr(action, 'str_content'):
                        content = action.str_content

                    # Store the action in agent's state
                    agent.last_action = content
                    
                    # Log the action using the same format as perception/reflection
                    log_action(self.logger, agent.index, content)

                except Exception as e:
                    self.logger.error(f"Error processing action for agent {agent.id}: {str(e)}", exc_info=True)
                    agent.last_action = None

            # Construct a GlobalAction and step the environment
            global_actions = {}
            for agent, action_response in zip(self.agents, actions or []):
                try:
                    if action_response and action_response.json_object and action_response.json_object.object:
                        summary_dict = action_response.json_object.object
                        summary_instance = self.summary_model.model_validate(summary_dict)
                    else:
                        summary_instance = self.summary_model.model_construct()
                    
                    local_action = ResearchAction(agent_id=agent.id, action=summary_instance)
                    global_actions[agent.id] = local_action
                    
                except Exception as e:
                    self.logger.error(
                        f"Error creating action for agent {agent.id}: {str(e)}\n"
                        f"Raw action data: {action_response.json_object.object if action_response and action_response.json_object else None}",
                        exc_info=True
                    )
                    summary_instance = self.summary_model.model_construct()
                    local_action = ResearchAction(agent_id=agent.id, action=summary_instance)
                    global_actions[agent.id] = local_action

                research_global_action = ResearchGlobalAction(actions=global_actions)
                step_result = self.environment.step(research_global_action)

                # After environment step, store the observation in agent state
                for agent in self.agents:
                    if (step_result and 
                        step_result.global_observation and 
                        step_result.global_observation.observations):
                        # Access observations through global_observation
                        agent.last_observation = step_result.global_observation.observations.get(agent.id)
                    else:
                        agent.last_observation = None

                # Agents reflect on new environment observation
                self.logger.info(f"Round {round_num}: Agents reflecting on environment changes...")
                try:
                    reflection_prompts = []
                    agents_with_observations = []
                    
                    for agent in self.agents:
                        if agent.last_observation and agent.last_observation.observation:
                            reflect_prompt = await agent.reflect(
                                self.config.name, 
                                return_prompt=True, 
                                structured_tool=self.orchestrator_config.tool_mode
                            )
                            reflection_prompts.append(reflect_prompt)
                            agents_with_observations.append(agent)
                    
                    if reflection_prompts:
                        reflections = await self.cognitive_processor.run_parallel_reflect(
                            agents_with_observations, 
                            self.config.name
                        )
                        
                        # Log reflections
                        if reflections:
                            for agent, reflection in zip(agents_with_observations, reflections):
                                try:
                                    content = None
                                    if reflection and reflection.json_object and reflection.json_object.object:
                                        content = reflection.json_object.object
                                    elif reflection and hasattr(reflection, 'str_content'):
                                        content = reflection.str_content

                                    if content:
                                        # Store reflection in agent's state
                                        agent.last_reflection = content
                                        # Log the reflection
                                        log_reflection(self.logger, agent.index, content)
                                    else:
                                        self.logger.warning(f"No reflection content for agent {agent.index}")
                                except Exception as e:
                                    self.logger.error(f"Error processing reflection for agent {agent.id}: {str(e)}")
                        else:
                            self.logger.warning("No reflections received from agents")
                    else:
                        self.logger.warning("No agents had observations to reflect on")

                except Exception as e:
                    self.logger.error(f"Error during reflection step: {str(e)}", exc_info=True)
                    self.logger.exception("Reflection step failed but continuing...")

                self.logger.info(f"Round {round_num} complete.\n")

            # Process results and store in database
            await self.process_round_results(round_num, step_result)
            
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

        # Insert environment step data - removed 'tracker' parameter
        await self.data_inserter.insert_round_data(
            round_num=round_num,
            agents=self.agents,
            environment=self.environment,
            config=self.orchestrator_config,
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
        """Print or store any final summary, aggregator results, etc."""
        # ANSI color codes
        PINK = "\033[95m"
        TEAL = "\033[96m"
        RESET = "\033[0m"
        
        self.logger.info("=== RESEARCH SIMULATION SUMMARY ===")

        # Get environment state
        global_state = self.environment.get_global_state()
        self.logger.info(f"Final Environment State: {global_state}")

        # Print last actions in both formats
        print(f"\n{PINK}Final Agent States (JSONL):{RESET}")
        for agent in self.agents:
            if agent.last_action:
                try:
                    # Create a dictionary with agent index and action
                    jsonl_entry = {
                        "agent_index": agent.index,
                        "last_action": agent.last_action if isinstance(agent.last_action, dict) else json.loads(agent.last_action)
                    }
                    print(f"{PINK}{json.dumps(jsonl_entry)}{RESET}")
                except Exception as e:
                    self.logger.error(f"Error creating JSONL for agent {agent.index}: {e}")

        print(f"\n{TEAL}Final Agent States (Pretty):{RESET}")
        for agent in self.agents:
            print(f"\n{TEAL}Agent {agent.index}:{RESET}")
            if agent.last_action:
                try:
                    # Try to format as JSON if possible
                    if isinstance(agent.last_action, dict):
                        print(f"{TEAL}Last action = {json.dumps(agent.last_action, indent=2)}{RESET}")
                    else:
                        print(f"{TEAL}Last action = {agent.last_action}{RESET}")
                except Exception as e:
                    print(f"{TEAL}Last action = {str(agent.last_action)}{RESET}")
            else:
                print(f"{TEAL}Last action = None{RESET}")
        print()

        self.logger.info("Finished printing summary.")