# research_orchestrator.py

from datetime import datetime, timezone
import importlib
import json
from typing import List, Dict, Any, Type


from market_agents.orchestrators.logger_utils import log_action, log_perception, log_persona, log_reflection
from pydantic import BaseModel

from market_agents.orchestrators.base_orchestrator import BaseEnvironmentOrchestrator
from market_agents.orchestrators.config import OrchestratorConfig, ResearchConfig
from market_agents.agents.market_agent import MarketAgent
from market_agents.orchestrators.orchestration_data_inserter import OrchestrationDataInserter
from market_agents.orchestrators.parallel_cognitive_steps import ParallelCognitiveProcessor
from market_agents.environments.environment import EnvironmentStep
from market_agents.environments.mechanisms.research import (
    ResearchEnvironment,
    ResearchGlobalAction,
    ResearchAction,
    ResearchObservation
)

from market_agents.memory.agent_storage.storage_service import StorageService

class ResearchOrchestrator(BaseEnvironmentOrchestrator):
    def __init__(
        self,
        config: ResearchConfig,
        agents: List[MarketAgent],
        storage_service: StorageService,
        orchestrator_config: OrchestratorConfig,
        logger=None,
        ai_utils=None,
        **kwargs
    ):
        super().__init__(
            config=config,
            orchestrator_config=orchestrator_config,
            agents=agents,
            storage_service=storage_service,
            logger=logger,
            environment_name=config.name,
            ai_utils=ai_utils,
            **kwargs
        )
        
        self.data_inserter = OrchestrationDataInserter(storage_service=storage_service)
        self.cognitive_processor = ParallelCognitiveProcessor(
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
            max_steps=self.config.sub_rounds,
            initial_topic=self.config.initial_topic
        )

        for agent in self.agents:
            agent.environments[self.config.name] = self.environment

        self.logger.info(f"Initialized ResearchOrchestrator for environment: {self.config.name}")

    def get_schema_model(self, schema_name: str) -> Type[BaseModel]:
        try:
            schemas_module = importlib.import_module('market_agents.orchestrators.research_schemas')
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
        Run the environment for the specified round number.
        Within each round, run the configured number of sub-rounds.
        """
        if round_num is not None:
            await self.run_research_round(round_num)
        else:
            for r in range(1, self.orchestrator_config.max_rounds + 1):
                await self.run_research_round(r)

    async def run_research_round(self, round_num: int):
        """Orchestrates a single main round with multiple sub-rounds of research steps."""
        self.logger.info(f"=== Running Research Round {round_num} ===")

        # Initialize agents for this round
        self._initialize_agents_for_round()

        # Run each sub-round
        for sub_round in range(1, self.config.sub_rounds + 1):
            self.logger.info(f"=== Starting Sub-round {sub_round}/{self.config.sub_rounds} of Round {round_num} ===")
            try:
                step_result = await self._run_sub_round(round_num, sub_round)
                await self.process_round_results(round_num, step_result, sub_round)
            except Exception as e:
                self.logger.error(f"Error in round {round_num}, sub-round {sub_round}: {e}")
                self.logger.exception("Sub-round failed")
                raise

        self.logger.info(f"Round {round_num} complete with {self.config.sub_rounds} sub-rounds.\n")

    def _initialize_agents_for_round(self):
        """Initialize agents with the current research topic."""
        for agent in self.agents:
            agent.task = f"You are assigned with the following research topic:\n{self.environment.initial_topic}"
            agent._refresh_prompts()

    async def _run_sub_round(self, round_num: int, sub_round: int):
        """Executes a single sub-round of the research process."""
        try:
            # 1. Perception Phase
            perceptions = await self._run_perception_phase(round_num, sub_round)
            
            # 2. Action Phase
            step_result = await self._run_action_phase(round_num, sub_round)
            
            # 3. Reflection Phase
            await self._run_reflection_phase(round_num, sub_round)
            
            return step_result
            
        except Exception as e:
            self.logger.error(f"Error in sub-round {sub_round} of round {round_num}: {e}")
            raise

    async def _run_perception_phase(self, round_num: int, sub_round: int):
        """Handles the perception phase of the cognitive cycle."""
        self.logger.info(f"Round {round_num}.{sub_round}: Agents perceiving environment...")
        perceptions = await self.cognitive_processor.run_parallel_perception(self.agents, self.config.name)
        
        for agent, perception in zip(self.agents, perceptions or []):
            log_persona(self.logger, agent.id, agent.persona)
            log_perception(
                self.logger, 
                agent.id, 
                perception.json_object.object if perception and perception.json_object else None
            )
        
        return perceptions

    async def _run_action_phase(self, round_num: int, sub_round: int):
        """Handles the action phase of the cognitive cycle."""
        self.logger.info(f"Round {round_num}.{sub_round}: Gathering agent research summaries...")
        
        for agent in self.agents:
            self.logger.debug(f"Agent {agent.id} state before action: chat_thread={agent.chat_thread}, task={agent.task}")
        
        actions = await self.cognitive_processor.run_parallel_action(self.agents, self.config.name)
        
        for agent, action in zip(self.agents, actions or []):
            self.logger.debug(f"Agent {agent.id} action received: {action}")
        
        agent_summaries = await self._process_agent_actions(actions)
        self.logger.info(f"Processed summaries: {agent_summaries}")
        
        global_actions = await self._create_global_actions(actions)
        
        research_global_action = ResearchGlobalAction(actions=global_actions)
        step_result = self.environment.step(research_global_action)
        
        if step_result and step_result.global_observation:
            await self._update_agent_observations(step_result, agent_summaries)
        
        return step_result

    async def _process_agent_actions(self, actions):
        """Process individual agent actions and create summaries."""
        agent_summaries = {}
        
        for agent, action in zip(self.agents, actions or []):
            try:
                content = None
                if action and action.json_object and action.json_object.object:
                    content = action.json_object.object
                elif action and hasattr(action, 'str_content'):
                    content = action.str_content

                agent.last_action = content
                if content:
                    agent_summaries[agent.id] = content
                
                log_action(self.logger, agent.id, content)

            except Exception as e:
                self.logger.error(f"Error processing action for agent {agent.id}: {str(e)}", exc_info=True)
                agent.last_action = None
        
        return agent_summaries

    async def _create_global_actions(self, actions):
        """Create global actions from individual agent actions."""
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
        
        return global_actions

    async def _update_agent_observations(self, step_result, agent_summaries):
        """Update agent observations based on step results."""
        step_result.global_observation.all_actions_this_round = agent_summaries
        
        for agent in self.agents:
            if (step_result.global_observation and 
                step_result.global_observation.observations and 
                agent.id in step_result.global_observation.observations):
                
                local_obs = step_result.global_observation.observations[agent.id]
                if not local_obs.observation:
                    local_obs.observation = ResearchObservation()
                
                agent.last_observation = local_obs
            else:
                agent.last_observation = None

    async def _run_reflection_phase(self, round_num: int, sub_round: int):
        """Handles the reflection phase of the cognitive cycle."""
        self.logger.info(f"Round {round_num}.{sub_round}: Agents reflecting on environment changes...")
        try:
            agents_with_observations = [
                agent for agent in self.agents 
                if agent.last_observation and hasattr(agent.last_observation, 'observation') and agent.last_observation.observation
            ]

            if not agents_with_observations:
                self.logger.warning("No agents had observations to reflect on")
                return

            reflections = await self.cognitive_processor.run_parallel_reflection(
                agents_with_observations,
                environment_name=self.config.name
            )
            
            if not reflections:
                self.logger.warning("No reflections received from agents")
                return

            for agent, reflection_output in zip(agents_with_observations, reflections):
                try:
                    if reflection_output.json_object and reflection_output.json_object.object:
                        content = reflection_output.json_object.object
                    else:
                        content = reflection_output.str_content

                    if content:
                        self.logger.info(f"Agent {agent.id} reflection: {content}")
                    else:
                        self.logger.warning(f"No reflection content for agent {agent.id}")
                except Exception as e:
                    self.logger.error(f"Error processing reflection for agent {agent.id}: {str(e)}")

        except Exception as e:
            self.logger.error(f"Error during reflection step: {str(e)}", exc_info=True)
            self.logger.exception("Reflection step failed but continuing...")

    async def process_round_results(self, round_num: int, step_result=None, sub_round: int = None):
        """Process and store results for the round & insert data into DB.
        
        Args:
            round_num: The current round number
            step_result: Optional environment step result
            sub_round: Optional sub-round number
        """
        try:
            # Process actions
            actions_data = []
            for agent in self.agents:
                if hasattr(agent, 'last_action') and agent.last_action:
                    actions_data.append({
                        'agent_id': agent.id,
                        'environment_name': self.config.name,
                        'round': round_num,
                        'sub_round': sub_round,
                        'action': agent.last_action,
                        'type': 'research_summary'
                    })
            
            if actions_data:
                await self.data_inserter.insert_actions(actions_data)

            # Process environment state
            if hasattr(self.environment, 'get_global_state'):
                env_state = self.environment.get_global_state()
                config_dict = self.orchestrator_config.model_dump() if hasattr(self.orchestrator_config, 'model_dump') else vars(self.orchestrator_config)
                metadata = {
                    'config': config_dict,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'num_agents': len(self.agents),
                    'sub_round': sub_round
                }
                await self.data_inserter.insert_environment_state(
                    self.config.name,
                    round_num,
                    env_state,
                    metadata
                )

            self.logger.info(f"Data for round {round_num}, sub-round {sub_round} inserted.")
            
        except Exception as e:
            self.logger.error(f"Error processing round {round_num} results: {e}")
            self.logger.exception("Details:")
            raise

    def process_environment_state(self, env_state):
        """
        A required abstract method from BaseEnvironmentOrchestrator.
        You could do something with the env_state (like partial aggregator logic).
        """
        self.logger.info(f"Processing environment state: {env_state}")

    async def get_round_summary(self, round_num: int) -> Dict[str, Any]:
        """Return a summary of the round for later printing."""
        summary = {
            "round": round_num,
            "agent_states": [],
            "environment": self.config.name,
            "topic": self.environment.initial_topic
        }

        for agent in self.agents:
            recent_mem = None
            if agent.short_term_memory:
                try:
                    mems = await agent.short_term_memory.retrieve_recent_memories(limit=1)
                    if mems:
                        recent_mem = mems[0].content
                except Exception:
                    pass

            # Properly serialize the observation
            try:
                if agent.last_observation:
                    if hasattr(agent.last_observation, 'model_dump'):
                        observation_data = agent.last_observation.model_dump()
                    elif isinstance(agent.last_observation, dict):
                        observation_data = agent.last_observation
                    else:
                        # Convert to a basic dictionary representation
                        observation_data = {
                            'own_summary': getattr(agent.last_observation, 'own_summary', None),
                            'current_topic': getattr(agent.last_observation, 'current_topic', None),
                            # Add other relevant fields as needed
                        }
                else:
                    observation_data = None
            except Exception as e:
                self.logger.warning(f"Failed to serialize observation for agent {agent.id}: {e}")
                observation_data = str(agent.last_observation)

            summary["agent_states"].append({
                "id": agent.id,
                "index": getattr(agent, "index", None),
                "last_action": agent.last_action,
                "last_observation": observation_data,
                "memory": recent_mem
            })

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
        PINK = "\033[95m"
        TEAL = "\033[96m"
        RESET = "\033[0m"
        
        self.logger.info("=== RESEARCH SIMULATION SUMMARY ===")

        global_state = self.environment.get_global_state()
        self.logger.info(f"Final Environment State: {global_state}")

        print(f"\n{PINK}Final Agent States (JSONL):{RESET}")
        for agent in self.agents:
            if agent.last_action:
                try:
                    jsonl_entry = {
                        "agent_index": agent.id,
                        "last_action": agent.last_action if isinstance(agent.last_action, dict) else json.loads(agent.last_action)
                    }
                    print(f"{PINK}{json.dumps(jsonl_entry)}{RESET}")
                except Exception as e:
                    self.logger.error(f"Error creating JSONL for agent {agent.id}: {e}")

        print(f"\n{TEAL}Final Agent States (Pretty):{RESET}")
        for agent in self.agents:
            print(f"\n{TEAL}Agent {agent.id}:{RESET}")
            if agent.last_action:
                try:
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