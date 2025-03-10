# orchestrator.py

from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
import pkgutil
from typing import List, Dict, Any, Optional, Type
import logging
import json

from market_agents.orchestrators.logger_utils import log_action, log_perception, log_reflection, log_persona
from market_agents.orchestrators.parallel_cognitive_steps import ParallelCognitiveProcessor
from market_agents.orchestrators.orchestration_data_inserter import OrchestrationDataInserter
from market_agents.environments.environment import EnvironmentStep, GlobalAction, LocalAction, MultiAgentEnvironment
from market_agents.memory.agent_storage.storage_service import StorageService

class OrchestratorLocalAction(LocalAction):
    """Concrete implementation of LocalAction for orchestrator use."""
    @classmethod
    def sample(cls, agent_id: str) -> 'OrchestratorLocalAction':
        """Implement required sample method."""
        return cls(agent_id=agent_id, action={})
    
class MultiAgentOrchestrator:
    def __init__(
        self,
        config: Any,
        agents: List[Any],
        storage_service: StorageService,
        orchestrator_config: Any,
        environment_name: str,
        logger: Optional[logging.Logger] = None,
        ai_utils: Any = None,
        **kwargs
    ):
        self.config = config
        self.agents = agents
        self.storage_service = storage_service
        self.orchestrator_config = orchestrator_config
        self.logger = logger or logging.getLogger(__name__)
        self.ai_utils = ai_utils
        self.environment_name = environment_name

        # Initialize environment based on config
        self.environment = self._initialize_environment()
        
        self.data_inserter = OrchestrationDataInserter(storage_service=storage_service)
        self.cognitive_processor = ParallelCognitiveProcessor(
            ai_utils=self.ai_utils,
            storage_service=self.storage_service,
            logger=self.logger,
            tool_mode=self.orchestrator_config.tool_mode
        )

    def _get_environment_class(self) -> Type[MultiAgentEnvironment]:
        """Dynamically discover and load environment mechanism classes."""
        try:
            # Import the module directly
            module = import_module(f"market_agents.environments.mechanisms.{self.environment_name}")
            
            # Look for any class that inherits from MultiAgentEnvironment
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, MultiAgentEnvironment) and 
                    attr != MultiAgentEnvironment):
                    self.logger.debug(f"Found environment class: {attr.__name__}")
                    return attr
                        
            raise ValueError(f"No environment class found in {self.environment_name}")
                
        except ImportError as e:
            self.logger.error(f"Failed to import environment module: {e}")
            raise ValueError(f"Could not load environment: {self.environment_name}") from e

    def _initialize_environment(self) -> MultiAgentEnvironment:
        """Initialize the environment based on config."""
        try:
            environment_class = self._get_environment_class()
            
            # Get environment config from the orchestrator config
            env_config = self.orchestrator_config.environment_configs.get(self.environment_name)
            if not env_config:
                raise ValueError(f"No config found for environment {self.environment_name}")
                
            # Convert config to dict and ensure required fields
            env_params = env_config.dict() if hasattr(env_config, 'dict') else env_config
            
            # Debug logging
            self.logger.debug(f"Initializing {self.environment_name} with params: {env_params}")
            
            # Initialize environment with config parameters
            environment = environment_class(**env_params)
            self.logger.info(f"Successfully initialized {self.environment_name} environment")
            return environment
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.environment_name} environment: {e}", exc_info=True)
            raise

    async def setup_environment(self):
        """Set up or reset the environment."""
        self.logger.info("Setting up the Environment...")
        if self.environment:
            self.environment.reset()
        self.logger.info("Environment setup complete.")

    async def run_environment(self, round_num: Optional[int] = None):
        """Run the environment for the specified round or for all rounds."""
        if round_num is not None:
            await self.run_round(round_num)
        else:
            for r in range(1, self.orchestrator_config.max_rounds + 1):
                await self.run_round(r)

    async def run_round(self, round_num: int):
        """Execute one full round with multiple sub-rounds."""
        self.logger.info(f"=== Running Round {round_num} ===")
        self._initialize_agents_for_round()
        for sub_round in range(1, self.config.sub_rounds + 1):
            self.logger.info(f"=== Starting Sub-round {sub_round}/{self.config.sub_rounds} of Round {round_num} ===")
            try:
                step_result = await self._run_sub_round(round_num, sub_round)
                await self.process_round_results(round_num, step_result, sub_round)
            except Exception as e:
                self.logger.error(f"Error in round {round_num}, sub-round {sub_round}: {e}", exc_info=True)
                raise
        self.logger.info(f"Round {round_num} complete with {self.config.sub_rounds} sub-rounds.\n")

    def _initialize_agents_for_round(self):
        """Initialize agents for the round. Override this if environment-specific initialization is needed."""
        for agent in self.agents:
            # Initialize environment for each agent
            if not hasattr(agent, 'environments'):
                agent.environments = {}
            agent.environments[self.environment_name] = self.environment
            
            # Set task based on environment topic
            topic = getattr(self.environment, "initial_topic", None) or getattr(self.environment, "initial_query", "")
            agent.task = f"Your task: {topic}"
            if hasattr(agent, "_refresh_prompts"):
                agent._refresh_prompts()

    async def _run_sub_round(self, round_num: int, sub_round: int) -> EnvironmentStep:
        """Execute one sub-round by running perception, action, and reflection phases."""
        try:
            # Perception Phase
            await self._run_perception_phase(round_num, sub_round)
            # Action Phase
            step_result = await self._run_action_phase(round_num, sub_round)
            # Reflection Phase
            await self._run_reflection_phase(round_num, sub_round)
            return step_result
        except Exception as e:
            self.logger.error(f"Error in sub-round {sub_round} of round {round_num}: {e}", exc_info=True)
            raise

    async def _run_perception_phase(self, round_num: int, sub_round: int):
        """Generic perception phase."""
        self.logger.info(f"Round {round_num}.{sub_round}: Agents perceiving environment...")
        perceptions = await self.cognitive_processor.run_parallel_perception(self.agents, self.environment_name)
        for agent, perception in zip(self.agents, perceptions or []):
            log_persona(self.logger, agent.id, agent.persona)
            content = None
            if perception and perception.json_object:
                content = perception.json_object.object
            log_perception(self.logger, agent.id, content)
        return perceptions

    async def _run_action_phase(self, round_num: int, sub_round: int) -> EnvironmentStep:
        """Generic action phase that gathers agent actions and processes them."""
        self.logger.info(f"Round {round_num}.{sub_round}: Executing agent actions...")
        actions = await self.cognitive_processor.run_parallel_action(self.agents, self.environment_name)
        agent_summaries = await self._process_agent_actions(actions)
        self.logger.info(f"Processed agent summaries: {agent_summaries}")
        global_actions = await self._create_global_actions(actions)
        # Assume the environment accepts a global action to perform a step.
        step_result = self.environment.step(global_actions)
        if step_result and getattr(step_result, "global_observation", None):
            await self._update_agent_observations(step_result, agent_summaries)
        return step_result

    async def _process_agent_actions(self, actions) -> Dict[str, Any]:
        """Process individual agent actions and create a summary."""
        agent_summaries = {}
        for agent, action in zip(self.agents, actions or []):
            try:
                content = None
                if action and hasattr(action, 'json_object') and action.json_object and getattr(action.json_object, 'object', None):
                    content = action.json_object.object
                elif action and hasattr(action, 'str_content'):
                    content = action.str_content
                agent.last_action = content
                if content:
                    agent_summaries[agent.id] = content
                log_action(self.logger, agent.id, content, model_name=getattr(agent, 'llm_config', {}).get('model') if hasattr(agent, 'llm_config') else None)
            except Exception as e:
                self.logger.error(f"Error processing action for agent {agent.id}: {e}", exc_info=True)
                agent.last_action = None
        return agent_summaries

    async def _create_global_actions(self, actions) -> GlobalAction:
        """Create GlobalAction using orchestrator-specific LocalAction."""
        local_actions = {}
        for agent, action in zip(self.agents, actions or []):
            try:
                if action and hasattr(action, 'json_object') and action.json_object:
                    action_content = action.json_object.object
                else:
                    action_content = {}
                local_actions[agent.id] = OrchestratorLocalAction(
                    agent_id=agent.id, 
                    action=action_content
                )
            except Exception as e:
                self.logger.error(f"Error creating global action for agent {agent.id}: {e}", exc_info=True)
                local_actions[agent.id] = OrchestratorLocalAction(
                    agent_id=agent.id, 
                    action={}
                )
        return GlobalAction(actions=local_actions)

    async def _update_agent_observations(self, step_result: EnvironmentStep, agent_summaries: Dict[str, Any]):
        """Update agent observations based on the environment's step result."""
        if hasattr(step_result.global_observation, 'all_actions_this_round'):
            step_result.global_observation.all_actions_this_round = agent_summaries
        for agent in self.agents:
            if step_result.global_observation and hasattr(step_result.global_observation, 'observations') and agent.id in step_result.global_observation.observations:
                agent.last_observation = step_result.global_observation.observations[agent.id]
            else:
                agent.last_observation = None

    async def _run_reflection_phase(self, round_num: int, sub_round: int):
        """Generic reflection phase for agents to reflect on environment outcomes."""
        self.logger.info(f"Round {round_num}.{sub_round}: Agents reflecting...")
        agents_with_observations = [agent for agent in self.agents if getattr(agent, 'last_observation', None)]
        if not agents_with_observations:
            self.logger.warning("No agents had observations to reflect on")
            return
        reflections = await self.cognitive_processor.run_parallel_reflection(
            agents_with_observations,
            environment_name=self.environment_name
        )
        if not reflections:
            self.logger.warning("No reflections received from agents")
            return
        for agent, reflection_output in zip(agents_with_observations, reflections):
            try:
                content = reflection_output.json_object.object if reflection_output and hasattr(reflection_output, 'json_object') and reflection_output.json_object else reflection_output.str_content
                if content:
                    self.logger.info(f"Agent {agent.id} reflection: {content}")
                else:
                    self.logger.warning(f"No reflection content for agent {agent.id}")
            except Exception as e:
                self.logger.error(f"Error processing reflection for agent {agent.id}: {e}", exc_info=True)

    async def process_round_results(self, round_num: int, step_result: Optional[EnvironmentStep] = None, sub_round: Optional[int] = None):
        """Process and store results for the round."""
        try:
            actions_data = []
            for agent in self.agents:
                if hasattr(agent, 'last_action') and agent.last_action:
                    actions_data.append({
                        'agent_id': agent.id,
                        'environment_name': self.config.name,
                        'round': round_num,
                        'sub_round': sub_round,
                        'action': agent.last_action,
                        'type': 'generic_summary'
                    })
            if actions_data:
                await self.data_inserter.insert_actions(actions_data)
            if hasattr(self.environment, 'get_global_state'):
                env_state = self.environment.get_global_state()
                config_dict = self.orchestrator_config.model_dump() if hasattr(self.orchestrator_config, 'model_dump') else vars(self.orchestrator_config)
                metadata = {
                    'config': config_dict,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'num_agents': len(self.agents),
                    'sub_round': sub_round
                }
                await self.data_inserter.insert_environment_state(self.config.name, round_num, env_state, metadata)
            self.logger.info(f"Data for round {round_num}, sub-round {sub_round} inserted.")
        except Exception as e:
            self.logger.error(f"Error processing round {round_num} results: {e}", exc_info=True)
            raise

    async def get_round_summary(self, round_num: int) -> Dict[str, Any]:
        """Return a summary of the round."""
        summary = {
            "round": round_num,
            "agent_states": [],
            "environment": self.config.name,
            "topic": getattr(self.environment, "initial_topic", "")
        }
        for agent in self.agents:
            recent_mem = None
            if hasattr(agent, 'short_term_memory'):
                try:
                    mems = await agent.short_term_memory.retrieve_recent_memories(limit=1)
                    if mems:
                        recent_mem = mems[0].content
                except Exception:
                    pass
            try:
                if agent.last_observation:
                    observation_data = agent.last_observation.model_dump() if hasattr(agent.last_observation, 'model_dump') else agent.last_observation
                else:
                    observation_data = None
            except Exception as e:
                self.logger.warning(f"Failed to serialize observation for agent {agent.id}: {e}")
                observation_data = str(agent.last_observation)
            summary["agent_states"].append({
                "id": agent.id,
                "last_action": agent.last_action,
                "last_observation": observation_data,
                "memory": recent_mem
            })
        return summary

    async def run(self):
        """Main entrypoint to run the orchestrator."""
        await self.setup_environment()
        await self.run_environment()
        self.logger.info("All rounds complete. Printing final summary...\n")
        await self.print_summary()

    async def print_summary(self):
        """Print or log the final summary."""
        PINK = "\033[95m"
        TEAL = "\033[96m"
        RESET = "\033[0m"
        self.logger.info("=== SIMULATION SUMMARY ===")
        global_state = self.environment.get_global_state() if self.environment and hasattr(self.environment, 'get_global_state') else {}
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
                    self.logger.error(f"Error creating JSONL for agent {agent.id}: {e}", exc_info=True)
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