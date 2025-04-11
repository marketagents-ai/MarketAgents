# orchestrator.py
import uuid
import asyncio
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
import pkgutil
from typing import List, Dict, Any, Optional, Type
import logging
import json

from market_agents.orchestrators.logger_utils import log_action, log_cohort_formation, log_perception, log_persona
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
            # Get mechanism type from config
            if not hasattr(self.config, 'mechanism'):
                raise ValueError(f"No mechanism type specified for environment {self.environment_name}")
            
            mechanism_type = self.config.mechanism  # Access directly as attribute
            
            # Import the module based on mechanism type
            module = import_module(f"market_agents.environments.mechanisms.{mechanism_type}")
            
            # Look for any class that inherits from MultiAgentEnvironment
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, MultiAgentEnvironment) and 
                    attr != MultiAgentEnvironment):
                    self.logger.debug(f"Found environment class: {attr.__name__}")
                    return attr
                    
            raise ValueError(f"No environment class found for mechanism {mechanism_type}")
                
        except ImportError as e:
            self.logger.error(f"Failed to import environment module: {e}")
            raise ValueError(f"Could not load mechanism: {mechanism_type}") from e

    def _initialize_environment(self) -> MultiAgentEnvironment:
        """Initialize the environment based on config."""
        try:
            environment_class = self._get_environment_class()
            
            # Convert config to dict if it's a Pydantic model
            env_params = self.config.model_dump() if hasattr(self.config, 'model_dump') else dict(self.config)
            
            # Add ai_utils to the environment parameters
            env_params['ai_utils'] = self.ai_utils
            
            # Debug logging
            self.logger.debug(f"Initializing {self.environment_name} with params: {env_params}")
            
            # Initialize environment with config parameters and ai_utils
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
            # Add this line to properly initialize the environment
            if hasattr(self.environment, 'initialize'):
                self.logger.info("Initializing environment...")
                await self.environment.initialize()
                self.logger.info("Environment initialization complete")
                
            self._initialize_agents_for_environment()
            
            # Rest of the setup...
            if hasattr(self.environment.mechanism, '_cognitive_processor'):
                self.environment.mechanism._cognitive_processor = self.cognitive_processor
            
            form_cohorts = (
                hasattr(self.environment.mechanism, 'form_cohorts') and 
                getattr(self.environment.mechanism, 'form_cohorts', False)
            )
            
            if form_cohorts:
                self.logger.info(f"Forming cohorts with {len(self.agents)} agents")
                await self.environment.mechanism.form_agent_cohorts(self.agents)
                
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

        # Simply check if we should use cohorts for execution
        form_cohorts = (
            hasattr(self.environment.mechanism, 'form_cohorts') and 
            self.environment.mechanism.form_cohorts and
            hasattr(self.environment.mechanism, 'cohorts') and
            bool(self.environment.mechanism.cohorts)
        )
        
        for sub_round in range(1, self.config.sub_rounds + 1):
            self.logger.info(f"=== Starting Sub-round {sub_round}/{self.config.sub_rounds} of Round {round_num} ===")
            
            if form_cohorts:
                # Run each cohort's complete cycle in parallel
                cohort_tasks = []
                for cohort_id, cohort_agents in self.environment.mechanism.cohorts.items():
                    task = asyncio.create_task(
                        self._run_sub_round(
                            round_num=round_num,
                            sub_round=sub_round,
                            cohort_agents=cohort_agents
                        )
                    )
                    cohort_tasks.append(task)
                
                all_results = await asyncio.gather(*cohort_tasks)
                
            else:
                # Regular single-cohort execution
                step_result = await self._run_sub_round(round_num, sub_round)
                await self.process_round_results(round_num, sub_round)

    def _initialize_agents_for_environment(self):
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

    async def _run_sub_round(
        self, 
        round_num: int, 
        sub_round: int, 
        cohort_agents: Optional[List[Any]] = None
    ) -> EnvironmentStep:
        """Execute one sub-round by running perception, action, and reflection phases."""
        try:
            # Perception Phase
            await self._run_perception_phase(round_num, sub_round, cohort_agents)
            
            # Action Phase
            step_result = await self._run_action_phase(round_num, sub_round, cohort_agents)

             # Process results immediately
            if step_result:
                await self.process_round_results(round_num, sub_round, cohort_agents)
            
            
            # Reflection Phase
            await self._run_reflection_phase(round_num, sub_round, cohort_agents)
            
            return step_result
        except Exception as e:
            self.logger.error(f"Error in sub-round {sub_round} of round {round_num}: {e}", exc_info=True)
            raise

    async def _run_perception_phase(self, round_num: int, sub_round: int, cohort_agents: Optional[List[Any]] = None):
        """Generic perception phase."""
        agents = cohort_agents if cohort_agents is not None else self.agents
        self.logger.info(f"Round {round_num}.{sub_round}: Agents perceiving environment...")
        
        perceptions = await self.cognitive_processor.run_parallel_perception(
            agents, 
            self.environment_name
        )
        
        for agent, perception in zip(agents, perceptions or []):
            persona_str = agent.persona.persona if agent.persona else str(agent.persona)
            log_persona(self.logger, agent.id, persona_str)
            content = None
            if perception and perception.json_object:
                content = perception.json_object.object
            log_perception(self.logger, agent.id, content)
        return perceptions

    async def _run_action_phase(self, round_num: int, sub_round: int, cohort_agents: Optional[List[Any]] = None) -> EnvironmentStep:
        """Generic action phase."""
        agents = cohort_agents if cohort_agents is not None else self.agents
        self.logger.info(f"Round {round_num}.{sub_round}: Executing agent actions...")
        
        actions = await self.cognitive_processor.run_parallel_action(
            agents,
            self.environment_name
        )
        
        agent_summaries = await self._process_agent_actions(actions, agents)
        global_actions = await self._create_global_actions(actions, agents)
        
        # Get cohort_id if using cohorts
        cohort_id = None
        if cohort_agents and hasattr(self.environment.mechanism, 'cohorts'):
            cohort_id = next(
                (cid for cid, cohort_agents in self.environment.mechanism.cohorts.items() 
                if any(a.id == agents[0].id for a in cohort_agents)),
                None
            )
        
        # Pass cohort_id to environment step
        step_result = self.environment.step(global_actions, cohort_id=cohort_id)
        
        if step_result and getattr(step_result, "global_observation", None):
            await self._update_agent_observations(step_result, agent_summaries, agents)
            
        return step_result

    async def _process_agent_actions(self, actions: List[Any], cohort_agents: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Process individual agent actions and create a summary.
        
        Args:
            actions: List of agent actions
            agents: Optional list of agents (for cohort processing). If None, uses self.agents
        """
        agent_summaries = {}
        # Use provided agents list (for cohorts) or fall back to all agents
        agents = cohort_agents if cohort_agents is not None else self.agents
        
        for agent, action in zip(agents, actions or []):
            try:
                content = None
                if action and hasattr(action, 'json_object') and action.json_object and getattr(action.json_object, 'object', None):
                    content = action.json_object.object
                elif action and hasattr(action, 'str_content'):
                    content = action.str_content
                agent.last_action = content
                if content:
                    agent_summaries[agent.id] = content
                log_action(
                    self.logger, 
                    agent.id, 
                    content, 
                    model_name=getattr(agent, 'llm_config', {}).get('model') if hasattr(agent, 'llm_config') else None
                )
            except Exception as e:
                self.logger.error(f"Error processing action for agent {agent.id}: {e}", exc_info=True)
                agent.last_action = None
        return agent_summaries

    async def _create_global_actions(self, actions, cohort_agents: Optional[List[Any]] = None) -> GlobalAction:
        """Create GlobalAction using orchestrator-specific LocalAction."""
        local_actions = {}
        agents = cohort_agents if cohort_agents is not None else self.agents
        
        for agent, action in zip(agents, actions or []):
            try:
                # Extract action content exactly like the old orchestrator
                if action and hasattr(action, 'json_object') and action.json_object:
                    action_content = action.json_object.object
                else:
                    action_content = {}
                    
                # Convert UUID to string for agent_id
                agent_id_str = str(agent.id)
                    
                # Create LocalAction with string agent_id
                local_actions[agent_id_str] = OrchestratorLocalAction(
                    agent_id=agent_id_str,  # Use string version of UUID
                    action=action_content
                )
            except Exception as e:
                self.logger.error(f"Error creating global action for agent {agent.id}: {e}")
                # Use string version in fallback case too
                agent_id_str = str(agent.id)
                local_actions[agent_id_str] = OrchestratorLocalAction(
                    agent_id=agent_id_str,
                    action={}
                )
        
        return GlobalAction(actions=local_actions)

    async def _update_agent_observations(self, step_result: EnvironmentStep, agent_summaries: Dict[str, Any], agents: List[Any]):
        try:
            if (step_result.global_observation and 
                hasattr(step_result.global_observation, 'all_actions_this_round')):
                step_result.global_observation.all_actions_this_round = agent_summaries

            # Update individual agent observations
            if step_result.global_observation:
                for agent in agents:
                    # Convert UUID to string for comparison
                    agent_id_str = str(agent.id)
                    if agent_id_str in step_result.global_observation.observations:
                        agent.last_observation = step_result.global_observation.observations[agent_id_str]
                        self.logger.debug(f"Updated observation for agent {agent_id_str}: {agent.last_observation}")
                    else:
                        self.logger.warning(f"No observation for agent {agent_id_str}")
                        agent.last_observation = None
        except Exception as e:
            self.logger.error(f"Error in _update_agent_observations: {e}")
            raise

    async def _run_reflection_phase(self, round_num: int, sub_round: int, cohort_agents: Optional[List[Any]] = None):
        """Generic reflection phase."""
        agents = cohort_agents if cohort_agents is not None else self.agents
        self.logger.info(f"Round {round_num}.{sub_round}: Agents reflecting...")
        
        try:
            # Debug log all agents' observations
            self.logger.info("=== Agent Observations Before Reflection ===")
            for agent in agents:
                self.logger.info(
                    f"Agent {str(agent.id)}:"  # Convert UUID to string
                    f"\nlast_observation: {agent.last_observation}"
                )

            # Simplified filtering condition
            agents_with_observations = [
                agent for agent in agents 
                if agent.last_observation is not None
            ]

            self.logger.info(f"Found {len(agents_with_observations)} agents with observations")

            if not agents_with_observations:
                self.logger.info("No agents had observations to reflect on")
                self.logger.info("Agents without observations: " + 
                                ", ".join(str(agent.id) for agent in agents))
                return
                    
            self.logger.info(f"Running reflection for {len(agents_with_observations)} agents")
            reflections = await self.cognitive_processor.run_parallel_reflection(
                agents_with_observations,
                self.environment_name
            )
            
            if reflections:
                self.logger.info(f"Received {len(reflections)} reflections")
            else:
                self.logger.info("No reflections received")
                        
        except Exception as e:
            self.logger.error(f"Error during reflection step: {str(e)}", exc_info=True)
            self.logger.exception("Reflection step failed but continuing...")

    def _convert_uuids_to_strings(self, obj):
        """Recursively convert all UUIDs to strings in a nested structure."""
        if isinstance(obj, dict):
            return {k: self._convert_uuids_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_uuids_to_strings(item) for item in obj]
        elif isinstance(obj, uuid.UUID):
            return str(obj)
        return obj

    async def process_round_results(self, round_num: int, sub_round: int, cohort_agents: Optional[List[Any]] = None):
        """Process and store results for the round."""
        try:
            actions_data = []
            agents = cohort_agents if cohort_agents else self.agents
            for agent in agents:
                if hasattr(agent, 'last_action') and agent.last_action:
                    actions_data.append({
                        'agent_id': str(agent.id),
                        'environment_name': self.config.name,
                        'round': round_num,
                        'sub_round': sub_round,
                        'action': agent.last_action,
                        'type': 'generic_summary'
                    })
            if actions_data:
                await self.data_inserter.insert_actions(actions_data)
            
            # Get environment state for each agent in cohort
            if hasattr(self.environment, 'get_global_state'):
                for agent in agents:
                    # Get and convert environment state
                    env_state = self.environment.get_global_state(agent_id=str(agent.id))
                    env_state = self._convert_uuids_to_strings(env_state)
                    
                    # Convert config to dict and handle UUIDs
                    config_dict = self.orchestrator_config.model_dump() if hasattr(self.orchestrator_config, 'model_dump') else vars(self.orchestrator_config)
                    config_dict = self._convert_uuids_to_strings(config_dict)
                    
                    metadata = {
                        'config': config_dict,
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'num_agents': len(agents),
                        'sub_round': sub_round,
                        'agent_id': str(agent.id)
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
        self.logger.info("=== ORCHESTRATION SUMMARY ===")
        
        # Get environment state which includes cohort organization if enabled
        global_state = self.environment.get_global_state() if self.environment and hasattr(self.environment, 'get_global_state') else {}
        self.logger.info(f"Final Environment State: {global_state}")
        
        # Check if using cohorts
        form_cohorts = (
            hasattr(self.environment.mechanism, 'form_cohorts') and 
            getattr(self.environment.mechanism, 'form_cohorts', False)
        )

        if form_cohorts:
            print(f"\n{PINK}Final Cohort States:{RESET}")
            for cohort_id, cohort_agents in self.environment.mechanism.cohorts.items():
                print(f"\n{PINK}Cohort {cohort_id}:{RESET}")
                for agent in cohort_agents:
                    if agent.last_action:
                        try:
                            jsonl_entry = {
                                "cohort": cohort_id,
                                "agent_id": str(agent.id),
                                "last_action": agent.last_action if isinstance(agent.last_action, dict) else json.loads(agent.last_action)
                            }
                            print(f"{PINK}{json.dumps(jsonl_entry)}{RESET}")
                        except Exception as e:
                            self.logger.error(f"Error creating JSONL for agent {agent.id}: {e}", exc_info=True)
        else:
            # Original non-cohort summary logic
            print(f"\n{PINK}Final Agent States (JSONL):{RESET}")
            for agent in self.agents:
                if agent.last_action:
                    try:
                        jsonl_entry = {
                            "agent_id": str(agent.id),
                            "last_action": agent.last_action if isinstance(agent.last_action, dict) else json.loads(agent.last_action)
                        }
                        print(f"{PINK}{json.dumps(jsonl_entry)}{RESET}")
                    except Exception as e:
                        self.logger.error(f"Error creating JSONL for agent {agent.id}: {e}", exc_info=True)

        # Pretty print section remains similar but organized by cohorts if enabled
        print(f"\n{TEAL}Final States (Pretty):{RESET}")
        if form_cohorts:
            for cohort_id, cohort_agents in self.environment.mechanism.cohorts.items():
                print(f"\n{TEAL}Cohort {cohort_id}:{RESET}")
                for agent in cohort_agents:
                    print(f"\n{TEAL}Agent {str(agent.id)}:{RESET}")
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
        else:
            for agent in self.agents:
                print(f"\n{TEAL}Agent {str(agent.id)}:{RESET}")
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