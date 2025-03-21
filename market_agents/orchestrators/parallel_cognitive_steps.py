import asyncio
from datetime import datetime, timezone
import json
import logging
from typing import Any, List, Dict

from market_agents.agents.cognitive_steps import PerceptionStep, ActionStep, ReflectionStep
from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.storage_service import StorageService
from market_agents.memory.memory import MemoryObject
from minference.lite.models import ProcessedOutput

from market_agents.orchestrators.logger_utils import log_reflection

logger = logging.getLogger(__name__)

class ParallelCognitiveProcessor:
    def __init__(self, ai_utils, storage_service: StorageService, logger: logging.Logger, tool_mode=False):
        self.ai_utils = ai_utils
        self.storage_service = storage_service
        self.logger = logger
        self.tool_mode = tool_mode

        self.episode_steps: Dict[str, List[MemoryObject]] = {}

        asyncio.create_task(self._ensure_ai_requests_table())

    async def _ensure_ai_requests_table(self):
        """Ensure the AI requests table exists in the database."""
        try:
            await self.storage_service.create_tables(table_type="ai_requests")
        except Exception as e:
            self.logger.error(f"Failed to create AI requests table: {e}")

    def _serialize_content(self, content: Any) -> str:
        """Utility to serialize content (dict/string/etc.) to JSON safely."""
        if isinstance(content, (dict, list)):
            try:
                return json.dumps(content)
            except Exception as e:
                self.logger.warning(f"Failed to serialize content: {e}")
        return str(content)

    def _get_safe_id(self, agent_id: str) -> str:
        """Sanitize agent ids if needed for dictionary keys."""
        return agent_id.replace("-", "_")
    
    def get_all_requests(self):
        requests = self.ai_utils.all_requests
        self.ai_utils.all_requests = []  
        return requests

    async def run_parallel_perception(
        self,
        agents: List[MarketAgent],
        environment_name: str
    ) -> List[ProcessedOutput]:
        """Collect PerceptionStep with return_prompt=True for each agent"""
        perception_prompts = []
        for agent in agents:
            env_state = agent.environments[environment_name].get_global_state(agent_id=agent.id)
            step = PerceptionStep(
                agent_id=agent.id,
                environment_name=environment_name,
                environment_info=env_state,
                structured_tool=self.tool_mode,
                return_prompt=True
            )
            prompt_thread = await agent.run_step(step=step)
            print(agent.chat_thread.new_message)

            perception_prompts.append(prompt_thread)

        outputs = await self.ai_utils.run_parallel_ai_completion(perception_prompts)
        await self.storage_service.store_ai_requests(self.get_all_requests())

        for agent, output in zip(agents, outputs):
            text_content = (output.json_object.object if output.json_object else output.str_content)
            memory_obj = MemoryObject(
                agent_id=agent.id,
                cognitive_step="perception",
                content=self._serialize_content(text_content),
                metadata={"environment_name": environment_name},
                created_at=datetime.now(timezone.utc),
            )
            await agent.short_term_memory.store_memory(memory_obj)
            agent.episode_steps.append(memory_obj)
            agent.last_perception = text_content

        return outputs

    async def run_parallel_action(
        self,
        agents: List[MarketAgent],
        environment_name: str
    ) -> List[ProcessedOutput]:
        """Collect ActionStep prompts with return_prompt=True"""
        action_prompts = []
        for agent in agents:
            step = ActionStep(
                agent_id=agent.id,
                environment_name=environment_name,
                environment_info=agent.environments[environment_name].get_global_state(agent_id=agent.id),
                structured_tool=self.tool_mode,
                return_prompt=True
            )
            prompt_thread = await agent.run_step(step=step)
            print(agent.chat_thread.new_message)

            action_prompts.append(prompt_thread)

        outputs = await self.ai_utils.run_parallel_ai_completion(action_prompts)
        await self.storage_service.store_ai_requests(self.get_all_requests())

        for agent, output in zip(agents, outputs):
            text_content = (output.json_object.object if output.json_object else output.str_content)
            memory_obj = MemoryObject(
                agent_id=agent.id,
                cognitive_step="action",
                content=self._serialize_content(text_content),
                metadata={"environment_name": environment_name},
                created_at=datetime.now(timezone.utc),
            )
            await agent.short_term_memory.store_memory(memory_obj)
            agent.episode_steps.append(memory_obj)
            agent.last_action = text_content

        return outputs

    async def run_parallel_reflection(
        self,
        agents: List[MarketAgent],
        environment_name: str
    ) -> List[ProcessedOutput]:
        """Collect ReflectionStep prompts with return_prompt=True"""
        reflection_prompts = []
        agents_with_observations = []

        for agent in agents:
            if agent.last_observation:
                step = ReflectionStep(
                    agent_id=agent.id,
                    environment_name=environment_name,
                    environment_info=agent.environments[environment_name].get_global_state(agent_id=agent.id),
                    structured_tool=self.tool_mode,
                    return_prompt=True
                )
                prompt = await agent.run_step(step=step)
                print(agent.chat_thread.new_message)
                self.logger.info(f"Running reflection step for agent {agent.id}")
                reflection_prompts.append(prompt)
                agents_with_observations.append(agent)

        if not reflection_prompts:
            self.logger.info("No reflection prompts collected")

            return []

        outputs = await self.ai_utils.run_parallel_ai_completion(reflection_prompts)
        await self.storage_service.store_ai_requests(self.get_all_requests())

        for agent, output in zip(agents_with_observations, outputs):
            safe_id = self._get_safe_id(agent.id)
            if safe_id not in self.episode_steps:
                self.episode_steps[safe_id] = []

            reflection_content = (output.json_object.object
                                  if output and output.json_object
                                  else output.str_content)

            log_reflection(self.logger, agent.id, reflection_content)

            environment = agent.environments.get(environment_name)
            if hasattr(environment, "mechanism") and environment.mechanism and hasattr(environment.mechanism, "last_step"):
                last_step = environment.mechanism.last_step
                environment_reward = last_step.info.get("agent_rewards", {}).get(agent.id, 0.0) if last_step else None
            else:
                environment_reward = None

            self_reward = 0.0
            total_reward = None
            if isinstance(reflection_content, dict):
                self_reward = float(reflection_content.get("self_reward", 0.0))

            if environment_reward is not None:
                try:
                    normalized_env_reward = environment_reward / (1 + abs(environment_reward))
                    normalized_env_reward = max(0.0, min(normalized_env_reward, 1.0))
                    total_reward = (normalized_env_reward * 0.5) + (self_reward * 0.5)

                    self.logger.info(
                        f"Agent {getattr(agent, 'index', agent.id)} rewards - "
                        f"Environment: {environment_reward}, Normalized: {normalized_env_reward}, "
                        f"Self: {self_reward}, Total: {total_reward}"
                    )
                except Exception as e:
                    self.logger.warning(f"Error computing total_reward: {e}")

            try:
                if agent.last_observation and hasattr(agent.last_observation, "dict"):
                    observation_data = agent.last_observation.dict()
                else:
                    observation_data = str(agent.last_observation)
            except Exception as e:
                observation_data = str(agent.last_observation)
                self.logger.warning(f"Failed to serialize agent's observation: {e}")

            metadata = {
                "environment": environment_name,
                "self_reward": round(self_reward, 4),
            }
            if environment_reward is not None:
                metadata["environment_reward"] = round(environment_reward, 4)
            if total_reward is not None:
                metadata["total_reward"] = round(total_reward, 4)
            if observation_data:
                metadata["observation"] = observation_data

            memory_obj = MemoryObject(
                agent_id=agent.id,
                cognitive_step="reflection",
                content=self._serialize_content(reflection_content),
                metadata=metadata,
                created_at=datetime.now(timezone.utc)
            )
            await agent.short_term_memory.store_memory(memory_obj)
            self.episode_steps[safe_id].append(memory_obj)

            task_str = f"Task: {agent.task}" if agent.task else ""
            env_state_str = f"Environment: {str(agent.environments[environment_name].get_global_state(agent_id=agent.id))}"
            combined_query = (task_str + "\n" + env_state_str).strip()

            serializable_metadata = {
                "environment": environment_name,
                "observation": observation_data
            }

            await agent.long_term_memory.store_episode(
                task_query=combined_query,
                steps=self.episode_steps[safe_id],
                total_reward=total_reward,
                strategy_update=reflection_content.get("strategy_update", "") if isinstance(reflection_content, dict) else None,
                metadata=serializable_metadata
            )
            self.episode_steps[safe_id].clear()

        return outputs