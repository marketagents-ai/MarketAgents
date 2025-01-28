from datetime import datetime, timezone
import json
import logging
from typing import Any, List
from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.storage_service import StorageService
from market_agents.memory.memory import MemoryObject
from market_agents.orchestrators.logger_utils import log_perception, log_persona, log_reflection


class AgentCognitiveProcessor:
    def __init__(self, ai_utils, data_inserter, logger: logging.Logger, tool_mode=False):
        self.ai_utils = ai_utils
        self.data_inserter = data_inserter
        self.logger = logger
        self.tool_mode = tool_mode
        self.episode_steps = {}

    def _get_safe_id(self, agent_id: str) -> str:
        """Get sanitized agent ID consistent with memory storage"""
        return StorageService._sanitize_id(agent_id)
    
    def _serialize_content(self, content: Any) -> str:
        """Serialize content to JSON string, handling Pydantic models and datetimes"""
        def serialize_value(v):
            if isinstance(v, datetime):
                return v.isoformat()
            if hasattr(v, 'serialize_json'):
                return json.loads(v.serialize_json())
            if hasattr(v, 'model_dump'):
                return v.model_dump()
            if isinstance(v, dict):
                return {k: serialize_value(v2) for k, v2 in v.items()}
            if isinstance(v, list):
                return [serialize_value(v2) for v2 in v]
            return v

        try:
            if hasattr(content, 'serialize_json'):
                return content.serialize_json()
            elif hasattr(content, 'model_dump'):
                return json.dumps(content.model_dump(), default=serialize_value)
            elif isinstance(content, dict):
                return json.dumps({k: serialize_value(v) for k, v in content.items()})
            elif isinstance(content, list):
                return json.dumps([serialize_value(v) for v in content])
            return str(content)
        except Exception as e:
            self.logger.warning(f"Failed to serialize content: {e}")
            return str(content)

    async def run_parallel_perceive(self, agents: List[MarketAgent], environment_name: str) -> List[Any]:
        perception_prompts = []
        for agent in agents:
            perception_prompt = await agent.perceive(environment_name, return_prompt=True, structured_tool=self.tool_mode)
            perception_prompts.append(perception_prompt)
        
        perceptions = await self.ai_utils.run_parallel_ai_completion(perception_prompts, update_history=False)
        self.data_inserter.insert_ai_requests(self.ai_utils.get_all_requests())
        
        # Log personas and perceptions, and store in memory
        for agent, perception in zip(agents, perceptions):
            safe_id = self._get_safe_id(agent.id)
            if safe_id not in self.episode_steps:
                self.episode_steps[safe_id] = []
                
            log_persona(self.logger, agent.index, agent.persona)
            perception_content = perception.json_object.object if perception and perception.json_object else perception.str_content
            log_perception(self.logger, agent.index, perception_content)
            
            # Store in short-term memory and episode steps
            memory_obj = MemoryObject(
                agent_id=agent.id,
                cognitive_step="perception",
                content=self._serialize_content(perception_content),
                metadata={"environment": environment_name},
                created_at=datetime.now(timezone.utc)
            )
            await agent.short_term_memory.store_memory(memory_obj)
            self.episode_steps[safe_id].append(memory_obj)
            agent.last_perception = perception_content

        return perceptions

    async def run_parallel_action(self, agents: List[MarketAgent], environment_name: str) -> List[Any]:
        action_prompts = []
        for agent in agents:
            action_prompt = await agent.generate_action(environment_name, agent.last_perception, return_prompt=True, structured_tool=self.tool_mode)
            action_prompts.append(action_prompt)
            
        actions = await self.ai_utils.run_parallel_ai_completion(action_prompts, update_history=False)
        self.data_inserter.insert_ai_requests(self.ai_utils.get_all_requests())
        
        # Store actions in memory
        for agent, action in zip(agents, actions):
            safe_id = self._get_safe_id(agent.id)
            if safe_id not in self.episode_steps:
                self.episode_steps[safe_id] = []
                
            action_content = action.json_object.object if action and action.json_object else action.str_content
            memory_obj = MemoryObject(
                agent_id=agent.id,
                cognitive_step="action",
                content=self._serialize_content(action_content),
                metadata={"environment": environment_name},
                created_at=datetime.now(timezone.utc)
            )
            await agent.short_term_memory.store_memory(memory_obj)
            self.episode_steps[safe_id].append(memory_obj)
            
        return actions

    async def run_parallel_reflect(self, agents: List[MarketAgent], environment_name: str) -> None:
        reflection_prompts = []
        agents_with_observations = []
        
        for agent in agents:
            if agent.last_observation:
                reflect_prompt = await agent.reflect(environment_name, return_prompt=True, structured_tool=self.tool_mode)
                reflection_prompts.append(reflect_prompt)
                agents_with_observations.append(agent)
                
        if reflection_prompts:
            reflections = await self.ai_utils.run_parallel_ai_completion(reflection_prompts, update_history=False)
            self.data_inserter.insert_ai_requests(self.ai_utils.get_all_requests())
            
            for agent, reflection in zip(agents_with_observations, reflections):
                safe_id = self._get_safe_id(agent.id)
                if safe_id not in self.episode_steps:
                    self.episode_steps[safe_id] = []
                    
                if reflection.json_object:
                    reflection_content = reflection.json_object.object
                    log_reflection(self.logger, agent.index, reflection_content)
                    
                    # Calculate rewards
                    environment_reward = agent.last_step.info.get('agent_rewards', {}).get(agent.id, 0.0) if agent.last_step else None
                    self_reward = reflection_content.get("self_reward", 0.0)
                    
                    total_reward = None
                    if environment_reward is not None:
                        normalized_env_reward = environment_reward / (1 + abs(environment_reward))
                        normalized_env_reward = max(0.0, min(normalized_env_reward, 1.0))
                        total_reward = normalized_env_reward * 0.5 + self_reward * 0.5
                        
                        self.logger.info(
                            f"Agent {agent.index} rewards - Environment: {environment_reward}, "
                            f"Normalized: {normalized_env_reward}, Self: {self_reward}, "
                            f"Total: {total_reward}"
                        )

                    # Store reflection in memory
                    observation_data = self._serialize_content(agent.last_observation) if agent.last_observation else None
                            
                    memory_obj = MemoryObject(
                        agent_id=agent.id,
                        cognitive_step="reflection",
                        content=self._serialize_content(reflection_content),
                        metadata={
                            "environment": environment_name,
                            "self_reward": round(self_reward, 4),
                            **({"environment_reward": round(environment_reward, 4)} if environment_reward is not None else {}),
                            **({"total_reward": round(total_reward, 4)} if total_reward is not None else {}),
                            **({"observation": observation_data} if observation_data else {})
                        },
                        created_at=datetime.now(timezone.utc)
                    )
                    await agent.short_term_memory.store_memory(memory_obj)
                    self.episode_steps[safe_id].append(memory_obj)

                    # Store episodic memory and clear episode steps
                    task_str = f"Task: {agent.task}" if agent.task else ""
                    env_state_str = f"Environment state: {str(agent.environments[environment_name].get_global_state())}"
                    query_str = (task_str + "\n" + env_state_str).strip()
                    
                    await agent.long_term_memory.store_episodic_memory(
                        agent_id=agent.id,
                        task_query=query_str,
                        steps=self.episode_steps[safe_id],
                        total_reward=total_reward,
                        strategy_update=reflection_content.get("strategy_update", []),
                        metadata={
                            "environment": environment_name,
                            "observation": agent.last_observation
                        }
                    )
                    self.episode_steps[safe_id].clear()