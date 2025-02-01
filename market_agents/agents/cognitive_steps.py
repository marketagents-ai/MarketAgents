from typing import Dict, Any, Optional, Type, Union, List
from datetime import datetime, timezone
import json
import asyncio

from pydantic import BaseModel, Field

from market_agents.agents.cognitive_schemas import PerceptionSchema, ReflectionSchema
from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.memory import MemoryObject
from market_agents.agents.market_agent_prompter import MarketAgentPromptVariables

class CognitiveStep(BaseModel):
    """Base class for cognitive steps in the market agent's cognitive cycle."""
    
    step_name: str = Field(
        ..., 
        description="Name identifier for this cognitive step"
    )
    agent_id: str = Field(
        ...,
        description="ID of the agent executing this step"
    )
    environment_name: str = Field(
        ...,
        description="Name of the environment being interacted with"
    )
    environment_info: Any = Field(
        ...,
        description="Information about the current environment state"
    )
    structured_tool: bool = Field(
        default=False,
        description="Whether to use structured output schema"
    )
    return_prompt: bool = Field(
        default=False,
        description="Whether to return the prompt instead of executing"
    )

    async def execute(self, agent: 'MarketAgent') -> Union[str, Dict[str, Any]]:
        """Execute the cognitive step and return the result."""
        raise NotImplementedError

    async def store_memory(self, 
                          agent: 'MarketAgent',
                          content: Any,
                          metadata: Dict[str, Any]) -> None:
        """Store the step's result in agent memory."""
        memory = MemoryObject(
            agent_id=self.agent_id,
            cognitive_step=self.step_name,
            metadata=metadata,
            content=json.dumps(content),
            created_at=datetime.now(timezone.utc)
        )
        agent.episode_steps.append(memory)
        await agent.short_term_memory.store_memory(memory)

class CognitiveEpisode(BaseModel):
    """A sequence of cognitive steps forming an episode."""
    
    steps: List[Type[CognitiveStep]] = Field(
        ...,
        description="Ordered list of cognitive step classes to execute"
    )
    environment_name: str = Field(
        ...,
        description="Name of the environment for this episode"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata for the episode"
    )
    
    class Config:
        arbitrary_types_allowed = True

class PerceptionStep(CognitiveStep):
    step_name: str = "perception"

    async def execute(self, agent: 'MarketAgent') -> Union[str, Dict[str, Any]]:
        stm_cognitive = await agent.short_term_memory.retrieve_recent_memories(limit=10)
        short_term_memories = [
            {"cognitive_step": mem.cognitive_step, "content": mem.content, "metadata": mem.metadata}
            for mem in stm_cognitive
        ]

        task_str = f"Task: {agent.task}" if agent.task else ""
        env_state_str = f"Environment state: {str(self.environment_info)}"
        agent_state = f"Agent state: {str(agent.last_action)} {str(agent.last_observation)}"
        query_str = (task_str + "\n" + env_state_str + "\n" + agent_state).strip()

        ltm_episodes = await agent.long_term_memory.retrieve_episodic_memories(
            agent_id=self.agent_id,
            query=query_str,
            top_k=3
        )

        retrieved_documents = []
        if agent.knowledge_agent:
            retrieved_documents = await agent.knowledge_agent.retrieve(
                query_str,
                limit=5
            )

        variables = MarketAgentPromptVariables(
            environment_name=self.environment_name,
            environment_info=self.environment_info,
            short_term_memory=short_term_memories,
            long_term_memory=[episode.model_dump() for episode in ltm_episodes],
            documents=[doc.model_dump() for doc in retrieved_documents],
            observation=agent.last_observation,
            last_action=agent.last_action
        )

        perception_prompt = agent.prompt_manager.get_perception_prompt(variables.model_dump())
        if agent.chat_thread:
            agent.chat_thread.new_message = perception_prompt
            agent.chat_thread.forced_output = (
                PerceptionSchema.model_json_schema() if self.structured_tool else None
            )

        if self.return_prompt:
            return perception_prompt

        result = await agent.execute()
        await self.store_memory(
            agent,
            content=result,
            metadata={
                "environment_name": self.environment_name,
                "environment_info": self.environment_info,
                "observation": agent.last_observation,
                "last_action": agent.last_action
            }
        )
        agent.last_perception = result
        return result

class ActionStep(CognitiveStep):
    step_name: str = "action"
    perception: Optional[str] = Field(
        default=None,
        description="Agent's perception from previous step"
    )
    action_schema: Optional[Dict] = Field(
        default=None,
        description="Schema for structured action output"
    )

    async def execute(self, agent: 'MarketAgent') -> Union[str, Dict[str, Any]]:
        environment = agent.environments[self.environment_name]
        action_space = environment.action_space
        
        serialized_action_space = {
            "allowed_actions": [
                action_type.__name__ 
                for action_type in action_space.allowed_actions
            ],
            "constraints": action_space.get_constraints() if hasattr(action_space, 'get_constraints') else {}
        }

        variables = MarketAgentPromptVariables(
            environment_name=self.environment_name,
            environment_info=self.environment_info,
            perception=agent.last_perception,  # Using stored perception
            action_space=serialized_action_space,
            last_action=agent.last_action,
            observation=agent.last_observation
        )

        action_prompt = agent.prompt_manager.get_action_prompt(variables.model_dump())
        if agent.chat_thread:
            agent.chat_thread.new_message = action_prompt
            agent.chat_thread.forced_output = (
                (self.action_schema or action_space.get_action_schema())
                if self.structured_tool else None
            )

        if self.return_prompt:
            return action_prompt

        result = await agent.execute()
        action = {"sender": self.agent_id, "content": result}
        
        await self.store_memory(
            agent,
            content=result,
            metadata={
                "action_space": serialized_action_space,
                "last_action": agent.last_action,
                "observation": agent.last_observation,
                "perception": agent.last_perception,
                "environment_name": self.environment_name,
                "environment_info": self.environment_info
            }
        )
        agent.last_action = result
        return action

class ReflectionStep(CognitiveStep):
    step_name: str = "reflection"
    environment_reward_weight: float = Field(
        default=0.5,
        description="Weight for environment reward in total reward calculation"
    )
    self_reward_weight: float = Field(
        default=0.5,
        description="Weight for self-assigned reward in total reward calculation"
    )

    async def execute(self, agent: 'MarketAgent') -> Union[str, Dict[str, Any]]:
        total_weight = self.environment_reward_weight + self.self_reward_weight
        if total_weight == 0:
            raise ValueError("Sum of weights must not be zero.")
        
        environment_reward_weight = self.environment_reward_weight / total_weight
        self_reward_weight = self.self_reward_weight / total_weight

        environment = agent.environments[self.environment_name]
        last_step = environment.history.steps[-1][1] if environment.history.steps else None

        if last_step:
            reward = last_step.info.get('agent_rewards', {}).get(self.agent_id, 0.0) or 0.0
            local_observation = last_step.global_observation.observations.get(self.agent_id)
            observation = local_observation.observation if local_observation else {}
        else:
            observation = {}
            reward = 0.0

        previous_strategy = "No previous strategy available"
        try:
            previous_reflection = await agent.short_term_memory.retrieve_recent_memories(
                cognitive_step='reflection',
                limit=1
            )
            if previous_reflection:
                last_reflection_obj = previous_reflection[0]
                previous_strategy = last_reflection_obj.metadata.get("strategy_update", "")
                if isinstance(previous_strategy, list):
                    previous_strategy = " ".join(previous_strategy)
        except Exception as e:
            previous_strategy = f"Error retrieving previous strategy: {str(e)}"

        variables = MarketAgentPromptVariables(
            environment_name=self.environment_name,
            environment_info=self.environment_info,
            observation=observation,
            last_action=agent.last_action,
            reward=reward,
            previous_strategy=previous_strategy,
            perception=agent.last_perception
        )

        reflection_prompt = agent.prompt_manager.get_reflection_prompt(variables.model_dump())
        if agent.chat_thread:
            agent.chat_thread.new_message = reflection_prompt
            agent.chat_thread.forced_output = (
                ReflectionSchema.model_json_schema() if self.structured_tool else None
            )

        if self.return_prompt:
            return reflection_prompt

        result = await agent.execute()
        
        if isinstance(result, dict):
            self_reward = result.get("self_reward", 0.0)
            environment_reward = reward
            normalized_environment_reward = max(
                0.0,
                min(environment_reward / (1 + abs(environment_reward)), 1.0)
            )

            total_reward_val = (
                normalized_environment_reward * environment_reward_weight +
                self_reward * self_reward_weight
            )

            await self.store_memory(
                agent,
                content=result.get("reflection", ""),
                metadata={
                    "total_reward": round(total_reward_val, 4),
                    "self_reward": round(self_reward, 4),
                    "observation": observation,
                    "strategy_update": result.get("strategy_update", ""),
                    "environment_reward": round(environment_reward, 4),
                    "environment_name": self.environment_name,
                    "environment_info": self.environment_info,
                    "perception": agent.last_perception
                }
            )

            task_str = f"Task: {agent.task}" if agent.task else ""
            env_state_str = f"Environment state: {str(self.environment_info)}"
            agent_state = f"Agent state: {str(agent.last_action)} {str(agent.last_observation)}"
            query_str = (task_str + "\n" + env_state_str + "\n" + agent_state).strip()
            
            await agent.long_term_memory.store_episode(
                task_query=query_str,
                steps=agent.episode_steps,
                total_reward=round(total_reward_val, 4),
                strategy_update=result.get("strategy_update", ""),
                metadata={
                    "environment_name": self.environment_name,
                    "final_observation": observation,
                    "final_action": agent.last_action,
                    "final_perception": agent.last_perception
                }
            )
            agent.episode_steps.clear()

        return result