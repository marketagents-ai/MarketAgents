from typing import Dict, Any, Optional, Type, Union, List
from datetime import datetime, timezone
import json
import asyncio

from pydantic import BaseModel, Field

from market_agents.memory.memory import MemoryObject
from market_agents.agents.market_agent_prompter import MarketAgentPromptVariables
from minference.lite.models import CallableTool, StructuredTool
from market_agents.agents.cognitive_tools import (
    perception_tool,
    reflection_tool
)

class CognitiveStep(BaseModel):
    """
    Base class for cognitive steps in the market agent's cognitive cycle.
    """
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
        default=True,
        description="Whether to use structured output schema"
    )
    return_prompt: bool = Field(
        default=False,
        description="Whether to return the prompt instead of executing"
    )

    async def execute(self, agent: BaseModel) -> Union[str, Dict[str, Any]]:
        """
        Execute the cognitive step and return the result.
        This must be overridden in subclasses (PerceptionStep, ActionStep, ReflectionStep).
        """
        raise NotImplementedError

    async def store_memory(
        self, 
        agent: BaseModel,
        content: Any,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Store the step's result in agent memory.
        """
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
    """
    A sequence of cognitive steps forming an episode.
    """
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

    async def execute(self, agent: BaseModel) -> Union[str, Dict[str, Any]]:
        stm_cognitive = await agent.short_term_memory.retrieve_recent_memories(limit=10)
        short_term_memories = [
            {
                "cognitive_step": mem.cognitive_step,
                "content": mem.content,
                "metadata": mem.metadata
            }
            for mem in stm_cognitive
        ]

        # Debug: Color print short-term memories
        print("\033[94mShort-term Memories:\033[0m", short_term_memories)

        task_str = f"Task: {agent.task}" if agent.task else ""
        env_state_str = f"Environment state: {str(self.environment_info)}"
        agent_state = f"Agent state: {str(agent.last_action)} {str(agent.last_observation)}"
        query_str = (task_str + "\n" + env_state_str + "\n" + agent_state).strip()

        ltm_episodes = await agent.long_term_memory.retrieve_episodic_memories(
            agent_id=self.agent_id,
            query=query_str,
            top_k=3
        )

        # Debug: Color print long-term memories
        print("\033[92mLong-term Memories:\033[0m", [episode.model_dump() for episode in ltm_episodes])

        retrieved_documents = []
        if agent.knowledge_agent:
            retrieved_documents = await agent.knowledge_agent.retrieve(
                query_str,
                top_k=5
            )

        # Debug: Color print retrieved documents
        print("\033[93mRetrieved Documents:\033[0m", [doc.model_dump() for doc in retrieved_documents])

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


        if agent.chat_thread and self.structured_tool:
            agent.chat_thread.forced_output = perception_tool

        if agent.chat_thread:
            agent.chat_thread.new_message += perception_prompt

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
    """
    Decides the agent's next action using zero or more schemas/tools,
    then applies it in the environment.
    """
    step_name: str = "action"
    action_space: Optional[
        Union[
            Type[BaseModel],
            StructuredTool,
            CallableTool,
            List[Union[Type[BaseModel], StructuredTool, CallableTool]]
        ]
    ] = Field(
        default=None,
        description="Either a single or list of action schema(s)/tool(s)"
    )

    perception: Optional[str] = Field(
        default=None,
        description="Optional perception from the previous step"
    )

    async def execute(self, agent: BaseModel) -> Union[str, Dict[str, Any]]:
        """
        Decide on an action to execute in the environment,
        possibly using structured or callable tools.
        Then calls environment.step(...) with the resulting action.
        """
        environment = agent.environments[self.environment_name]
        action_space = environment.action_space if environment else None

        serialized_action_space = {
            "allowed_actions": [
                action_type.__name__
                for action_type in getattr(action_space, "allowed_actions", [])
            ],
            "constraints": getattr(action_space, "get_constraints", lambda: {})(),
        }

        variables = MarketAgentPromptVariables(
            environment_name=self.environment_name,
            environment_info=self.environment_info,
            perception=agent.last_perception,
            action_space=serialized_action_space,
            last_action=agent.last_action,
            observation=agent.last_observation
        )
        action_prompt = agent.prompt_manager.get_action_prompt(variables.model_dump())

        if agent.chat_thread and self.structured_tool:
            action_tool = StructuredTool(
                json_schema=action_space.get_action_schema(),
                name="react_reasoning",
                description="Generate thought-action-observation cycle"
            )
            agent.chat_thread.forced_output = action_tool

        if agent.chat_thread:
            agent.chat_thread.new_message += action_prompt

        if self.return_prompt:
            return action_prompt

        result = await agent.execute()

        await self.store_memory(
            agent,
            content=result,
            metadata={
                "action_space": serialized_action_space,
                "perception": agent.last_perception,
                "last_action": agent.last_action,
                "observation": agent.last_observation,
                "environment_name": self.environment_name,
                "environment_info": self.environment_info
            }
        )

        agent.last_action = result

        return result
    
class ReflectionStep(CognitiveStep):
    step_name: str = "reflection"

    async def execute(self, agent: BaseModel) -> Union[str, Dict[str, Any]]:
        environment = agent.environments[self.environment_name]
        last_step = (
            environment.history.steps[-1][1] 
            if environment.history.steps else None
        )

        if last_step:
            environment_reward = (
                last_step.info.get('agent_rewards', {}).get(self.agent_id, 0.0) or 0.0
            )
            local_observation = last_step.global_observation.observations.get(self.agent_id)
            observation = local_observation.observation if local_observation else {}
        else:
            observation = {}
            environment_reward = 0.0

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
            reward=environment_reward,
            previous_strategy=previous_strategy,
            perception=agent.last_perception
        )

        reflection_prompt = agent.prompt_manager.get_reflection_prompt(
            variables.model_dump()
        )

        if agent.chat_thread and self.structured_tool:
            agent.chat_thread.forced_output = reflection_tool

        if agent.chat_thread:
            agent.chat_thread.new_message += reflection_prompt

        if self.return_prompt:
            return reflection_prompt

        result = await agent.execute()
        
        if isinstance(result, dict):
            reward_data = agent.rl_agent.reward_function.compute(
                environment_reward=environment_reward,
                reflection_data=result,
                economic_value=agent.economic_agent.get_portfolio_value()
                              if agent.economic_agent else None
            )
            
            await agent.rl_agent.store_experience(
                state=agent.last_perception,
                action=agent.last_action,
                reward_data=reward_data,
                next_state=agent.last_observation,
                exploration_rate=agent.rl_agent.policy["exploration_rate"],
                created_at=datetime.now(timezone.utc)
            )
            
            agent.rl_agent.update_policy(reward_data["total_reward"])

            await self.store_memory(
                agent,
                content=result.get("reflection", ""),
                metadata={
                    **reward_data,
                    "last_action": agent.last_action,
                    "observation": agent.last_observation,
                    "self_critique": result.get("self_critique", []),
                    "strategy_update": result.get("strategy_update", ""),
                    "self_reward": result.get("self_reward", 0.0)
                }
            )

            await agent.long_term_memory.store_episode(
                task_query=agent.task if agent.task else None,
                steps=agent.episode_steps,
                total_reward=reward_data["total_reward"],
                strategy_update=result.get("strategy_update", ""),
                metadata={
                    "environemnt_state": self.environment_info,
                    "environment_name": self.environment_name
                }
            )
            agent.episode_steps.clear()

        return result