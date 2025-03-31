import asyncio
import json
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Union
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict

from market_agents.memory.agent_storage.agent_storage_api import CognitiveMemoryParams
from market_agents.memory.storage_models import (
    CreateTablesRequest,
    MemoryObject, 
    EpisodicMemoryObject, 
    CognitiveStep,
    CognitiveMemoryParams
)
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils


class ShortTermMemory(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=()
    )

    items_cache: List[MemoryObject] = Field(default_factory=list)
    agent_storage_utils: AgentStorageAPIUtils
    agent_id: str
    default_top_k: int = Field(default=2)

    def __init__(
        self,
        agent_id: str,
        agent_storage_utils: AgentStorageAPIUtils,
        default_top_k: int = 2
    ):
        super().__init__(
            agent_storage_utils=agent_storage_utils,
            agent_id=agent_id,
            default_top_k=default_top_k
        )

    async def initialize(self):
        """Initialize the cognitive memory tables for this agent."""
        await self.agent_storage_utils.create_tables(CreateTablesRequest(
            table_type="cognitive",
            agent_id=self.agent_id
        ))


    async def store_memory(self, memory_object: MemoryObject):
        """
        Method that calls `store_cognitive_item` and updates items_cache.
        """
        await self.agent_storage_utils.store_cognitive_memory(memory_object)
        self.items_cache.append(memory_object)

    async def retrieve_recent_memories(
        self,
        limit: int = 5,
        cognitive_step: Optional[str] = None,
        metadata_filters: Optional[Dict] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[MemoryObject]:
        memories = await self.agent_storage_utils.get_cognitive_memory_sql(
            self.agent_id,
            CognitiveMemoryParams(
                limit=limit if limit is not None else self.default_top_k,
                cognitive_step=cognitive_step,
                metadata_filters=metadata_filters,
                start_time=start_time,
                end_time=end_time
            )
        )
        return memories 

    async def clear_memories(
        self,
        cognitive_step: Optional[Union[str, List[str]]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> int:
        memory_count = await self.agent_storage_utils.clear_agent_memory(
            self.agent_id,
            "cognitive",
        )
        return memory_count["deleted_count"]

class LongTermMemory(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        protected_namespaces=()
    )
    agent_storage_utils: AgentStorageAPIUtils
    agent_id: str
    default_top_k: int = Field(default=1)


    def __init__(
        self,
        agent_id: str,
        agent_storage_utils: AgentStorageAPIUtils,
        default_top_k: int = 1
    ):
        super().__init__(
            agent_storage_utils=agent_storage_utils,
            agent_id=agent_id,
            default_top_k=default_top_k
        )

    async def initialize(self):
        """Initialize the episodic memory tables for this agent."""
        await self.agent_storage_utils.create_tables(CreateTablesRequest(
            table_type="episodic",
            agent_id=self.agent_id
        ))

    async def store_episode(
        self,
        task_query: str,
        steps: List[MemoryObject],
        total_reward: Optional[float] = None,
        strategy_update: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Store an episode in long-term memory.
        Converts MemoryObject instances to CognitiveStep instances before storage.
        """
        cognitive_steps = []
        for step in steps:
            if isinstance(step.content, dict):
                content_dict = step.content
            else:
                try:
                    content_dict = json.loads(step.content)
                    if isinstance(content_dict, str):
                        content_dict = json.loads(content_dict)
                except (json.JSONDecodeError, TypeError):
                    content_dict = {"text": step.content}

            cognitive_step = CognitiveStep(
                step=step.cognitive_step,
                step_type="cognitive",
                content=content_dict,
                metadata=step.metadata,
                timestamp=step.created_at
            )
            cognitive_steps.append(cognitive_step)

        await self.agent_storage_utils.store_episodic_memory(
            EpisodicMemoryObject(
                agent_id=self.agent_id,
                task_query=task_query,
                cognitive_steps=cognitive_steps,
                total_reward=total_reward,
                strategy_update=strategy_update,
                metadata=metadata
            )
        )

    async def retrieve_episodic_memories(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5
    ) -> List[EpisodicMemoryObject]:
        """Retrieve episodic memories using semantic search."""
        retrieved = await self.agent_storage_utils.get_episodic_memory_vector(
            agent_id=self.agent_id,
            top_k=top_k if top_k is not None else self.default_top_k,
            query=query
        )
        
        episodes = []
        for memory_item in retrieved["matches"]:
            content_dict = json.loads(memory_item["text"])
            
            # Parse cognitive steps if they're stored as a string
            cognitive_steps = content_dict.get("cognitive_steps", [])
            if isinstance(cognitive_steps, str):
                try:
                    cognitive_steps = json.loads(cognitive_steps)
                except json.JSONDecodeError:
                    cognitive_steps = []
            
            step_objs = []
            for step in cognitive_steps:
                if isinstance(step, dict):
                    step_objs.append(CognitiveStep(**step))
            
            strategy_update = content_dict.get("strategy_update")
            if isinstance(strategy_update, str):
                try:
                    strategy_update = json.loads(strategy_update)
                except json.JSONDecodeError:
                    strategy_update = []
            
            metadata = content_dict.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}
            
            created_at_str = content_dict.get("created_at")
            created_at = datetime.fromisoformat(created_at_str) if created_at_str else datetime.now(timezone.utc)

            eobj = EpisodicMemoryObject(
                memory_id=UUID(content_dict["memory_id"]),
                agent_id=self.agent_id,
                task_query=content_dict.get("task_query", ""),
                cognitive_steps=step_objs,
                total_reward=content_dict.get("total_reward"),
                strategy_update=strategy_update,
                created_at=created_at,
                metadata=metadata,
                similarity=round(memory_item["similarity"], 2)
            )
            episodes.append(eobj)

        return episodes

    async def get_episodic_memories(
        self,
        agent_id: str,
        limit: int = 10,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[EpisodicMemoryObject]:
        """Retrieve episodic memories using SQL filters."""
        retrieved = await self.agent_storage_utils.get_episodic_memory_sql(
            agent_id=self.agent_id,
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            metadata_filters=metadata_filters
        )
        
        episodes = []
        for memory_item in retrieved["memories"]:
            steps_json = memory_item.get("cognitive_steps", [])
            
            step_objs = []
            for step in steps_json:
                if isinstance(step, str):
                    try:
                        step_dict = json.loads(step)
                    except json.JSONDecodeError:
                        step_dict = {"step": "unknown", "step_type": "cognitive", "content": {"text": step}}
                else:
                    step_dict = step
                
                if "step_type" not in step_dict:
                    step_dict["step_type"] = "cognitive"
                
                step_objs.append(CognitiveStep(**step_dict))
            
            eobj = EpisodicMemoryObject(
                memory_id=UUID(memory_item["memory_id"]),
                agent_id=self.agent_id,
                task_query=memory_item.get("task_query", ""),
                cognitive_steps=step_objs,
                total_reward=memory_item.get("total_reward"),
                strategy_update=memory_item.get("strategy_update"),
                created_at=memory_item.get("created_at"),
                metadata=memory_item.get("metadata", {})
            )
            episodes.append(eobj)

        return episodes

    async def delete_episodic_memory(
        self,
        agent_id: str,
        task_query: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> int:
        memories_deleted = await self.agent_storage_utils.clear_agent_memory(
            self.agent_id,
            "episodic"
        )

        return memories_deleted["deleted_count"]
