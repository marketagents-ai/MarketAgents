import asyncio
import json
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Union
from uuid import UUID

from pydantic import BaseModel, Field, ConfigDict

from market_agents.memory.agent_storage.agent_storage_api import CognitiveMemoryParams
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.memory_models import MemoryObject, EpisodicMemoryObject, CognitiveStep


class ShortTermMemory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    items_cache: List[MemoryObject] = Field(default_factory=list)
    agent_storage_utils: AgentStorageAPIUtils

    def __init__(self, agent_id: str, agent_storage_utils: AgentStorageAPIUtils):
        super().__init__(agent_storage_utils=agent_storage_utils)


    async def _store_memory_sync(self, memory_object: MemoryObject):
        """
        Synchronous method that calls `store_cognitive_item` and updates items_cache.
        """
        await self.agent_storage_utils.store_cognitive_memory(memory_object)
        self.items_cache.append(memory_object)

    async def retrieve_recent_memories(
        self,
        limit: int = 10,
        cognitive_step: Optional[Union[str, List[str]]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[MemoryObject]:
        memories = await self.agent_storage_utils.get_cognitive_memory(
            self.agent_id,
            CognitiveMemoryParams(
                limit=limit,
                cognitive_step=cognitive_step,
                metadata_filters=metadata_filters,
                start_time=start_time,
                end_time=end_time
            )
        )
        return memories["memories"]

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
    agent_storage_utils: AgentStorageAPIUtils

    def __init__(self, agent_storage_utils: AgentStorageAPIUtils):

        super().__init__(
            agent_storage_utils=agent_storage_utils
        )

    async def store_episodic_memory(
        self,
        agent_id: str,
        task_query: str,
        steps: List[CognitiveStep],
        total_reward: Optional[float] = None,
        strategy_update: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        An async wrapper that calls `_store_episodic_memory_sync` via run_in_executor
        so we can schedule it with create_task(...).
        """
        await self.agent_storage_utils.store_episodic_memory(
            EpisodicMemoryObject(
                agent_id=agent_id,
                task_query=task_query,
                cognitive_steps=steps,
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
        loop = asyncio.get_event_loop()
        retrieved = await loop.run_in_executor(
            None,
            lambda: self.memory_retriever.search_agent_episodic_memory(
                agent_id=agent_id,
                query=query,
                top_k=top_k
            )
        )
        episodes = []
        for memory_item in retrieved:
            content_dict = json.loads(memory_item.text)
            steps_json = content_dict.get("cognitive_steps", [])
            step_objs = [CognitiveStep(**step) for step in steps_json]
            
            created_at_str = content_dict.get("created_at")
            created_at = datetime.fromisoformat(created_at_str)

            eobj = EpisodicMemoryObject(
                memory_id=UUID(content_dict["memory_id"]),
                agent_id=agent_id,
                task_query=content_dict.get("task_query", ""),
                cognitive_steps=step_objs,
                total_reward=content_dict.get("total_reward"),
                strategy_update=content_dict.get("strategy_update"),
                created_at=created_at,
                metadata=content_dict.get("metadata", {}),
                similarity=round(memory_item.similarity, 2)
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
            agent_id,
            "episodic"
        )

        return memories_deleted["deleted_count"]
