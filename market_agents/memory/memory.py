import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Union
from uuid import UUID

from fastapi import logger
from pydantic import BaseModel, Field, ConfigDict

from market_agents.memory.embedding import MemoryEmbedder
from market_agents.memory.config import MarketMemoryConfig
from market_agents.memory.setup_db import DatabaseConnection
from market_agents.memory.vector_search import MemoryRetriever


class MemoryObject(BaseModel):
    """A single step of cognitive memory."""
    memory_id: UUID = Field(default_factory=uuid.uuid4)
    agent_id: str
    cognitive_step: str
    content: str
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def serialize_metadata(self) -> Optional[str]:
            """Serialize metadata to JSON string, handling datetime objects"""
            if not self.metadata:
                return None
                
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

            serialized_metadata = {k: serialize_value(v) for k, v in self.metadata.items()}
            return json.dumps(serialized_metadata)


class CognitiveStep(BaseModel):
    """
    A single step stored *inside* an EpisodicMemoryObject.
    """
    step_type: str = Field(..., description="Type of cognitive step (e.g., 'perception')")
    content: Dict[str, Any] = Field(..., description="Content of the step")


class EpisodicMemoryObject(BaseModel):
    """
    An entire 'episode' of task execution. 
    It aggregates multiple steps (CognitiveStep),
    plus optional fields like total_reward, strategy_update, etc.
    """
    memory_id: UUID = Field(default_factory=uuid.uuid4)
    agent_id: str
    task_query: str
    cognitive_steps: List[CognitiveStep] = Field(default_factory=list)
    total_reward: Optional[float] = None
    strategy_update: Optional[List[str]] = None
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None
    similarity: Optional[float] = None

class BaseMemory:
    """Base class with common initialization and sanitization"""
    
    def __init__(self, config: MarketMemoryConfig, db_conn: DatabaseConnection, 
                 embedder: MemoryEmbedder, agent_id: str):
        self.config = config
        self.db = db_conn
        self.embedder = embedder
        self.agent_id = agent_id
        self.safe_id = self._sanitize_id(agent_id)
        
        # Initialize database connection
        self._initialize_database()

    @staticmethod
    def _sanitize_id(agent_id: str) -> str:
        """Sanitize agent ID for table names"""
        return agent_id.replace('-', '_')

    def _initialize_database(self):
        """Ensure database connection is ready"""
        self.db.connect()

class CognitiveMemory(BaseMemory):
    """
    Handles storing and retrieving single-step memory items in agent_{agent_id}_cognitive table.
    """

    def __init__(
        self,
        config: MarketMemoryConfig,
        db_conn: DatabaseConnection,
        embedder: MemoryEmbedder,
        agent_id: str
    ):
        BaseMemory.__init__(self, config, db_conn, embedder, agent_id)
        self.cognitive_table = f"agent_{self.safe_id}_cognitive"
        self.db.create_agent_cognitive_memory_table(self.agent_id)

    def store_cognitive_item(self, memory_object: MemoryObject):
        """
        Insert a single cognitive step into the 'cognitive' table,
        returning the DB's actual created_at timestamp.
        """
        self.db.connect()
        try:
            if memory_object.embedding is None:
                memory_object.embedding = self.embedder.get_embeddings(memory_object.content)

            now = memory_object.created_at or datetime.now(timezone.utc)

            self.db.cursor.execute(f"""
                INSERT INTO {self.cognitive_table}
                (memory_id, cognitive_step, content, embedding, created_at, metadata)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING created_at
            """, (
                str(memory_object.memory_id),
                memory_object.cognitive_step,
                memory_object.content,
                memory_object.embedding,
                now,
                memory_object.serialize_metadata()
            ))
            row = self.db.cursor.fetchone()
            if row:
                memory_object.created_at = row[0]
            else:
                memory_object.created_at = now

            self.db.conn.commit()
        except Exception as e:
            self.db.conn.rollback()
            raise e

    def get_cognitive_items(
        self,
        limit: int = 10,
        cognitive_step: Optional[Union[str, List[str]]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[MemoryObject]:
        """
        Retrieve a list of single-step memory items (short-term) from the agent's cognitive table.
        """
        self.db.connect()

        conditions = []
        params = []

        if cognitive_step:
            if isinstance(cognitive_step, str):
                conditions.append("cognitive_step = %s")
                params.append(cognitive_step)
            else:
                placeholders = ", ".join(["%s"] * len(cognitive_step))
                conditions.append(f"cognitive_step IN ({placeholders})")
                params.extend(cognitive_step)

        if metadata_filters:
            for k, v in metadata_filters.items():
                conditions.append("metadata->>%s = %s")
                params.append(k)
                params.append(str(v))

        if start_time:
            conditions.append("created_at >= %s")
            params.append(start_time)

        if end_time:
            conditions.append("created_at <= %s")
            params.append(end_time)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        params.append(limit)
        
        query = f"""
            WITH recent_memories AS (
                SELECT 
                    memory_id, 
                    cognitive_step, 
                    content,
                    embedding, 
                    created_at, 
                    metadata
                FROM {self.cognitive_table}
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT %s
            )
            SELECT * FROM recent_memories
            ORDER BY created_at ASC;
        """

        try:
            self.db.cursor.execute(query, tuple(params))
            rows = self.db.cursor.fetchall()

            items = []
            for row in rows:
                if len(row) != 6:
                    continue

                mem_id, step, content, embedding, created_at, meta = row
                if isinstance(embedding, str):
                    embedding = [float(x) for x in embedding.strip('[]').split(',')]

                mo = MemoryObject(
                    memory_id=UUID(mem_id),
                    agent_id=self.agent_id,
                    cognitive_step=step,
                    content=content,
                    embedding=embedding,
                    created_at=created_at,
                    metadata=meta if meta else {}
                )
                items.append(mo)
            return items
        except Exception as e:
            raise e

    def delete_cognitive_items(
        self,
        cognitive_step: Optional[Union[str, List[str]]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> int:
        """
        Delete rows from the short-term cognitive memory based on filters.
        Returns how many rows were deleted.
        """
        self.db.connect()

        conditions = []
        params = []

        if cognitive_step:
            if isinstance(cognitive_step, str):
                conditions.append("cognitive_step = %s")
                params.append(cognitive_step)
            else:
                placeholders = ", ".join(["%s"] * len(cognitive_step))
                conditions.append(f"cognitive_step IN ({placeholders})")
                params.extend(cognitive_step)

        if metadata_filters:
            for k, v in metadata_filters.items():
                conditions.append("metadata->>%s = %s")
                params.append(k)
                params.append(str(v))

        if start_time:
            conditions.append("created_at >= %s")
            params.append(start_time)

        if end_time:
            conditions.append("created_at <= %s")
            params.append(end_time)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        try:
            self.db.cursor.execute(
                f"DELETE FROM {self.cognitive_table} WHERE {where_clause} RETURNING *;",
                tuple(params)
            )
            deleted_count = self.db.cursor.rowcount
            self.db.conn.commit()
            return deleted_count
        except Exception as e:
            self.db.conn.rollback()
            raise e


class EpisodicMemory(BaseMemory):
    """
    Manages the 'episodic' memory table named agent_{agent_id}_episodic.
    Each row represents a full 'episode' containing multiple steps.
    """

    def __init__(
        self,
        config: MarketMemoryConfig,
        db_conn: DatabaseConnection,
        embedder: MemoryEmbedder,
        agent_id: str
    ):
        BaseMemory.__init__(self, config, db_conn, embedder, agent_id)
        self.episodic_table = f"agent_{self.safe_id}_episodic"
        self.db.create_agent_episodic_memory_table(self.agent_id)

    def store_episode(self, episode: EpisodicMemoryObject):
        """
        Insert an entire 'episode' in agent_{agent_id}_episodic.
        """
        self.db.connect()
        try:
            # If no embedding was provided, derive from the task_query + cognitive steps
            if episode.embedding is None:
                concat_str = f"Task:{episode.task_query} + Steps:{episode.cognitive_steps}"
                episode.embedding = self.embedder.get_embeddings(concat_str)

            step_data = [step.dict() for step in episode.cognitive_steps]
            strategy_data = episode.strategy_update if episode.strategy_update else []
            meta = episode.metadata if episode.metadata else {}

            now = episode.created_at or datetime.now(timezone.utc)

            self.db.cursor.execute(f"""
                INSERT INTO {self.episodic_table}
                (memory_id, task_query, cognitive_steps, total_reward,
                 strategy_update, embedding, created_at, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """, (
                str(episode.memory_id),
                episode.task_query,
                json.dumps(step_data),
                episode.total_reward,
                json.dumps(strategy_data),
                episode.embedding,
                now,
                json.dumps(meta),
            ))
            self.db.conn.commit()
        except Exception as e:
            self.db.conn.rollback()
            raise e

    def get_episodes(
        self,
        limit: int = 5,
        metadata_filters: Optional[Dict[str, Any]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[EpisodicMemoryObject]:
        """
        Retrieve episodes in descending order of created_at.
        """
        self.db.connect()

        conditions = []
        params = []

        if metadata_filters:
            for k, v in metadata_filters.items():
                conditions.append("metadata->>%s = %s")
                params.append(k)
                params.append(str(v))

        if start_time:
            conditions.append("created_at >= %s")
            params.append(start_time)

        if end_time:
            conditions.append("created_at <= %s")
            params.append(end_time)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        query = f"""
            SELECT 
                memory_id, 
                task_query, 
                cognitive_steps, 
                total_reward,
                strategy_update, 
                embedding, 
                created_at, 
                metadata
            FROM {self.episodic_table}
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT %s;
        """
        params.append(limit)

        try:
            self.db.cursor.execute(query, tuple(params))
            rows = self.db.cursor.fetchall()

            episodes = []
            for row in rows:
                (mem_id, task_query, steps_json, reward, strategy_json,
                 embedding, created_at, meta) = row

                if isinstance(steps_json, str):
                    steps_list = json.loads(steps_json)
                else:
                    steps_list = steps_json or []

                if isinstance(strategy_json, str):
                    strategy_list = json.loads(strategy_json)
                else:
                    strategy_list = strategy_json or []

                if isinstance(embedding, str):
                    embedding = [float(x) for x in embedding.strip('[]').split(',')]

                csteps = [CognitiveStep(**step) for step in steps_list]

                episode_obj = EpisodicMemoryObject(
                    memory_id=UUID(mem_id),
                    agent_id=self.agent_id,
                    task_query=task_query,
                    cognitive_steps=csteps,
                    total_reward=reward,
                    strategy_update=strategy_list,
                    embedding=embedding,
                    created_at=created_at,
                    metadata=meta if meta else {}
                )
                episodes.append(episode_obj)
            return episodes
        except Exception as e:
            raise e

    def delete_episodes(
        self,
        task_query: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> int:
        """
        Delete entire episodes based on optional filters.
        Returns how many rows were deleted.
        """
        self.db.connect()

        conditions = []
        params = []

        if task_query:
            conditions.append("task_query = %s")
            params.append(task_query)

        if start_time:
            conditions.append("created_at >= %s")
            params.append(start_time)

        if end_time:
            conditions.append("created_at <= %s")
            params.append(end_time)

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        try:
            self.db.cursor.execute(
                f"DELETE FROM {self.episodic_table} WHERE {where_clause} RETURNING *;",
                tuple(params)
            )
            deleted_count = self.db.cursor.rowcount
            self.db.conn.commit()
            return deleted_count
        except Exception as e:
            self.db.conn.rollback()
            raise e

class ShortTermMemory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    cognitive_memory: CognitiveMemory
    items_cache: List[MemoryObject] = Field(default_factory=list)

    def __init__(self, memory_config: MarketMemoryConfig, db_conn: DatabaseConnection, agent_id: str):
        super().__init__(
            cognitive_memory=CognitiveMemory(memory_config, db_conn, MemoryEmbedder(memory_config), agent_id)
        )

    async def store_memory(self, memory_object: MemoryObject):
        """
        Asynchronously store memory by offloading the synchronous DB call
        to a thread pool. Use run_in_executor so we can call create_task().
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._store_memory_sync, memory_object)

    def _store_memory_sync(self, memory_object: MemoryObject):
        """
        Synchronous method that calls `store_cognitive_item` and updates items_cache.
        """
        self.cognitive_memory.store_cognitive_item(memory_object)
        self.items_cache.append(memory_object)

    async def retrieve_recent_memories(
        self,
        limit: int = 10,
        cognitive_step: Optional[Union[str, List[str]]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[MemoryObject]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.cognitive_memory.get_cognitive_items(
                limit=limit,
                cognitive_step=cognitive_step,
                metadata_filters=metadata_filters,
                start_time=start_time,
                end_time=end_time
            )
        )

    async def clear_memories(
        self,
        cognitive_step: Optional[Union[str, List[str]]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> int:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.cognitive_memory.delete_cognitive_items(
                cognitive_step=cognitive_step,
                metadata_filters=metadata_filters,
                start_time=start_time,
                end_time=end_time
            )
        )

class LongTermMemory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    memory_retriever: MemoryRetriever
    episodic_store: EpisodicMemory

    def __init__(self, memory_config: MarketMemoryConfig, db_conn: DatabaseConnection, agent_id: str):
        embedder = MemoryEmbedder(memory_config)
        super().__init__(
            memory_retriever=MemoryRetriever(config=memory_config, db_conn=db_conn, embedding_service=embedder),
            episodic_store=EpisodicMemory(memory_config, db_conn, embedder, agent_id)
        )

    async def store_episodic_memory(
        self,
        agent_id: str,
        task_query: str,
        steps: List[MemoryObject],
        total_reward: Optional[float] = None,
        strategy_update: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        An async wrapper that calls `_store_episodic_memory_sync` via run_in_executor
        so we can schedule it with create_task(...).
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._store_episodic_memory_sync,
            agent_id,
            task_query,
            steps,
            total_reward,
            strategy_update,
            metadata
        )

    def _store_episodic_memory_sync(
        self,
        agent_id: str,
        task_query: str,
        steps: List[MemoryObject],
        total_reward: Optional[float],
        strategy_update: Optional[List[str]],
        metadata: Optional[Dict[str, Any]]
    ):
        """
        The synchronous logic that builds EpisodicMemoryObject and calls store_episode(...).
        """

        csteps = []
        for step in steps:
            raw_content = (step.content or "").strip()
            try:
                if not raw_content:
                    parsed_content = {}
                else:
                    parsed_content = json.loads(raw_content)

                if not isinstance(parsed_content, dict):
                    parsed_content = {}
            except (json.JSONDecodeError, TypeError):
                parsed_content = {}

            csteps.append(
                CognitiveStep(
                    step_type=step.cognitive_step,
                    content=parsed_content
                )
            )

        episode = EpisodicMemoryObject(
            agent_id=agent_id,
            task_query=task_query,
            cognitive_steps=csteps,
            total_reward=total_reward,
            strategy_update=strategy_update,
            metadata=metadata,
            created_at=datetime.now(timezone.utc)
        )
        self.episodic_store.store_episode(episode)


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
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.episodic_store.delete_episodes(
                task_query=task_query,
                start_time=start_time,
                end_time=end_time
            )
        )
