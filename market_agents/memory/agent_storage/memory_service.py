import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Union, Any
from uuid import UUID
import json
import logging

from market_agents.memory.agent_storage.setup_db import AsyncDatabase
from market_agents.memory.config import MarketMemoryConfig
from market_agents.memory.embedding import MemoryEmbedder
from market_agents.memory.knowledge_base import KnowledgeChunk, SemanticChunker
from market_agents.memory.memory import EpisodicMemoryObject, MemoryObject, CognitiveStep


class MemoryService:
    def __init__(self, db: AsyncDatabase, embedding_service: MemoryEmbedder, config: MarketMemoryConfig):
        self.db = db
        self.embedder = embedding_service
        self.config = config
        self.logger = logging.getLogger("memory_service")

    async def create_tables(self, table_type: str, agent_id: Optional[str] = None, table_prefix: Optional[str] = None):
        """
        Creates tables based on the specified type.

        Args:
            table_type: One of 'cognitive', 'episodic', or 'knowledge'
            agent_id: Required for cognitive/episodic tables
            table_prefix: Required for knowledge tables
        """
        try:
            if table_type == "cognitive":
                await self.create_agent_cognitive_memory_table(agent_id)
            elif table_type == "episodic":
                await self.create_agent_episodic_memory_table(agent_id)
            elif table_type == "knowledge":
                await self.create_knowledge_base_tables(table_prefix)
            else:
                raise ValueError(f"Invalid table type: {table_type}")
        except Exception as e:
            self.logger.error(f"Error creating {table_type} tables: {str(e)}")
            raise

    async def create_knowledge_base_tables(self, table_prefix: str) -> None:
        """Create separate tables for a specific knowledge base."""
        try:
            knowledge_objects_table = f"{table_prefix}_knowledge_objects"
            knowledge_chunks_table = f"{table_prefix}_knowledge_chunks"

            # Use a transaction to execute all queries atomically
            async with self.db.safe_transaction() as conn:
                # Create knowledge objects table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {knowledge_objects_table} (
                        knowledge_id UUID PRIMARY KEY,
                        content TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB DEFAULT '{{}}'::jsonb
                    );
                """)

                # Create knowledge chunks table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {knowledge_chunks_table} (
                        id SERIAL PRIMARY KEY,
                        knowledge_id UUID REFERENCES {knowledge_objects_table}(knowledge_id),
                        text TEXT,
                        start_pos INTEGER,
                        end_pos INTEGER,
                        embedding vector({self.config.vector_dim}),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # Add vector index for the chunks
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {table_prefix}_chunks_index
                    ON {knowledge_chunks_table} USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = {self.config.lists});
                """)

        except Exception as e:
            self.logger.error(f"Error creating knowledge base tables: {str(e)}")
            raise

    async def create_agent_cognitive_memory_table(self, agent_id: str) -> None:
        """
        Create a separate cognitive memory table for a specific agent.
        This can store single-step or short-horizon items (akin to 'STM').
        """
        try:
            sanitized_agent_id = self.db._sanitize_table_name(agent_id)
            cognitive_table = f"agent_{sanitized_agent_id}_cognitive"

            # Use a transaction to ensure atomic execution
            async with self.db.safe_transaction() as conn:
                # Create cognitive memory table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {cognitive_table} (
                        memory_id UUID PRIMARY KEY,
                        cognitive_step TEXT,
                        content TEXT,
                        embedding vector({self.config.vector_dim}),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB DEFAULT '{{}}'::jsonb
                    );
                """)

                # Create index for cognitive memory table
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS agent_{sanitized_agent_id}_cognitive_index
                    ON {cognitive_table} USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = {self.config.lists});
                """)

        except Exception as e:
            self.logger.error(f"Error creating agent cognitive memory table: {str(e)}")
            raise

    async def create_agent_episodic_memory_table(self, agent_id: str) -> None:
        """
        Create a separate episodic memory table for a specific agent.
        Each row will store an entire 'episode' (cognitive_steps in JSON),
        plus other relevant episodic info (task_query, total_reward, etc.).
        """
        try:
            sanitized_agent_id = self.db._sanitize_table_name(agent_id)
            episodic_table = f"agent_{sanitized_agent_id}_episodic"

            # Use a transaction to ensure atomic execution
            async with self.db.safe_transaction() as conn:
                # Create episodic memory table
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {episodic_table} (
                        memory_id UUID PRIMARY KEY,
                        task_query TEXT,
                        cognitive_steps JSONB,
                        total_reward DOUBLE PRECISION,
                        strategy_update JSONB,
                        embedding vector({self.config.vector_dim}),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB DEFAULT '{{}}'::jsonb
                    );
                """)

                # Create index for episodic memory table
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS agent_{sanitized_agent_id}_episodic_index
                    ON {episodic_table} USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = {self.config.lists});
                """)

        except Exception as e:
            self.logger.error(f"Error creating agent episodic memory table: {str(e)}")
            raise

    async def store_cognitive_memory(self, memory_object: MemoryObject) -> datetime:
        """Store a cognitive memory and return its creation timestamp."""
        try:
            if memory_object.embedding is None:
                memory_object.embedding = await self.embedder.get_embeddings(memory_object.content)

            cognitive_table = f"agent_{memory_object.agent_id}_cognitive"
            now = datetime.now(timezone.utc)

            result = await self.db.execute(
                f"""
                INSERT INTO {cognitive_table}
                (memory_id, cognitive_step, content, embedding, created_at, metadata)
                VALUES ($1, $2, $3, $4, $5, $6)
                RETURNING created_at
                """,
                str(memory_object.memory_id),
                memory_object.cognitive_step,
                memory_object.content,
                memory_object.embedding,
                now,
                json.dumps(memory_object.metadata)
            )

            return result[0] if result else now
        except Exception as e:
            self.logger.error(f"Error storing cognitive memory: {str(e)}")
            raise

    async def store_episodic_memory(self, episode: EpisodicMemoryObject) -> None:
        """Store an episodic memory."""
        try:
            if episode.embedding is None:
                concat_str = f"Task:{episode.task_query} + Steps:{episode.cognitive_steps}"
                episode.embedding = await self.embedder.get_embeddings(concat_str)

            episodic_table = f"agent_{episode.agent_id}_episodic"

            await self.db.execute(
                f"""
                INSERT INTO {episodic_table}
                (memory_id, task_query, cognitive_steps, total_reward,
                 strategy_update, embedding, created_at, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                str(episode.memory_id),
                episode.task_query,
                json.dumps([step.model_dump() for step in episode.cognitive_steps]),
                episode.total_reward,
                json.dumps(episode.strategy_update or []),
                episode.embedding,
                episode.created_at or datetime.now(timezone.utc),
                json.dumps(episode.metadata or {})
            )
        except Exception as e:
            self.logger.error(f"Error storing episodic memory: {str(e)}")
            raise

    async def get_cognitive_memory(
            self,
            agent_id: str,
            limit: int = 10,
            cognitive_step: Optional[Union[str, List[str]]] = None,
            metadata_filters: Optional[Dict] = None,
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None
    ) -> List[MemoryObject]:
        """Retrieve cognitive memories based on filters."""
        try:
            conditions = []
            params = []
            cognitive_table = f"agent_{agent_id}_cognitive"

            # Build query conditions
            if cognitive_step:
                if isinstance(cognitive_step, str):
                    conditions.append("cognitive_step = $" + str(len(params) + 1))
                    params.append(cognitive_step)
                else:
                    placeholders = [f"${i + 1}" for i in range(len(params), len(params) + len(cognitive_step))]
                    conditions.append(f"cognitive_step = ANY(ARRAY[{','.join(placeholders)}])")
                    params.extend(cognitive_step)

            # Add time and metadata filters...
            query = self._build_memory_query(cognitive_table, conditions, params, limit)

            rows = await self.db.fetch(query, *params)
            return [self._build_memory_object(row, agent_id) for row in rows]
        except Exception as e:
            self.logger.error(f"Error retrieving cognitive memory: {str(e)}")
            raise

    async def get_episodic_memory(
            self,
            agent_id: str,
            limit: int = 5,
            metadata_filters: Optional[Dict] = None,
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None
    ) -> List[EpisodicMemoryObject]:
        """Retrieve episodic memories based on filters."""
        try:
            episodic_table = f"agent_{agent_id}_episodic"
            conditions = []
            params = []

            # Build query conditions based on filters
            if metadata_filters:
                for key, value in metadata_filters.items():
                    conditions.append(f"metadata->>{key} = ${len(params) + 1}")
                    params.append(str(value))

            if start_time:
                conditions.append("created_at >= $" + str(len(params) + 1))
                params.append(start_time)

            if end_time:
                conditions.append("created_at <= $" + str(len(params) + 1))
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
                FROM {episodic_table}
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ${len(params) + 1};
            """
            params.append(limit)

            # Fetch the rows from the database
            rows = await self.db.fetch(query, *params)

            # Process each row and convert to EpisodicMemoryObject
            episodes = []
            for row in rows:
                (mem_id, task_query, steps_json, reward, strategy_json,
                 embedding, created_at, meta) = row

                # Parsing the JSON fields for cognitive_steps and strategy_update
                steps_list = json.loads(steps_json) if isinstance(steps_json, str) else steps_json or []
                strategy_list = json.loads(strategy_json) if isinstance(strategy_json, str) else strategy_json or []

                # Parse embedding from string (assuming it's a list of floats)
                embedding = [float(x) for x in embedding.strip('[]').split(',')] if isinstance(embedding,
                                                                                               str) else embedding

                # Convert cognitive steps to CognitiveStep objects
                csteps = [CognitiveStep(**step) for step in steps_list]

                # Create the EpisodicMemoryObject
                episode_obj = EpisodicMemoryObject(
                    memory_id=UUID(mem_id),
                    agent_id=agent_id,
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
            self.logger.error(f"Error retrieving episodic memory: {str(e)}")
            raise

    async def clear_agent_memory(self, agent_id: str, memory_type: Optional[str] = None) -> int:
        """Clear agent memories of specified type(s)."""
        try:
            deleted_count = 0
            if memory_type in (None, "cognitive"):
                deleted_count += await self.db.execute(
                    f"TRUNCATE TABLE agent_{agent_id}_cognitive;"
                )
            if memory_type in (None, "episodic"):
                deleted_count += await self.db.execute(
                    f"TRUNCATE TABLE agent_{agent_id}_episodic;"
                )
            return deleted_count
        except Exception as e:
            self.logger.error(f"Error clearing agent memory: {str(e)}")
            raise

    # Knowledge base methods...
    async def ingest_knowledge(self, text: str, metadata: Optional[Dict] = None, table_prefix: str = "default") -> UUID:
        """Ingest knowledge into the specified knowledge base."""
        try:
            chunks = self._chunk_text(text)
            embeddings = await self.embedder.get_embeddings([c.text for c in chunks])

            knowledge_id = uuid.uuid4()

            # Insert knowledge object
            await self.db.execute(
                f"""
                INSERT INTO {table_prefix}_knowledge_objects 
                (knowledge_id, content, metadata)
                VALUES ($1, $2, $3)
                """,
                str(knowledge_id),
                text,
                json.dumps(metadata or {})
            )

            # Insert chunks
            for chunk, embedding in zip(chunks, embeddings):
                await self.db.execute(
                    f"""
                    INSERT INTO {table_prefix}_knowledge_chunks
                    (knowledge_id, text, start_pos, end_pos, embedding)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    str(knowledge_id),
                    chunk.text,
                    chunk.start,
                    chunk.end,
                    embedding
                )

            return knowledge_id
        except Exception as e:
            self.logger.error(f"Error ingesting knowledge: {str(e)}")
            raise

    def _chunk_text(self, text: str) -> List[KnowledgeChunk]:
        """Internal method to chunk text using configured chunking method."""
        chunker = SemanticChunker(
            min_size=self.config.min_chunk_size,
            max_size=self.config.max_chunk_size
        )
        return chunker.chunk(text)

    async def search_knowledge(
            self,
            query: str,
            top_k: int = 5,
            table_prefix: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge base using semantic similarity.
        """
        try:
            prefix = table_prefix or "default"
            chunks_table = f"{prefix}_knowledge_chunks"
            objects_table = f"{prefix}_knowledge_objects"

            # Get embedding for the query
            query_embedding = await self.embedder.get_embeddings(query)

            # Perform vector similarity search
            results = await self.db.fetch(f"""
                   WITH similarity_matches AS (
                       SELECT 
                           c.knowledge_id,
                           c.text as chunk_text,
                           c.embedding <-> $1 as similarity,
                           o.content as full_content,
                           o.metadata
                       FROM {chunks_table} c
                       JOIN {objects_table} o ON c.knowledge_id = o.knowledge_id
                       ORDER BY similarity ASC
                       LIMIT $2
                   )
                   SELECT * FROM similarity_matches;
               """, query_embedding, top_k)

            return [{
                "knowledge_id": str(row["knowledge_id"]),
                "chunk_text": row["chunk_text"],
                "similarity_score": row["similarity"],
                "full_content": row["full_content"],
                "metadata": row["metadata"]
            } for row in results]
        except Exception as e:
            self.logger.error(f"Error searching knowledge base: {str(e)}")
            raise

    async def delete_knowledge(
            self,
            knowledge_id: UUID,
            table_prefix: Optional[str] = None
    ) -> bool:
        """
        Delete a knowledge entry and its associated chunks.
        Returns True if successful.
        """
        try:
            prefix = table_prefix or "default"
            chunks_table = f"{prefix}_knowledge_chunks"
            objects_table = f"{prefix}_knowledge_objects"

            # Use the safe_transaction to ensure retryable, atomic behavior
            async with self.db.safe_transaction() as conn:
                # Delete chunks first due to foreign key constraint
                await conn.execute(
                    f"DELETE FROM {chunks_table} WHERE knowledge_id = $1",
                    str(knowledge_id)
                )

                # Delete the main knowledge object
                result = await conn.execute(
                    f"DELETE FROM {objects_table} WHERE knowledge_id = $1",
                    str(knowledge_id)
                )

                return result == "DELETE 1"
        except Exception as e:
            self.logger.error(f"Error deleting knowledge: {str(e)}")
            raise

    def _build_memory_query(self, table: str, conditions: List[str], params: List[Any], limit: int) -> str:
        """Helper to build memory query with conditions."""
        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        return f"""
            WITH recent_memories AS (
                SELECT *
                FROM {table}
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ${len(params) + 1}
            )
            SELECT * FROM recent_memories
            ORDER BY created_at ASC;
        """

    def _build_memory_object(self, row: Dict[str, Any], agent_id: str) -> MemoryObject:
        """Convert a database row to a MemoryObject."""
        try:
            memory_id = UUID(row["memory_id"])
            cognitive_step = row["cognitive_step"]
            content = row["content"]
            embedding = row["embedding"]  # This would typically be a vector, or a string representation
            created_at = row["created_at"]
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}

            return MemoryObject(
                memory_id=memory_id,
                agent_id=agent_id,
                cognitive_step=cognitive_step,
                content=content,
                embedding=embedding,
                created_at=created_at,
                metadata=metadata
            )
        except KeyError as e:
            self.logger.error(f"Missing expected column in cognitive memory row: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error building MemoryObject: {str(e)}")
            raise

    def _build_episodic_object(self, row: Dict[str, Any], agent_id: str) -> EpisodicMemoryObject:
        """Convert a database row to an EpisodicMemoryObject."""
        try:
            memory_id = UUID(row["memory_id"])
            task_query = row["task_query"]
            cognitive_steps = json.loads(row["cognitive_steps"])  # Assuming it's stored as JSON in the DB
            total_reward = row["total_reward"]
            strategy_update = json.loads(row["strategy_update"]) if row["strategy_update"] else []
            embedding = row["embedding"]
            created_at = row["created_at"]
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}

            # Construct and return the EpisodicMemoryObject
            return EpisodicMemoryObject(
                memory_id=memory_id,
                agent_id=agent_id,
                task_query=task_query,
                cognitive_steps=cognitive_steps,  # List of cognitive steps
                total_reward=total_reward,
                strategy_update=strategy_update,
                embedding=embedding,
                created_at=created_at,
                metadata=metadata
            )
        except KeyError as e:
            self.logger.error(f"Missing expected column in episodic memory row: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error building EpisodicMemoryObject: {str(e)}")
            raise