import json
import logging
import time
from typing import List, Optional, Dict, Any, Union
from uuid import UUID
from datetime import datetime, timezone
import uuid

from market_agents.memory.agent_storage.setup_db import AsyncDatabase
from market_agents.memory.config import AgentStorageConfig
from market_agents.memory.embedding import MemoryEmbedder
from market_agents.memory.knowledge_base import KnowledgeChunk, SemanticChunker
from market_agents.memory.storage_models import (
    EpisodicMemoryObject,
    MemoryObject,
    CognitiveStep,
    RetrievedMemory
)

class StorageService:
    def __init__(self, db: AsyncDatabase, embedding_service: MemoryEmbedder, config: AgentStorageConfig):
        self.db = db
        self.embedder = embedding_service
        self.config = config
        self.logger = logging.getLogger("storage_service")

    async def create_tables(self, table_type: str, agent_id: Optional[str] = None, table_prefix: Optional[str] = None):
        """Creates tables based on the specified type."""
        try:
            # Verify database connection
            if not self.db.pool:
                await self.db.initialize()

            # Add validation
            if table_type in ["cognitive", "episodic"] and not agent_id:
                raise ValueError(f"agent_id is required for {table_type} tables")
            if table_type == "knowledge" and not table_prefix:
                raise ValueError("table_prefix is required for knowledge tables")

            self.logger.info(f"Creating {table_type} tables with agent_id={agent_id}, prefix={table_prefix}")

            # Create pgvector extension first
            async with self.db.safe_transaction() as conn:
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                self.logger.info("Created pgvector extension")

            # Then create the specific tables
            if table_type == "cognitive":
                await self.create_agent_cognitive_memory_table(agent_id)
            elif table_type == "episodic":
                await self.create_agent_episodic_memory_table(agent_id)
            elif table_type == "knowledge":
                await self.create_knowledge_base_tables(table_prefix)
            elif table_type == "ai_requests":
                await self.create_ai_requests_table()
            else:
                raise ValueError(f"Invalid table type: {table_type}")
                
            self.logger.info(f"Successfully created {table_type} tables")
        except Exception as e:
            self.logger.error(f"Error creating {table_type} tables: {str(e)}")
            raise

    async def create_ai_requests_table(self) -> None:
        """Create the AI requests table."""
        try:
            async with self.db.safe_transaction() as conn:
                # Create AI requests table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS ai_requests (
                        id SERIAL PRIMARY KEY,
                        request_id TEXT NOT NULL,
                        agent_id TEXT,
                        prompt TEXT NOT NULL,
                        response JSONB,
                        metadata JSONB DEFAULT '{}',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(request_id)
                    )
                """)
                
                # Create indexes
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ai_requests_request_id ON ai_requests(request_id);
                    CREATE INDEX IF NOT EXISTS idx_ai_requests_agent_id ON ai_requests(agent_id);
                    CREATE INDEX IF NOT EXISTS idx_ai_requests_created_at ON ai_requests(created_at);
                """)
        except Exception as e:
            self.logger.error(f"Error creating AI requests table: {str(e)}")
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
            # Sanitize the agent_id to create a valid table name
            sanitized_agent_id = self.db._sanitize_table_name(agent_id)
            cognitive_table = f"agent_{sanitized_agent_id}_cognitive"
            index_name = f"agent_{sanitized_agent_id}_cognitive_index"

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
                    CREATE INDEX IF NOT EXISTS {index_name}
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
            # Sanitize the agent_id to create a valid table name
            sanitized_agent_id = self.db._sanitize_table_name(agent_id)
            episodic_table = f"agent_{sanitized_agent_id}_episodic"
            index_name = f"agent_{sanitized_agent_id}_episodic_index"

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
                    CREATE INDEX IF NOT EXISTS {index_name}
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
            
            # Convert embedding list to PostgreSQL vector string format
            vector_str = f"[{','.join(map(str, memory_object.embedding))}]"

            cognitive_table = f"agent_{memory_object.agent_id}_cognitive"
            now = datetime.now(timezone.utc)

            result = await self.db.execute(
                f"""
                INSERT INTO {cognitive_table}
                (memory_id, cognitive_step, content, embedding, created_at, metadata)
                VALUES ($1, $2, $3, $4::vector, $5, $6)
                RETURNING created_at
                """,
                str(memory_object.memory_id),
                memory_object.cognitive_step,
                memory_object.content,
                vector_str,
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
            
            # Convert embedding list to PostgreSQL vector string format
            vector_str = f"[{','.join(map(str, episode.embedding))}]"

            episodic_table = f"agent_{episode.agent_id}_episodic"

            await self.db.execute(
                f"""
                INSERT INTO {episodic_table}
                (memory_id, task_query, cognitive_steps, total_reward,
                 strategy_update, embedding, created_at, metadata)
                VALUES ($1, $2, $3, $4, $5, $6::vector, $7, $8)
                """,
                str(episode.memory_id),
                episode.task_query,
                json.dumps([step.model_dump() for step in episode.cognitive_steps]),
                episode.total_reward,
                json.dumps(episode.strategy_update or []),
                vector_str,
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
                    conditions.append("cognitive_step = $1")
                    params.append(cognitive_step)
                elif cognitive_step:
                    placeholders = ", ".join(
                        [f"${i + 1}" for i in range(len(params) + 1, len(params) + 1 + len(cognitive_step))])
                    conditions.append(f"cognitive_step IN ({placeholders})")
                    params.extend(cognitive_step)

            if metadata_filters:
                for k, v in metadata_filters.items():
                    conditions.append(f"metadata->>{k} = ${len(params) + 1}")
                    params.append(str(v))

            if start_time:
                conditions.append(f"created_at >= ${len(params) + 1}")
                params.append(start_time)

            if end_time:
                conditions.append(f"created_at <= ${len(params) + 1}")
                params.append(end_time)

            where_clause = " AND ".join(conditions) if conditions else "TRUE"
            params.append(limit)

            query = f"""
                    WITH recent_memories AS (
                        SELECT
                            memory_id::text,
                            cognitive_step,
                            content,
                            embedding,
                            created_at,
                            metadata
                        FROM (
                            SELECT
                                memory_id,
                                cognitive_step,
                                content,
                                embedding,
                                created_at,
                                metadata
                            FROM {cognitive_table}
                            WHERE {where_clause}
                            ORDER BY created_at DESC
                        ) AS subquery
                        LIMIT ${len(params)}
                    )
                    SELECT * FROM recent_memories
                    ORDER BY created_at ASC;
                """

            rows = await self.db.fetch(query, *params)
            
            items = []
            for row in rows:
                mem_id, step, content, embedding, created_at, meta = row
                
                # Parse embedding if it's a string
                if isinstance(embedding, str):
                    embedding = [float(x) for x in embedding.strip('[]').split(',')]

                # Parse metadata if it's a string
                if isinstance(meta, str):
                    try:
                        metadata = json.loads(meta)
                    except json.JSONDecodeError:
                        metadata = {}
                else:
                    metadata = meta if meta else {}

                # Create MemoryObject instance
                mo = MemoryObject(
                    memory_id=UUID(mem_id),
                    agent_id=agent_id,
                    cognitive_step=step,
                    content=content,
                    embedding=embedding,
                    created_at=created_at,
                    metadata=metadata
                )
                items.append(mo)
            return items

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

            if metadata_filters:
                for k, v in metadata_filters.items():
                    conditions.append(f"metadata->>{k} = ${len(params) + 1}")
                    params.append(str(v))

            if start_time:
                conditions.append(f"created_at >= ${len(params) + 1}")
                params.append(start_time)

            if end_time:
                conditions.append(f"created_at <= ${len(params) + 1}")
                params.append(end_time)

            where_clause = " AND ".join(conditions) if conditions else "TRUE"
            params.append(limit)

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
                LIMIT ${len(params)};
            """

            rows = await self.db.fetch(query, *params)
            episodes = []
            
            for row in rows:
                mem_id, task_query, steps_json, reward, strategy_json, embedding, created_at, meta = row

                # Parse JSON fields
                steps_list = json.loads(steps_json) if isinstance(steps_json, str) else steps_json or []
                strategy_list = json.loads(strategy_json) if isinstance(strategy_json, str) else strategy_json or []

                # Parse embedding if it's a string
                if isinstance(embedding, str):
                    embedding = [float(x) for x in embedding.strip('[]').split(',')]

                # Convert cognitive steps to CognitiveStep objects
                csteps = [CognitiveStep(**step) for step in steps_list]

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
            self.logger.debug("Starting knowledge ingestion...")
            # Get chunks
            chunks = self._chunk_text(text)
            chunk_texts = [c.text for c in chunks]
            
            # Get embeddings in batches and assign to chunks
            all_embeddings = []
            all_embeddings = await self.embedder.get_embeddings(chunk_texts)

            for chunk, embedding in zip(chunks, all_embeddings):
                chunk.embedding = embedding

            knowledge_id = uuid.uuid4()

            # Use a transaction for database operations
            async with self.db.safe_transaction() as conn:
                # Insert knowledge object
                await conn.execute(
                    f"""
                    INSERT INTO {table_prefix}_knowledge_objects 
                    (knowledge_id, content, metadata)
                    VALUES ($1, $2, $3)
                    """,
                    str(knowledge_id),
                    text,
                    json.dumps(metadata or {})
                )

                # Insert chunks with their embeddings
                for chunk in chunks:
                    # Convert embedding list to PostgreSQL vector format
                    vector_str = f"[{','.join(map(str, chunk.embedding))}]"
                    await conn.execute(
                        f"""
                        INSERT INTO {table_prefix}_knowledge_chunks
                        (knowledge_id, text, start_pos, end_pos, embedding)
                        VALUES ($1, $2, $3, $4, $5::vector)
                        """,
                        str(knowledge_id),
                        chunk.text,
                        chunk.start,
                        chunk.end,
                        vector_str
                    )

            return knowledge_id
        except Exception as e:
            self.logger.error(f"Error in ingest_knowledge: {str(e)}", exc_info=True)
            raise
    
    def _chunk_text(self, text: str) -> List[KnowledgeChunk]:
        """Internal method to chunk text using configured chunking method."""
        chunker = SemanticChunker(
            min_size=self.config.min_chunk_size,
            max_size=self.config.max_chunk_size
        )
        result = chunker.chunk(text)
        return result

    async def search_knowledge(
            self,
            query: str,
            table_prefix: str = "default",
            top_k: int = 5
    ) -> List[RetrievedMemory]:
        """Search knowledge base using semantic similarity."""
        try:
            # Get query embedding
            query_embedding = await self.embedder.get_embeddings(query)
            vector_str = f"[{','.join(map(str, query_embedding))}]"

            chunks_table = f"{table_prefix}_knowledge_chunks"
            objects_table = f"{table_prefix}_knowledge_objects"

            rows = await self.db.fetch(f"""
                WITH ranked_chunks AS (
                    SELECT DISTINCT ON (c.text)
                        c.text, 
                        c.start_pos, 
                        c.end_pos, 
                        k.content,
                        (1 - (c.embedding <=> $1::vector)) AS similarity
                    FROM {chunks_table} c
                    JOIN {objects_table} k ON c.knowledge_id = k.knowledge_id
                    WHERE (1 - (c.embedding <=> $1::vector)) >= $2
                    ORDER BY c.text, similarity DESC
                )
                SELECT * FROM ranked_chunks
                ORDER BY similarity DESC
                LIMIT $3;
            """, vector_str, self.config.similarity_threshold, top_k)

            results = []
            for row in rows:
                # Create RetrievedMemory object with proper fields
                retrieved = RetrievedMemory(
                    text=row['text'],
                    similarity=float(row['similarity']),
                    context=self._get_context(row['start_pos'], row['end_pos'], row['content'])
                )
                results.append(retrieved)

            return results

        except Exception as e:
            self.logger.error(f"Error searching knowledge base: {str(e)}")
            raise

    async def search_cognitive_memory(
            self,
            query: str,
            agent_id: str,
            top_k: int = 5
    ) -> List[RetrievedMemory]:
        query_embedding = self.embedding_service.get_embeddings(query)
        top_k = top_k or self.config.top_k

        safe_id = self.db._sanitize_table_name(agent_id)
        agent_cognitive_table = f"agent_{safe_id}_cognitive"

        rows = await self.db.fetch(f"""
            SELECT content,
                   (1 - (embedding <=> %s::vector)) AS similarity
            FROM {agent_cognitive_table}
            ORDER BY similarity DESC
            LIMIT %s;
        """, query_embedding, top_k)

        results = []
        for row in rows:
            content, sim = row
            results.append(RetrievedMemory(text=content, similarity=sim))
        return results

    async def search_episodic_memory(
        self,
        agent_id: str,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search episodic memory (vector/embedding mode)."""
        try:
            # Get query embedding first
            query_embedding = await self.embedder.get_embeddings(query)
            
            # Format the embedding vector for PostgreSQL
            vector_str = f"[{','.join(map(str, query_embedding))}]"
            
            safe_id = self.db._sanitize_table_name(agent_id)
            episodic_table = f"agent_{safe_id}_episodic"

            # Use the vector_str in the query
            results = await self.db.fetch(f"""
                SELECT 
                    memory_id,
                    task_query,
                    cognitive_steps,
                    total_reward,
                    strategy_update,
                    metadata,
                    created_at,
                    (1 - (embedding <=> $1::vector)) as similarity
                FROM {episodic_table}
                WHERE (1 - (embedding <=> $1::vector)) > $2
                ORDER BY similarity DESC
                LIMIT $3;
            """, vector_str, self.config.similarity_threshold, top_k)

            return [
                {
                    "text": json.dumps({
                        "memory_id": str(row["memory_id"]),
                        "task_query": row["task_query"],
                        "cognitive_steps": row["cognitive_steps"],
                        "total_reward": row["total_reward"],
                        "strategy_update": row["strategy_update"],
                        "metadata": row["metadata"],
                        "created_at": row["created_at"].isoformat()
                    }),
                    "similarity": float(row["similarity"])
                }
                for row in results
            ]

        except Exception as e:
            self.logger.error(f"Error searching episodic memory: {str(e)}")
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

            async with self.db.safe_transaction() as conn:
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
            embedding = row["embedding"]
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
            cognitive_steps = json.loads(row["cognitive_steps"])
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
                cognitive_steps=cognitive_steps,
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

    def _get_context(self, start: int, end: int, full_text: str) -> str:
        """
        Extracts context around a specific text chunk within a full document.
        """
        start_idx = max(0, start - self.config.max_input)
        end_idx = min(len(full_text), end + self.config.max_input)
        context = full_text[start_idx:end_idx].strip()
        if start_idx > 0:
            context = "..." + context
        if end_idx < len(full_text):
            context = context + "..."
        return context

    def uuid_encoder(self, obj):
        """JSON encoder function for handling UUIDs and datetimes."""
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

    async def store_ai_requests(self, requests: List[Dict[str, Any]]):
        """Store AI requests in the database."""
        try:
            values = []
            self.logger.info(f"Processing {len(requests)} requests")
            
            for req in requests:
                try:
                    self.logger.debug(f"Processing request: {req}")
                    
                    chat_thread = req.get('chat_thread')
                    if not chat_thread:
                        self.logger.warning("Skipping request - no chat thread found")
                        continue
                        
                    output = req.get('output')
                    if not output:
                        self.logger.warning("Skipping request - no output found")
                        continue
                        
                    timestamp = req.get('timestamp', time.time())

                    # Generate new unique request ID
                    request_id = str(uuid.uuid4())
                    self.logger.info(f"Generated new request ID: {request_id} for chat thread: {chat_thread.id}")

                    messages = []
                    #if chat_thread.system_prompt:
                    #    messages.append({
                    #        'role': 'system',
                    #        'content': chat_thread.system_prompt.content
                    #    })
                    #for msg in chat_thread.messages:
                    #    if isinstance(msg, dict):
                    #        messages.append(msg)
                    #    else:
                    #        messages.append({
                    #            'role': msg.role,
                    #            'content': msg.content
                    #        })
                    #if chat_thread.new_message:
                    #    messages.append({
                    #        'role': 'user',
                    #        'content': chat_thread.new_message
                    #    })
#
                    response = output.raw_output.raw_result if output and output.raw_output else None
                    
                    request_data = (
                        request_id,
                        str(chat_thread.id),
                        json.dumps(chat_thread.messages, default=self.uuid_encoder),
                        json.dumps(response, default=self.uuid_encoder),
                        json.dumps({
                            'chat_thread_id': str(chat_thread.id),
                            'model': chat_thread.llm_config.model if chat_thread.llm_config else None,
                            'client': chat_thread.llm_config.client if chat_thread.llm_config else None,
                            'start_time': output.raw_output.start_time if output and output.raw_output else None,
                            'end_time': output.raw_output.end_time if output and output.raw_output else None,
                            'completion_kwargs': chat_thread.llm_config.model_dump() if chat_thread.llm_config else None,
                            'usage': response.get('usage') if response else None
                        }, default=self.uuid_encoder),
                        datetime.fromtimestamp(timestamp, tz=timezone.utc)
                    )
                    values.append(request_data)
                    self.logger.debug(f"Successfully processed request ID: {request_id} for chat thread: {chat_thread.id}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing request: {e}")
                    self.logger.error(f"Problematic request: {req}")
                    continue

            if not values:
                self.logger.warning("No valid requests to insert")
                return

            self.logger.info(f"Attempting to store {len(values)} AI requests")
            async with self.db.safe_transaction() as conn:
                await conn.executemany("""
                    INSERT INTO ai_requests 
                    (request_id, agent_id, prompt, response, metadata, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                """, values)
                self.logger.info(f"Successfully stored {len(values)} AI requests")

        except Exception as e:
            self.logger.error(f"Failed to store AI requests: {e}")
            self.logger.error(f"Request data sample: {requests[0] if requests else None}")
            raise