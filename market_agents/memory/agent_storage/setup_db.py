import asyncio

import asyncpg
from asyncpg import Pool, create_pool
from asyncpg.exceptions import DuplicateDatabaseError
import logging
import re
from typing import Optional, List, Dict, Any


class AsyncDatabase:
    def __init__(self, config):
        self.config = config
        self.pool: Optional[Pool] = None
        self.logger = logging.getLogger("async_db")
        self._lock = asyncio.Lock()
        self._is_initialized = False

    async def initialize(self):
        """Initialize connection pool and verify database setup"""
        if not self._is_initialized:
            async with self._lock:
                if not self._is_initialized:
                    await self._ensure_database_exists()
                    self.pool = await create_pool(
                        min_size=self.config.pool_min,
                        max_size=self.config.pool_max,
                        host=self.config.host,
                        port=self.config.port,
                        user=self.config.user,
                        password=self.config.password,
                        database=self.config.dbname,
                        timeout=30,
                        command_timeout=60,
                    )
                    await self._verify_extensions()
                    self._is_initialized = True

    async def close(self):
        """Close all connections in the pool"""
        if self.pool:
            await self.pool.close()

    async def _ensure_database_exists(self):
        """Create database if it doesn't exist"""
        try:
            # Connect to maintenance database to check/create target DB
            temp_conn = await asyncpg.connect(
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                database='postgres'
            )

            try:
                await temp_conn.execute(
                    f"CREATE DATABASE {self.config.dbname}"
                )
                self.logger.info(f"Created database {self.config.dbname}")
            except DuplicateDatabaseError:
                pass
            finally:
                await temp_conn.close()
        except Exception as e:
            self.logger.error(f"Database creation failed: {e}")
            raise

    async def _verify_extensions(self):
        """Ensure required PostgreSQL extensions are installed"""
        async with self.pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    async def execute(self, query: str, *args) -> str:
        """Execute a SQL command and return status"""
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> List[Dict]:
        """Execute a query and return results"""
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def _sanitize_table_name(self, name: str) -> str:
        """Safely sanitize table names using regex"""
        return re.sub(r'[^a-zA-Z0-9_]', '_', name)

    async def create_knowledge_base_tables(self, base_name: str):
        """Create knowledge base tables with connection pooling"""
        safe_base = await self._sanitize_table_name(base_name)
        knowledge_objects_table = f"{safe_base}_knowledge_objects"
        knowledge_chunks_table = f"{safe_base}_knowledge_chunks"

        async with self.pool.acquire() as conn:
            async with conn.transaction():
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

                # Create vector index
                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS {safe_base}_chunks_index
                    ON {knowledge_chunks_table} USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = {self.config.lists});
                """)

    async def create_agent_cognitive_memory_table(self, agent_id: str):
        """Create cognitive memory table for an agent"""
        sanitized_id = await self._sanitize_table_name(agent_id)
        table_name = f"agent_{sanitized_id}_cognitive"

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        memory_id UUID PRIMARY KEY,
                        cognitive_step TEXT,
                        content TEXT,
                        embedding vector({self.config.vector_dim}),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB DEFAULT '{{}}'::jsonb
                    );
                """)

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS agent_{sanitized_id}_cognitive_index
                    ON {table_name} USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = {self.config.lists});
                """)

    async def init_agent_cognitive_memory(self, agent_ids: List[str]):
        """Initialize cognitive memory tables for multiple agents"""
        async with self.pool.acquire() as conn:
            for agent_id in agent_ids:
                await self.create_agent_cognitive_memory_table(agent_id)

    async def clear_agent_cognitive_memory(self, agent_id: str):
        """Clear cognitive memory for an agent"""
        sanitized_id = await self._sanitize_table_name(agent_id)
        table_name = f"agent_{sanitized_id}_cognitive"

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(f"TRUNCATE TABLE {table_name};")

    async def create_agent_episodic_memory_table(self, agent_id: str):
        """Create episodic memory table for an agent"""
        sanitized_id = await self._sanitize_table_name(agent_id)
        table_name = f"agent_{sanitized_id}_episodic"

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
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

                await conn.execute(f"""
                    CREATE INDEX IF NOT EXISTS agent_{sanitized_id}_episodic_index
                    ON {table_name} USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = {self.config.lists});
                """)

    async def clear_agent_episodic_memory(self, agent_id: str):
        """Clear episodic memory for an agent"""
        sanitized_id = await self._sanitize_table_name(agent_id)
        table_name = f"agent_{sanitized_id}_episodic"

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(f"TRUNCATE TABLE {table_name};")

    async def ensure_connection(self):
        """Verify database connectivity"""
        try:
            async with self.pool.acquire() as conn:
                await conn.fetch("SELECT 1")
        except Exception as e:
            self.logger.error("Connection verification failed: %s", e)
            await self.initialize()