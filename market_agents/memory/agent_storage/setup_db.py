import asyncio
import random

import asyncpg
from asyncpg import Pool, create_pool
from asyncpg.exceptions import DuplicateDatabaseError
import logging
import re
from typing import Optional, List, Dict, AsyncIterator
from contextlib import asynccontextmanager


class AsyncDatabase:
    def __init__(self, config):
        self.config = config
        self.pool: Optional[Pool] = None
        self.logger = logging.getLogger("async_db")
        self._lock = asyncio.Lock()
        self._is_initialized = False
        self.retry_config = {
            'initial_delay': self.config.retry_delay,
            'max_delay': self.config.retry_max_delay,
            'backoff_factor': self.config.retry_backoff_factor,
            'jitter': self.config.retry_jitter
        }

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
                        database=self.config.db_name,
                        timeout=30,
                        command_timeout=60,
                    )
                    await self._verify_extensions()
                    self._is_initialized = True

    async def close(self):
        """Close all connections in the pool"""
        if self.pool:
            await self.pool.close()

    @asynccontextmanager
    async def transaction(self, isolation: str = "repeatable_read") -> AsyncIterator[asyncpg.Connection]:
        """Transaction context manager with retry support"""
        async with self.pool.acquire() as conn:
            async with conn.transaction(isolation=isolation):
                yield conn

    @asynccontextmanager
    async def safe_transaction(self, max_retries: int = 3) -> AsyncIterator[asyncpg.Connection]:
        """Retryable transaction with error classification."""
        last_error = None
        for attempt in range(max_retries):
            try:
                async with self.transaction() as conn:
                    yield conn
                    return
            except Exception as e:
                last_error = e
                self.logger.error(f"Transaction failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(self._calculate_backoff(attempt))
                else:
                    raise last_error

    async def execute_in_transaction(self, queries: list[tuple[str, tuple]]):
        """Execute multiple queries in single transaction"""
        async with self.safe_transaction() as conn:
            for query, params in queries:
                await conn.execute(query, *params)

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
                    f"CREATE DATABASE {self.config.db_name}"
                )
                self.logger.info(f"Created database {self.config.db_name}")
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

    async def ensure_connection(self):
        """Verify database connectivity"""
        try:
            async with self.pool.acquire() as conn:
                await conn.fetch("SELECT 1")
        except Exception as e:
            self.logger.error("Connection verification failed: %s", e)
            await self.initialize()

    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff with jitter"""
        current_delay = self.retry_config['initial_delay'] * \
                        (self.retry_config['backoff_factor'] ** attempt)

        # Apply jitter
        jitter_amount = current_delay * self.retry_config['jitter'] * random.uniform(-1, 1)
        next_delay = int(current_delay + jitter_amount)

        # Enforce bounds
        return max(0, min(next_delay, self.retry_config['max_delay']))
    
    def _sanitize_table_name(self, name: str) -> str:
        """Safely sanitize table names using regex"""
        return re.sub(r'[^a-zA-Z0-9_]', '_', name)