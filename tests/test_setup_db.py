import os
import pytest
import pytest_asyncio
from asyncpg import PostgresError

from market_agents.memory.agent_storage.setup_db import AsyncDatabase
from market_agents.memory.config import load_config_from_yaml

@pytest.fixture
def db_config():
    yaml_path = os.path.join(
        os.path.dirname(__file__),
        "../market_agents/memory/memory_config.yaml"
    )
    return load_config_from_yaml(yaml_path)

@pytest_asyncio.fixture
async def db_connection(db_config):
    db = AsyncDatabase(db_config)
    await db.initialize()
    yield db
    await db.close()


@pytest.mark.asyncio
async def test_database_connection(db_connection):
    """Test basic database connectivity and simple query execution"""
    try:
        # Test simple query execution
        result = await db_connection.fetch("SELECT 1")
        assert len(result) == 1, "Should return one row"
        assert result[0][0] == 1, "Should return value 1"
    except Exception as e:
        pytest.fail(f"Database connection test failed: {e}")

@pytest.mark.asyncio
async def test_extension_verification(db_connection):
    """Verify required PostgreSQL extensions are installed"""
    result = await db_connection.fetch(
        "SELECT * FROM pg_extension WHERE extname = 'vector'"
    )
    assert len(result) == 1, "Vector extension should be installed"


@pytest.mark.asyncio
async def test_transaction_rollback(db_connection):
    """Test transaction rollback on error"""
    await db_connection.execute(
        "CREATE TEMPORARY TABLE test_table (id SERIAL PRIMARY KEY, data TEXT)"
    )

    try:
        async with db_connection.transaction() as conn:
            await conn.execute("INSERT INTO test_table (data) VALUES ('test')")
            raise Exception("Simulated error for rollback")
    except Exception:
        pass

    # Verify rollback occurred
    result = await db_connection.fetch("SELECT * FROM test_table")
    assert len(result) == 0, "Transaction should have rolled back"


@pytest.mark.asyncio
async def test_transaction_commit(db, setup_test_tables):
    """Test successful transaction commit"""
    async with db.transaction() as conn:
        await conn.execute(
            "INSERT INTO users (name, email) VALUES ($1, $2)",
            "Test User", "test@example.com"
        )

    result = await db.fetch("SELECT * FROM users WHERE email = $1", "test@example.com")
    assert len(result) == 1
    assert result[0]['name'] == "Test User"

@pytest.mark.asyncio
async def test_execute_in_transaction(db_connection):
    """Test atomic execution of multiple queries in a transaction"""
    await db_connection.execute(
        "CREATE TEMPORARY TABLE test_table (id SERIAL PRIMARY KEY, data TEXT)"
    )

    queries = [
        ("INSERT INTO test_table (data) VALUES ($1)", ("first",)),
        ("INSERT INTO test_table (data) VALUES ($1)", ("second",)),
    ]

    await db_connection.execute_in_transaction(queries)

    result = await db_connection.fetch("SELECT * FROM test_table")
    assert len(result) == 2, "Both inserts should be committed"


@pytest.mark.asyncio
async def test_safe_transaction_retry(db, setup_test_tables):
    """Test transaction retry mechanism"""
    retry_count = 0

    async def insert_with_retry():
        nonlocal retry_count
        async with db.safe_transaction(max_retries=3) as conn:
            retry_count += 1
            if retry_count < 2:  # Fail first attempt
                raise PostgresError("Simulated deadlock")
            await conn.execute(
                "INSERT INTO users (name, email) VALUES ($1, $2)",
                "Retry User", "retry@example.com"
            )

    await insert_with_retry()
    assert retry_count == 2, "Should have retried once"

    result = await db.fetch("SELECT * FROM users WHERE email = $1", "retry@example.com")
    assert len(result) == 1, "Insert should have succeeded after retry"


if __name__ == "__main__":
    pytest.main([__file__])