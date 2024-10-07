import psycopg2
import psycopg2.extras  # Add this line
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
import numpy as np

# Database connection parameters
DB_NAME = "market_simulation"
DB_USER = "db_user"
DB_PASSWORD = "db_pwd@123"
DB_HOST = "localhost"
DB_PORT = "5433"

def create_database():
    # Connect to PostgreSQL server
    conn = psycopg2.connect(
        dbname='postgres',  # Connect to default 'postgres' database initially
        user=os.environ.get('DB_USER', 'db_user'),
        password=os.environ.get('DB_PASSWORD', 'db_pwd@123'),
        host=os.environ.get('DB_HOST', 'localhost'),  # Use 'localhost' as default
        port=os.environ.get('DB_PORT', '5433')
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()

    # Create database if it doesn't exist
    cursor.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{DB_NAME}'")
    exists = cursor.fetchone()
    if not exists:
        cursor.execute(f"CREATE DATABASE {DB_NAME}")
        print(f"Database '{DB_NAME}' created successfully.")
    else:
        print(f"Database '{DB_NAME}' already exists.")

    cursor.close()
    conn.close()

def create_tables():
    # Connect to the market_simulation database
    conn = psycopg2.connect(
        dbname=os.environ.get('DB_NAME', 'market_simulation'),
        user=os.environ.get('DB_USER', 'db_user'),
        password=os.environ.get('DB_PASSWORD', 'db_pwd@123'),
        host=os.environ.get('DB_HOST', 'localhost'),  # Use 'localhost' as default
        port=os.environ.get('DB_PORT', '5433')
    )
    cursor = conn.cursor()

    # Drop existing tables
    cursor.execute("DROP TABLE IF EXISTS agent_memories, preference_schedules, allocations, orders, trades, interactions, agents CASCADE")

    # Create tables with correct UUID types
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS agents (
        id UUID PRIMARY KEY,
        role VARCHAR(10) NOT NULL CHECK (role IN ('buyer', 'seller')),
        is_llm BOOLEAN NOT NULL,
        max_iter INTEGER NOT NULL,
        llm_config JSONB,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS agent_memories (
        id SERIAL PRIMARY KEY,
        agent_id UUID REFERENCES agents(id),
        step_id INTEGER NOT NULL,
        memory_data JSONB,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Update other tables that reference agents
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS preference_schedules (
        id SERIAL PRIMARY KEY,
        agent_id UUID REFERENCES agents(id),
        is_buyer BOOLEAN NOT NULL,
        values JSONB NOT NULL,
        initial_endowment DECIMAL(15, 2) NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS allocations (
        id SERIAL PRIMARY KEY,
        agent_id UUID REFERENCES agents(id),
        goods INTEGER NOT NULL,
        cash DECIMAL(15, 2) NOT NULL,
        locked_goods INTEGER NOT NULL,
        locked_cash DECIMAL(15, 2) NOT NULL,
        initial_goods INTEGER NOT NULL,
        initial_cash DECIMAL(15, 2) NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id SERIAL PRIMARY KEY,
        agent_id UUID REFERENCES agents(id),
        is_buy BOOLEAN NOT NULL,
        quantity INTEGER NOT NULL,
        price DECIMAL(15, 2) NOT NULL,
        base_value DECIMAL(15, 2),
        base_cost DECIMAL(15, 2),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id SERIAL PRIMARY KEY,
        buyer_id UUID REFERENCES agents(id),
        seller_id UUID REFERENCES agents(id),
        quantity INTEGER NOT NULL,
        price DECIMAL(15, 2) NOT NULL,
        buyer_surplus DECIMAL(15, 2) NOT NULL,
        seller_surplus DECIMAL(15, 2) NOT NULL,
        total_surplus DECIMAL(15, 2) NOT NULL,
        round INTEGER NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS interactions (
        id SERIAL PRIMARY KEY,
        agent_id UUID REFERENCES agents(id),
        round INTEGER NOT NULL,
        task TEXT NOT NULL,
        response TEXT NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS auctions (
        id SERIAL PRIMARY KEY,
        max_rounds INTEGER NOT NULL,
        current_round INTEGER NOT NULL,
        total_surplus_extracted DECIMAL(15, 2) NOT NULL,
        average_prices JSONB,
        order_book JSONB,
        trade_history JSONB,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS environments (
        id SERIAL PRIMARY KEY,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS environment_agents (
        id SERIAL PRIMARY KEY,
        environment_id INTEGER REFERENCES environments(id),
        agent_id UUID REFERENCES agents(id),
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS perceptions (
        id SERIAL PRIMARY KEY,
        memory_id UUID REFERENCES agents(id),
        environment_name TEXT NOT NULL,
        environment_info JSONB,
        recent_memories JSONB
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS actions (
        id SERIAL PRIMARY KEY,
        memory_id INTEGER REFERENCES agent_memories(id),
        environment_name TEXT NOT NULL,
        perception JSONB,
        environment_info JSONB,
        recent_memories JSONB,
        action_space JSONB
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reflections (
        id SERIAL PRIMARY KEY,
        memory_id UUID REFERENCES agents(id),  
        environment_name TEXT NOT NULL,
        observation JSONB,
        environment_info JSONB,
        last_action JSONB,
        reward FLOAT,
        previous_strategy TEXT
    )
    """)

    # Create pgvector extension
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create a new table for vector embeddings
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS memory_embeddings (
        id SERIAL PRIMARY KEY,
        agent_id UUID REFERENCES agents(id),
        embedding vector(1536),
        memory_data JSONB,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Create an index on the embedding column
    cursor.execute("CREATE INDEX ON memory_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")

    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_auction_id ON trades(id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_agent_id ON orders(agent_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_allocations_agent_id ON allocations(agent_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_agent_id ON interactions(agent_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_memories_agent_id ON agent_memories(agent_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_memories_step_id ON agent_memories(step_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_perceptions_memory_id ON perceptions(memory_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_actions_memory_id ON actions(memory_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_reflections_memory_id ON reflections(memory_id)")

    conn.commit()
    print("Tables, indexes, and pgvector extension created successfully.")

    cursor.close()
    conn.close()
def insert_test_data():
    conn = psycopg2.connect(
        dbname=os.environ.get('DB_NAME', 'market_simulation'),
        user=os.environ.get('DB_USER', 'db_user'),
        password=os.environ.get('DB_PASSWORD', 'db_pwd@123'),
        host=os.environ.get('DB_HOST', 'localhost'),
        port=os.environ.get('DB_PORT', '5433')
    )
    cursor = conn.cursor()

    # Insert a test agent
    cursor.execute("""
    INSERT INTO agents (id, role, is_llm, max_iter, llm_config)
    VALUES (gen_random_uuid(), 'buyer', true, 10, '{"model": "gpt-3.5-turbo"}')
    RETURNING id
    """)
    agent_id = cursor.fetchone()[0]

    # Insert test memory embeddings
    for _ in range(5):
        embedding = np.random.rand(1536).tolist()
        memory_data = {"text": f"Test memory {_}", "timestamp": "2023-04-01T12:00:00Z"}
        cursor.execute("""
        INSERT INTO memory_embeddings (agent_id, embedding, memory_data)
        VALUES (%s, %s, %s)
        """, (agent_id, embedding, psycopg2.extras.Json(memory_data)))

    conn.commit()
    print("Test data inserted successfully.")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    create_tables()
    insert_test_data()
