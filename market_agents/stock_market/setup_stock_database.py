# setup_stock_database.py

import psycopg2
import psycopg2.extras
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def create_database(db_params):
    # Connect to PostgreSQL server
    conn = psycopg2.connect(
        dbname='postgres',  # Connect to default 'postgres' database initially
        user=db_params['user'],
        password=db_params['password'],
        host=db_params['host'],
        port=db_params['port']
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()

    # Create database if it doesn't exist
    cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_params['dbname'],))
    exists = cursor.fetchone()
    if not exists:
        cursor.execute(f"CREATE DATABASE {db_params['dbname']}")
        print(f"Database '{db_params['dbname']}' created successfully.")
    else:
        print(f"Database '{db_params['dbname']}' already exists.")

    cursor.close()
    conn.close()

def create_tables(db_params):
    # Connect to the specified database
    conn = psycopg2.connect(
        dbname=db_params['dbname'],
        user=db_params['user'],
        password=db_params['password'],
        host=db_params['host'],
        port=db_params['port']
    )
    cursor = conn.cursor()

    # Create pgvector extension
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create tables with correct UUID types
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS agents (
        id UUID PRIMARY KEY,
        role VARCHAR(10),
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

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS allocations (
        id SERIAL PRIMARY KEY,
        agent_id UUID REFERENCES agents(id),
        cash DECIMAL(15, 2) NOT NULL,
        initial_cash DECIMAL(15, 2) NOT NULL,
        positions JSONB NOT NULL,
        initial_positions JSONB NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE agent_positions (
        agent_id UUID NOT NULL,
        round INTEGER NOT NULL,
        cash FLOAT NOT NULL,
        positions JSONB NOT NULL,
        PRIMARY KEY (agent_id, round),
        FOREIGN KEY (agent_id) REFERENCES agents(id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        id SERIAL PRIMARY KEY,
        agent_id UUID REFERENCES agents(id),
        order_type VARCHAR(10) NOT NULL CHECK (order_type IN ('buy', 'sell', 'hold')),
        quantity INTEGER,
        price DECIMAL(15, 2),
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
    CREATE TABLE IF NOT EXISTS perceptions (
        id SERIAL PRIMARY KEY,
        memory_id UUID REFERENCES agents(id),
        environment_name TEXT NOT NULL,
        monologue TEXT,
        strategy TEXT,
        confidence FLOAT,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS actions (
        id SERIAL PRIMARY KEY,
        memory_id UUID REFERENCES agents(id),
        environment_name TEXT NOT NULL,
        action JSONB,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS reflections (
        id SERIAL PRIMARY KEY,
        memory_id UUID REFERENCES agents(id),
        environment_name TEXT NOT NULL,
        reflection TEXT,
        self_reward FLOAT,
        environment_reward FLOAT,
        total_reward FLOAT,
        strategy_update TEXT,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS observations (
        id SERIAL PRIMARY KEY,
        memory_id UUID REFERENCES agents(id),
        environment_name TEXT NOT NULL,
        observation JSONB,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS groupchat (
        message_id UUID PRIMARY KEY,
        agent_id UUID REFERENCES agents(id),
        round INTEGER NOT NULL,
        content TEXT NOT NULL,
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        topic TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS memory_embeddings (
        id SERIAL PRIMARY KEY,
        agent_id UUID REFERENCES agents(id),
        embedding vector(1536),
        memory_data JSONB,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS requests (
        id SERIAL PRIMARY KEY,
        prompt_context_id TEXT,
        start_time TIMESTAMP WITH TIME ZONE,
        end_time TIMESTAMP WITH TIME ZONE,
        total_time FLOAT,
        model TEXT,
        max_tokens INTEGER,
        temperature FLOAT,
        messages JSONB,
        system TEXT,
        tools JSONB,
        tool_choice JSONB,
        raw_response JSONB,
        completion_tokens INTEGER,
        prompt_tokens INTEGER,
        total_tokens INTEGER,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_buyer_id ON trades(buyer_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_seller_id ON trades(seller_id)")

    conn.commit()
    print("Tables and indexes created successfully.")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    db_params = {
        'dbname': 'market_simulation',
        'user': 'db_user',
        'password': 'db_pwd@123',
        'host': 'localhost',
        'port': '5433'
    }
    create_database(db_params)
    create_tables(db_params)
