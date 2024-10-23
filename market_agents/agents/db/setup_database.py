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

    # Create other tables
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS preference_schedules (
        id SERIAL PRIMARY KEY,
        agent_id UUID REFERENCES agents(id),
        is_buyer BOOLEAN NOT NULL,
        values JSONB,
        costs JSONB,
        initial_endowment JSONB,
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
        sub_round INTEGER NOT NULL,
        batch INTEGER NOT NULL,
        content TEXT NOT NULL,
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        topic TEXT
    )
    """)
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
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_requests_prompt_context_id ON requests(prompt_context_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_memory_embeddings_embedding ON memory_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_auction_id ON trades(id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_agent_id ON orders(agent_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_allocations_agent_id ON allocations(agent_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_agent_id ON interactions(agent_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_memories_agent_id ON agent_memories(agent_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_memories_step_id ON agent_memories(step_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_perceptions_memory_id ON perceptions(memory_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_actions_memory_id ON actions(memory_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_reflections_memory_id ON reflections(memory_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_observations_memory_id ON observations(memory_id)")

    conn.commit()
    print("Tables, indexes, and pgvector extension created successfully.")

    cursor.close()
    conn.close()

def insert_test_data(db_params):
    conn = psycopg2.connect(
        dbname=db_params['dbname'],
        user=db_params['user'],
        password=db_params['password'],
        host=db_params['host'],
        port=db_params['port']
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
        embedding = [0.0] * 1536  # Create a list of 1536 zeros
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
    # Example usage with default parameters
    db_params = {
        'dbname': 'market_simulation',
        'user': 'db_user',
        'password': 'db_pwd@123',
        'host': 'localhost',
        'port': '5433'
    }
    create_database(db_params)
    create_tables(db_params)
    # Optionally insert test data
    # insert_test_data(db_params)
