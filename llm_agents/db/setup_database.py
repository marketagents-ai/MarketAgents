import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Database connection parameters
DB_NAME = "market_simulation"
DB_USER = "your_username"
DB_PASSWORD = "your_password"
DB_HOST = "localhost"
DB_PORT = "5432"

def create_database():
    # Connect to PostgreSQL server
    conn = psycopg2.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
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
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cursor = conn.cursor()

    # Create tables
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
        buyer_value DECIMAL(15, 2) NOT NULL,
        seller_cost DECIMAL(15, 2) NOT NULL,
        round INTEGER NOT NULL,
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
    CREATE TABLE IF NOT EXISTS interactions (
        id SERIAL PRIMARY KEY,
        agent_id UUID REFERENCES agents(id),
        round INTEGER NOT NULL,
        task TEXT NOT NULL,
        response TEXT NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_auction_id ON trades(id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_agent_id ON orders(agent_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_allocations_agent_id ON allocations(agent_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_agent_id ON interactions(agent_id)")

    conn.commit()
    print("Tables and indexes created successfully.")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    create_database()
    create_tables()