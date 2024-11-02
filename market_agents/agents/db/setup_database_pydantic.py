from typing import Optional, Dict, Any, List, Union
from datetime import datetime
from decimal import Decimal
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict, Json, PostgresDsn, validator
import psycopg2
import psycopg2.extras
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Database Configuration Model
class DatabaseParams(BaseModel):
    dbname: str
    user: str
    password: str
    host: str
    port: str

    def get_connection_string(self) -> str:
        return f"dbname={self.dbname} user={self.user} password={self.password} host={self.host} port={self.port}"

class TimestampedModel(BaseModel):
    created_at: datetime = Field(default_factory=datetime.now)

class BaseIDModel(TimestampedModel):
    id: int = Field(default=None)

# Table Models
class Agent(TimestampedModel):
    id: UUID
    role: str = Field(..., pattern="^(buyer|seller)$")
    persona: Optional[str] = None
    system: Optional[str] = None
    task: Optional[str] = None
    tools: Optional[Dict[str, Any]] = None
    output_format: Optional[Dict[str, Any]] = None
    llm_config: Optional[Dict[str, Any]] = None
    max_retries: int = 2
    metadata: Optional[Dict[str, Any]] = None
    interactions: Optional[Dict[str, Any]] = None

class AgentMemory(BaseIDModel):
    agent_id: UUID
    step_id: int
    memory_data: Dict[str, Any]

class PreferenceSchedule(BaseIDModel):
    agent_id: UUID
    is_buyer: bool
    values: Dict[str, Any]
    costs: Dict[str, Any]
    initial_endowment: Dict[str, Any]

class Allocation(BaseIDModel):
    agent_id: UUID
    goods: int
    cash: Decimal
    locked_goods: int
    locked_cash: Decimal
    initial_goods: int
    initial_cash: Decimal

class Order(BaseIDModel):
    agent_id: UUID
    is_buy: bool
    quantity: int
    price: Decimal
    base_value: Optional[Decimal]
    base_cost: Optional[Decimal]

class Trade(BaseIDModel):
    buyer_id: UUID
    seller_id: UUID
    quantity: int
    price: Decimal
    buyer_surplus: Decimal
    seller_surplus: Decimal
    total_surplus: Decimal
    round: int

class Interaction(BaseIDModel):
    agent_id: UUID
    round: int
    task: str
    response: str

class Auction(BaseIDModel):
    max_rounds: int
    current_round: int
    total_surplus_extracted: Decimal
    average_prices: Dict[str, Any]
    order_book: Dict[str, Any]
    trade_history: Dict[str, Any]

class Environment(BaseIDModel):
    pass

class EnvironmentAgent(BaseIDModel):
    environment_id: int
    agent_id: UUID

class Perception(BaseIDModel):
    memory_id: UUID
    environment_name: str
    monologue: Optional[str]
    strategy: Optional[str]
    confidence: Optional[float]

class Action(BaseIDModel):
    memory_id: UUID
    environment_name: str
    action: Dict[str, Any]

class Reflection(BaseIDModel):
    memory_id: UUID
    environment_name: str
    reflection: str
    self_reward: float
    environment_reward: float
    total_reward: float
    strategy_update: str

class Observation(BaseIDModel):
    memory_id: UUID
    environment_name: str
    observation: Dict[str, Any]

class MemoryEmbedding(BaseIDModel):
    agent_id: UUID
    embedding: List[float]
    memory_data: Dict[str, Any]

class Request(BaseIDModel):
    prompt_context_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_time: Optional[float]
    model: str
    max_tokens: Optional[int]
    temperature: Optional[float]
    messages: Dict[str, Any]
    system: Optional[str]
    tools: Optional[Dict[str, Any]]
    tool_choice: Optional[Dict[str, Any]]
    raw_response: Dict[str, Any]
    completion_tokens: Optional[int]
    prompt_tokens: Optional[int]
    total_tokens: Optional[int]

class GroupChat(BaseIDModel):
    round_number: int
    topic: str
    agent_id: UUID
    message_type: str
    content: str
    is_human: bool = False

# Database Operations
def create_database(db_params: DatabaseParams) -> None:
    """Create the database if it doesn't exist."""
    conn = psycopg2.connect(
        dbname='postgres',
        user=db_params.user,
        password=db_params.password,
        host=db_params.host,
        port=db_params.port
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()

    cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (db_params.dbname,))
    exists = cursor.fetchone()
    if not exists:
        cursor.execute(f"CREATE DATABASE {db_params.dbname}")
        print(f"Database '{db_params.dbname}' created successfully.")
    else:
        print(f"Database '{db_params.dbname}' already exists.")

    cursor.close()
    conn.close()

def create_tables(db_params: DatabaseParams) -> None:
    """Create all necessary tables and indexes."""
    conn = psycopg2.connect(db_params.get_connection_string())
    cursor = conn.cursor()

    # Create pgvector extension
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create tables with correct UUID types
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS agents (
        id UUID PRIMARY KEY,
        role VARCHAR(10) NOT NULL CHECK (role IN ('buyer', 'seller')),
        persona TEXT,
        system TEXT,
        task TEXT,
        tools JSONB,
        output_format JSONB,
        llm_config JSONB,
        max_retries INTEGER DEFAULT 2,
        metadata JSONB,
        interactions JSONB,
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
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS groupchat (
        id SERIAL PRIMARY KEY,
        round_number INTEGER NOT NULL,
        topic TEXT NOT NULL,
        agent_id UUID NOT NULL,
        message_type VARCHAR(20) NOT NULL,
        content TEXT NOT NULL,
        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
        is_human BOOLEAN DEFAULT FALSE,
        FOREIGN KEY (agent_id) REFERENCES agents(id) ON DELETE CASCADE
    )
    """)
        cursor.execute("""
    CREATE TABLE IF NOT EXISTS personas (
        id SERIAL PRIMARY KEY,
        agent_id UUID REFERENCES agents(id),
        name VARCHAR(255) NOT NULL,
        role VARCHAR(10) NOT NULL CHECK (role IN ('buyer', 'seller')),
        persona TEXT NOT NULL,
        objectives JSONB NOT NULL,
        trader_type JSONB NOT NULL,
        demographic_characteristics JSONB NOT NULL,
        economic_attributes JSONB NOT NULL,
        personality_traits JSONB NOT NULL,
        hobbies_and_interests JSONB NOT NULL,
        dynamic_attributes JSONB NOT NULL,
        financial_objectives JSONB NOT NULL,
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

    cursor.execute("CREATE INDEX idx_groupchat_round_number ON groupchat(round_number);")
    cursor.execute("CREATE INDEX idx_groupchat_agent_id ON groupchat(agent_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_personas_agent_id ON personas(agent_id)")


    conn.commit()
    cursor.close()
    conn.close()

def drop_all_tables(db_params: DatabaseParams) -> None:
    """Drop all tables in the database."""
    conn = psycopg2.connect(db_params.get_connection_string())
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()

    drop_tables_sql = """
    DO $$ 
    DECLARE 
        r RECORD;
    BEGIN
        EXECUTE 'SET CONSTRAINTS ALL DEFERRED';
        FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public') LOOP
            EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(r.tablename) || ' CASCADE';
        END LOOP;
        EXECUTE 'SET CONSTRAINTS ALL IMMEDIATE';
    END $$;
    """

    try:
        cursor.execute(drop_tables_sql)
        print("All tables have been dropped successfully.")
    except Exception as e:
        print(f"An error occurred while dropping tables: {e}")
    finally:
        cursor.close()
        conn.close()

def reset_database(db_params: DatabaseParams) -> None:
    """Reset the entire database."""
    create_database(db_params)
    drop_all_tables(db_params)
    create_tables(db_params)
    print("Database reset complete.")

def insert_test_data(db_params: DatabaseParams) -> None:
    """Insert test data into the database."""
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
    db_params = DatabaseParams(
        dbname='market_simulation',
        user='db_user',
        password='db_pwd@123',
        host='localhost',
        port='5433'
    )
    
    # Create database and tables
    create_database(db_params)
    create_tables(db_params)
    
    # Uncomment to reset database or insert test data
    # reset_database(db_params)
    # insert_test_data(db_params)