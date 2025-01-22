import logging
import psycopg2
import psycopg2.extras
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def create_database(db_params):
    """Create the database if it doesn't exist"""
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

def setup_orchestrator_tables(db_params):
    """Create all necessary tables for the orchestrator with proper schemas"""
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    
    try:
        # Create agents table with persona column
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id UUID PRIMARY KEY,
                role VARCHAR(50),
                persona JSONB,
                is_llm BOOLEAN,
                max_iter INTEGER,
                llm_config JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Create cognitive memory table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cognitive_memory (
                memory_id UUID PRIMARY KEY,
                agent_id UUID REFERENCES agents(id),
                cognitive_step VARCHAR(50),
                content TEXT,
                embedding FLOAT[],
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Create perceptions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS perceptions (
                id SERIAL PRIMARY KEY,
                memory_id UUID,
                agent_id UUID REFERENCES agents(id),
                environment_name VARCHAR(50),
                round INTEGER,
                observation JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Create actions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS actions (
                id SERIAL PRIMARY KEY,
                memory_id UUID,
                agent_id UUID REFERENCES agents(id),
                environment_name VARCHAR(50),
                round INTEGER,
                action JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Create observations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                id SERIAL PRIMARY KEY,
                agent_id UUID REFERENCES agents(id),
                environment_name VARCHAR(50),
                round INTEGER,
                observation JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Create reflections table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS reflections (
                id SERIAL PRIMARY KEY,
                memory_id UUID,
                agent_id UUID REFERENCES agents(id),
                environment_name VARCHAR(50),
                round INTEGER,
                reflection TEXT,
                self_reward FLOAT,
                environment_reward FLOAT,
                total_reward FLOAT,
                strategy_update TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Create groupchat table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS groupchat (
                id SERIAL PRIMARY KEY,
                message_id UUID,
                agent_id UUID REFERENCES agents(id),
                round INTEGER,
                sub_round INTEGER,
                cohort_id VARCHAR(50),
                content TEXT,
                timestamp TIMESTAMP,
                topic TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Create requests table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS requests (
                id SERIAL PRIMARY KEY,
                prompt_context_id UUID,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                total_time FLOAT,
                model VARCHAR(255),
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
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        conn.commit()
        print("Successfully created all tables.")

        conn.commit()
        print("Successfully created all tables.")
        
    except Exception as e:
        conn.rollback()
        print(f"Error creating tables: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

def check_tables_exist(self):
    cursor = self.conn.cursor()
    tables = [
        'agents', 
        'agent_memories', 
        'groupchat', 
        'perceptions', 
        'actions', 
        'reflections',
        'requests'
    ]
    for table in tables:
        cursor.execute(f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{table}')")
        exists = cursor.fetchone()[0]
        if not exists:
            logging.error(f"Table {table} does not exist")
            cursor.close()
            return False
    cursor.close()
    return True

def drop_all_tables(db_params):
    """Drop all tables (useful for reset/testing)"""
    conn = psycopg2.connect(**db_params)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
        DROP TABLE IF EXISTS 
            reflections,
            actions,
            perceptions,
            environment_states,
            interactions,
            groupchat,
            agent_memories,
            agents,
            requests
        CASCADE
        """)
        conn.commit()
        print("Successfully dropped all tables.")
    except Exception as e:
        conn.rollback()
        print(f"Error dropping tables: {e}")
        raise
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    # Example usage
    db_params = {
        'dbname': 'market_agents',
        'user': 'postgres',
        'password': 'password',
        'host': 'localhost',
        'port': '5433'
    }
    
    # Create database if it doesn't exist
    create_database(db_params)
    
    # Optional: Drop all existing tables
    drop_all_tables(db_params)
    
    # Create new tables
    setup_orchestrator_tables(db_params)