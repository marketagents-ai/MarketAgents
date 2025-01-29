import asyncpg
from typing import Dict

async def setup_orchestrator_tables(conn: asyncpg.Connection):
    """Create all necessary tables for the orchestrator with proper schemas"""
    try:
        # Create agents table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS agents (
                id UUID PRIMARY KEY,
                role VARCHAR(50),
                persona JSONB,
                is_llm BOOLEAN,
                max_iter INTEGER,
                llm_config JSONB,
                economic_agent JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Create actions table
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS actions (
            id SERIAL PRIMARY KEY,
            agent_id UUID REFERENCES agents(id),
            environment_name TEXT NOT NULL,
            action_data JSONB,
            round INTEGER NOT NULL,
            sub_round INTEGER,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Create environment states table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS environment_states (
                id SERIAL PRIMARY KEY,
                environment_name VARCHAR(100),
                round INTEGER,
                state_data JSONB,
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        print("Successfully created all tables.")
        
    except Exception as e:
        print(f"Error creating tables: {e}")
        raise