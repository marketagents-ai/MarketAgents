import os
import pytest
from market_agents.memory.setup_db import DatabaseConnection
from market_agents.memory.config import load_config_from_yaml

@pytest.fixture
def db_config():
    yaml_path = os.path.join(
        os.path.dirname(__file__),
        "../market_agents/memory/memory_config.yaml"
    )
    return load_config_from_yaml(yaml_path)

@pytest.fixture
def db_connection(db_config):
    db = DatabaseConnection(db_config)
    yield db
    # Cleanup after tests
    db.close()

def test_database_creation(db_connection):
    """Test basic database creation and connection"""
    try:
        db_connection._ensure_database_exists()
        db_connection.connect()
        # Simple test query
        db_connection.cursor.execute("SELECT 1")
        result = db_connection.cursor.fetchone()
        assert result[0] == 1
    except Exception as e:
        pytest.fail(f"Database creation failed: {e}")

def test_knowledge_base_tables(db_connection):
    """Test creation of knowledge base tables"""
    test_base_name = "test_market_knowledge"
    try:
        db_connection.create_knowledge_base_tables(test_base_name)
        # Verify tables exist
        db_connection.cursor.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            );
        """, (f"{test_base_name}_knowledge_objects",))
        assert db_connection.cursor.fetchone()[0] is True
    except Exception as e:
        pytest.fail(f"Knowledge base table creation failed: {e}")

def test_agent_memory_tables(db_connection):
    """Test creation of agent memory tables"""
    test_agent_id = "market_agent_123"
    try:
        # Test cognitive memory
        db_connection.create_agent_cognitive_memory_table(test_agent_id)
        # Test episodic memory
        db_connection.create_agent_episodic_memory_table(test_agent_id)
        
        # Verify tables exist
        for table_type in ['cognitive', 'episodic']:
            table_name = f"agent_{test_agent_id}_{table_type}"
            db_connection.cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                );
            """, (table_name,))
            assert db_connection.cursor.fetchone()[0] is True
    except Exception as e:
        pytest.fail(f"Agent memory table creation failed: {e}")

def test_memory_clearing(db_connection):
    """Test clearing agent memory tables"""
    test_agent_id = "test_market_agent_123"
    try:
        # Initialize tables
        db_connection.init_agent_cognitive_memory([test_agent_id])
        db_connection.init_agent_episodic_memory([test_agent_id])
        
        # Test clearing
        db_connection.clear_agent_cognitive_memory(test_agent_id)
        db_connection.clear_agent_episodic_memory(test_agent_id)
    except Exception as e:
        pytest.fail(f"Memory clearing operations failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])