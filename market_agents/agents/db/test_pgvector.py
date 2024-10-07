import psycopg2
import psycopg2.extras
import os
import numpy as np

def test_pgvector():
    conn = psycopg2.connect(
        dbname=os.environ.get('DB_NAME', 'market_simulation'),
        user=os.environ.get('DB_USER', 'db_user'),
        password=os.environ.get('DB_PASSWORD', 'db_pwd@123'),
        host=os.environ.get('DB_HOST', 'localhost'),
        port=os.environ.get('DB_PORT', '5433')
    )
    cursor = conn.cursor()

    # Create vector extension if it doesn't exist
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()

    # Test vector similarity search
    query_vector = np.random.rand(1536).tolist()
    cursor.execute("""
    SELECT id, memory_data, embedding <-> %s::vector AS distance
    FROM memory_embeddings
    ORDER BY distance
    LIMIT 3
    """, (query_vector,))

    results = cursor.fetchall()
    print("Top 3 similar vectors:")
    for result in results:
        print(f"ID: {result[0]}, Memory: {result[1]}, Distance: {result[2]}")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    test_pgvector()