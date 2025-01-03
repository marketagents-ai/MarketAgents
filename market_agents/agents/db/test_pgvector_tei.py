import os
import psycopg2
import requests
import json
import numpy as np

# Configuration
SERVER_URL = 'http://38.128.232.35:8080/embed'
DB_CONFIG = {
    'dbname': os.environ.get('DB_NAME', 'market_simulation'),
    'user': os.environ.get('DB_USER', 'db_user'),
    'password': os.environ.get('DB_PASSWORD', 'db_pwd@123'),
    'host': os.environ.get('DB_HOST', 'localhost'),
    'port': os.environ.get('DB_PORT', '5433')
}

# Sample documents
documents = [
    "Artificial Intelligence is transforming the world in unprecedented ways. From healthcare to transportation, AI systems are revolutionizing how we live and work. The rapid advancement of AI technologies has sparked both excitement and ethical debates about their impact on society.",
    "Machine Learning enables computers to learn from data and improve their performance over time. By analyzing patterns in large datasets, ML algorithms can make predictions and decisions with increasing accuracy. This technology has found applications in recommendation systems, fraud detection, and autonomous vehicles.",
    "Deep Learning is a subset of Machine Learning that uses multi-layered neural networks. These deep neural networks can automatically learn hierarchical representations of data, making them particularly effective for complex tasks like image and speech recognition. The field has seen remarkable breakthroughs in recent years.",
    "Natural Language Processing allows machines to understand human language and interact with humans naturally. NLP systems can perform tasks like translation, sentiment analysis, and question answering. The technology has enabled virtual assistants and chatbots that can engage in meaningful conversations.",
    "Neural networks are the foundation of Deep Learning, mimicking the structure and function of biological brains. They consist of interconnected layers of artificial neurons that process and transform input data. The architecture of neural networks can be optimized for specific tasks through training.",
    "Transformers have revolutionized natural language processing with their attention mechanism. This architecture allows models to process input sequences in parallel and capture long-range dependencies effectively. The self-attention mechanism has become a cornerstone of modern NLP.",
    "BERT (Bidirectional Encoder Representations from Transformers) introduced powerful pre-training techniques for language understanding. By training on massive amounts of text data, BERT models develop rich contextual representations that can be fine-tuned for specific tasks.",
    "GPT (Generative Pre-trained Transformer) models demonstrate remarkable text generation capabilities. These autoregressive models can generate coherent and contextually relevant text across various domains and styles. Each new version has shown significant improvements in performance.",
    "The Vision Transformer (ViT) successfully adapted the Transformer architecture for computer vision tasks. By treating images as sequences of patches, ViT models achieve state-of-the-art performance on image classification and other vision tasks.",
    "T5 (Text-to-Text Transfer Transformer) unified various NLP tasks into a single text-to-text framework. This versatile architecture can handle multiple tasks like translation, summarization, and question answering using the same model architecture."
]

# Function to get embedding for a document
def get_embedding(text):
    response = requests.post(
        SERVER_URL,
        headers={'Content-Type': 'application/json'},
        data=json.dumps({'inputs': text})
    )
    print(f"Response Status: {response.status_code}")
    print(f"Response Content: {response.text}")
    
    if response.status_code == 200:
        try:
            embedding = response.json()
            if isinstance(embedding, list):
                return embedding[0]  
            elif isinstance(embedding, dict) and 'embedding' in embedding:
                return embedding['embedding']
            else:
                print("Unexpected response format:", embedding)
                return None
        except ValueError as e:
            print(f"Failed to parse JSON response: {e}")
            return None
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# Connect to the database
conn = psycopg2.connect(**DB_CONFIG)
cursor = conn.cursor()

# Ensure the vector extension is enabled
cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
conn.commit()

# Create the new table with vector(768)
cursor.execute("""
CREATE TABLE IF NOT EXISTS memory_embeddings_768 (
    id SERIAL PRIMARY KEY,
    agent_id UUID,
    embedding vector(768),
    memory_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
""")
conn.commit()

# Insert documents and their embeddings into the new table
for doc in documents:
    embedding = get_embedding(doc)
    if embedding and len(embedding) == 768:
        agent_id = '00000000-0000-0000-0000-000000000000'
        memory_data = {'text': doc}
        cursor.execute("""
            INSERT INTO memory_embeddings_768 (agent_id, embedding, memory_data)
            VALUES (%s, %s, %s::jsonb)
        """, (agent_id, embedding, json.dumps(memory_data)))
        conn.commit()
    else:
        print("Embedding size mismatch or retrieval error.")

# Query for similar documents
query_text = "What is Deep Learning"
query_embedding = get_embedding(query_text)

if query_embedding and len(query_embedding) == 768:
    cursor.execute("""
    SELECT id, memory_data, embedding <=> %s::vector AS cosine_distance
    FROM memory_embeddings_768
    ORDER BY cosine_distance
    LIMIT 3
    """, (query_embedding,))
    results = cursor.fetchall()
    print("\nTop 3 similar documents:")
    for result in results:
        print(f"ID: {result[0]}, Memory: {result[1]['text']}, Distance: {result[2]}")

# Clean up
cursor.close()
conn.close()
