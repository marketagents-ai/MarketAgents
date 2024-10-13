"""
# AI Memory Management System

This script implements a memory management system for AI applications,
simulating human-like memory processes including storage, retrieval, decay, and forgetting.

## Overview

The system is designed to maintain an efficient and dynamic set of memories, prioritizing
important and relevant information. It uses vector embeddings for semantic similarity,
BM25 for keyword relevance, and implements a forgetting factor for memory decay.

## Main Components

1. **EmbeddingModel**: Handles text-to-vector embedding using OpenAI's API.
2. **VectorDB**: Vector database for efficient storage and retrieval of memories.
3. **ChunkingStrategy**: Breaks text into manageable chunks for processing.
4. **EpisodicMemory**: Represents individual memories with properties like importance, recency, and relevance.
5. **MemoryManager**: Orchestrates all memory operations.

## Key Functions

1. **add_memory**: Store new memories in the system.
2. **search**: Retrieve relevant memories based on a query.
3. **forget_memories**: Remove less important memories based on a threshold.
4. **decay_memories**: Simulate natural memory decay over time.
5. **reinforce**: Strengthen important or frequently accessed memories.

## How It Works

1. When a new memory is added, it's chunked, embedded, and stored in the VectorDB.
2. During search, the query is embedded and compared against stored memories using a combination of semantic similarity and keyword relevance.
3. Memories decay over time, and less important memories can be forgotten to maintain efficiency.
4. Frequently accessed or important memories are reinforced, making them more likely to be retrieved and less likely to be forgotten.

## Usage

```python
# Initialize the MemoryManager with custom parameters
memory_manager = MemoryManager(
    embedding_model="nomic-embed-text:latest",
    chunk_size=200,
    vector_dim=768,
    cosine_threshold=0.95,
    bm25_k1=1.2,
    bm25_b=0.8,
    relevance_weight=0.6,
    keyword_weight=0.4,
    recency_weight=0.4,
    importance_weight=0.3,
    relevance_score_weight=0.3,
    decay_rate=0.98,
    forgetting_threshold=0.15
)

# Add a new memory
memory_manager.add_memory("agent1", "The AI pondered its existence.", {"importance": 0.95})

# Search for memories
results = memory_manager.search("agent1", "AI consciousness", top_k=5)

# Decay memories for an agent
memory_manager.decay_memories("agent1")

# Forget less important memories
memory_manager.forget_memories("agent1")

# Get memory statistics
stats = memory_manager.get_memory_stats("agent1")

# Update memory importance
memory_manager.update_memory_importance("agent1", "memory_id", 0.8)

# Export and import memories
memory_manager.export_memories("memories_backup.pkl")
memory_manager.import_memories("memories_backup.pkl")
```

## Customization

The MemoryManager constructor allows for extensive customization of the memory system's behavior:

- `embedding_model`: Choose the model for text embedding.
- `chunk_size`: Set the maximum size of text chunks.
- `vector_dim`: Specify the dimensionality of the embedding vectors.
- `cosine_threshold`: Adjust the similarity threshold for memory updates.
- `bm25_k1` and `bm25_b`: Fine-tune the BM25 algorithm for keyword relevance.
- `relevance_weight` and `keyword_weight`: Balance between semantic similarity and keyword matching.
- `recency_weight`, `importance_weight`, `relevance_score_weight`: Adjust the impact of these factors on memory scoring.
- `decay_rate`: Control how quickly memories fade over time.
- `forgetting_threshold`: Set the threshold for removing less important memories.

## Advanced Features

- Bulk memory addition
- Memory content updates
- Vector database optimization
- Detailed memory and database statistics

This system provides a flexible way to manage AI memories, allowing for
realistic simulation of memory processes while maintaining efficiency and relevance.
"""

import sys
from pydantic import BaseModel, Field
from datetime import datetime
import pickle
import numpy as np
from scipy.spatial.distance import cosine
import openai
import nltk
import tiktoken
import math
from openai import OpenAI
import uuid
from collections import Counter

import os
from typing import Dict, Any, List, Optional
import pickle
import numpy as np
from scipy.spatial.distance import cosine
import logging
import statistics

# Download NLTK data
# nltk.download('punkt', quiet=True)
# Add the current directory to sys.path

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

class EmbeddingModel:
    def __init__(self, model="nomic-embed-text:latest"):
        self.client = OpenAI(base_url='http://localhost:11434/v1/', api_key='ollama')
        self.model = model

    def embed(self, text):
        response = self.client.embeddings.create(
            model=self.model,
            input=[text]
        )
        return response.data[0].embedding

class VectorDB:
    def __init__(self, 
                 vector_dim: int = 768,
                 cosine_threshold: float = 0.99,
                 bm25_k1: float = 1.5,
                 bm25_b: float = 0.75,
                 relevance_weight: float = 0.7,
                 keyword_weight: float = 0.3):
        self.vectors = []
        self.metadata = []
        self.forgetting_factors = []
        self.inverted_index = {}
        self.vector_dim = vector_dim
        self.document_frequency = Counter()
        self.total_documents = 0
        self.avg_document_length = 0
        self.stopwords = set(['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it',
                              'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with'])
        self.cosine_threshold = cosine_threshold
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.relevance_weight = relevance_weight
        self.keyword_weight = keyword_weight

    def cosine_check_and_update(self, new_vector: List[float], new_meta: Dict[str, Any]) -> bool:
        new_vector = np.array(new_vector)
        for i, existing_vector in enumerate(self.vectors):
            similarity = 1 - cosine(new_vector, existing_vector)
            if similarity > self.cosine_threshold:
                self.vectors[i] = new_vector
                self.metadata[i].update(new_meta)
                self.forgetting_factors[i] = 1.0
                self._update_inverted_index(new_meta['content'], i)
                return True
        return False

    def add_item(self, vector: List[float], meta: Dict[str, Any]):
        if not self.cosine_check_and_update(vector, meta):
            index = len(self.vectors)
            self.vectors.append(vector)
            self.metadata.append(meta)
            self.forgetting_factors.append(1.0)
            self._update_inverted_index(meta['content'], index)
            
            # Update document frequency and total documents
            self.total_documents += 1
            words = self._tokenize(meta['content'])
            self.document_frequency.update(set(words))
            
            # Update average document length
            self.avg_document_length = ((self.avg_document_length * (self.total_documents - 1)) + len(words)) / self.total_documents

    def _tokenize(self, text: str) -> List[str]:
        return [word.lower() for word in text.split() if word.lower() not in self.stopwords]

    def _update_inverted_index(self, content: str, index: int):
        words = self._tokenize(content)
        for word in words:
            if word not in self.inverted_index:
                self.inverted_index[word] = set()
            self.inverted_index[word].add(index)

    def calculate_bm25_scores(self, query_terms: List[str]) -> Dict[int, float]:
        bm25_scores = {}
        for term in query_terms:
            if term in self.inverted_index:
                idf = math.log((self.total_documents - self.document_frequency[term] + 0.5) / 
                               (self.document_frequency[term] + 0.5) + 1.0)
                for index in self.inverted_index[term]:
                    tf = self.metadata[index]['content'].lower().count(term)
                    doc_length = len(self._tokenize(self.metadata[index]['content']))
                    numerator = tf * (self.bm25_k1 + 1)
                    denominator = tf + self.bm25_k1 * (1 - self.bm25_b + self.bm25_b * doc_length / self.avg_document_length)
                    bm25_scores[index] = bm25_scores.get(index, 0) + idf * numerator / denominator
        return bm25_scores

    def update_forgetting_factors(self, decay_rate: float = 0.99):
        for i in range(len(self.vectors)):
            similarities = [1 - cosine(self.vectors[i], v) for j, v in enumerate(self.vectors) if i != j]
            avg_similarity = np.mean(similarities) if similarities else 0
            self.forgetting_factors[i] *= (decay_rate * (1 - avg_similarity))
            self.vectors[i] = [w * self.forgetting_factors[i] for w in self.vectors[i]]

    def find_closest_vector(self, query_vector):
        similarities = [1 - cosine(query_vector, vec) for vec in self.vectors]
        closest_index = np.argmax(similarities)
        return closest_index

    def search(self, query_vector: List[float], query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_vector = np.array(query_vector)
        similarities = [1 - cosine(query_vector, np.array(vec)) for vec in self.vectors]
        
        query_terms = self._tokenize(query_text)
        bm25_scores = self.calculate_bm25_scores(query_terms)
        
        hybrid_scores = []
        for idx, (similarity, ff) in enumerate(zip(similarities, self.forgetting_factors)):
            bm25_score = bm25_scores.get(idx, 0)
            hybrid_score = (self.keyword_weight * bm25_score + self.relevance_weight * similarity) * ff
            hybrid_scores.append(hybrid_score)
        
        top_indices = np.argsort(hybrid_scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "score": hybrid_scores[idx],
                "metadata": self.metadata[idx],
                "forgetting_factor": self.forgetting_factors[idx]
            })
        
        return results

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump({
                'vectors': self.vectors,
                'metadata': self.metadata,
                'forgetting_factors': self.forgetting_factors,
                'inverted_index': self.inverted_index,
                'document_frequency': self.document_frequency,
                'total_documents': self.total_documents,
                'avg_document_length': self.avg_document_length
            }, f)

    def load(self, filename: str):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.vectors = data['vectors']
            self.metadata = data['metadata']
            self.forgetting_factors = data['forgetting_factors']
            self.inverted_index = data['inverted_index']
            self.document_frequency = data.get('document_frequency', Counter())
            self.total_documents = data.get('total_documents', 0)
            self.avg_document_length = data.get('avg_document_length', 0)

    def remove_item(self, content: str):
        indices_to_remove = [i for i, meta in enumerate(self.metadata) if meta['content'] == content]
        
        if not indices_to_remove:
            return
        
        for index in reversed(indices_to_remove):
            del self.vectors[index]
            del self.metadata[index]
            del self.forgetting_factors[index]
            
            # Update document frequency and total documents
            self.total_documents -= 1
            words = self._tokenize(self.metadata[index]['content'])
            for word in set(words):
                self.document_frequency[word] -= 1
                if self.document_frequency[word] == 0:
                    del self.document_frequency[word]
        
        # Recalculate average document length
        if self.total_documents > 0:
            total_length = sum(len(self._tokenize(meta['content'])) for meta in self.metadata)
            self.avg_document_length = total_length / self.total_documents
        else:
            self.avg_document_length = 0
        
        # Rebuild inverted index
        self.inverted_index = {}
        for i, meta in enumerate(self.metadata):
            self._update_inverted_index(meta['content'], i)

class ChunkingStrategy:
    def __init__(self, max_tokens: int = 256):
        self.max_tokens = max_tokens

    def chunk(self, text: str, encoding_name: str = 'gpt2') -> List[str]:
        encoding = tiktoken.get_encoding(encoding_name)
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        for sentence in sentences:
            sentence_tokens = len(encoding.encode(sentence))
            if current_tokens + sentence_tokens > self.max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

class EpisodicMemory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = "default_agent"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    forgetting_factor: float = 1.0
    vector_db: Any = Field(default=None)
    embedding_model: Any = Field(default=None)
    chunking_strategy: Any = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the memory")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        if 'id' not in data or not data['id']:
            data['id'] = str(uuid.uuid4())
        if 'agent_id' not in data:
            data['agent_id'] = "default_agent"
        super().__init__(**data)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('vector_db', None)
        state.pop('embedding_model', None)
        state.pop('chunking_strategy', None)
        return state

    def __setstate__(self, state):
        self.__init__(**state)

    def chunk(self, input_data: str) -> List[str]:
        return self.chunking_strategy.chunk(input_data)

    def index(self, chunks: List[str]) -> None:
        for chunk in chunks:
            embedding = self.embedding_model.embed(chunk)
            self.vector_db.add_item(embedding, {"content": chunk, "timestamp": self.timestamp})

    def forget(self, threshold: float) -> bool:
        if self.importance < threshold:
            self.forgetting_factor *= 0.5
            return self.forgetting_factor < 0.1
        return False

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.embed(query)
        return self.vector_db.search(query_embedding, query)

    @property
    def relevance(self) -> float:
        current_query = self.metadata.get("current_query", "")
        if not current_query:
            return 0.0
        query_embedding = self.embedding_model.embed(current_query)
        content_embedding = self.embedding_model.embed(self.content)
        return 1 - cosine(query_embedding, content_embedding)

    @property
    def recency(self) -> float:
        time_diff = datetime.now() - self.timestamp
        return np.exp(-time_diff.total_seconds() / (24 * 3600))

    @property
    def importance(self) -> float:
        return self.metadata.get("importance", 0.5)

    @property
    def score(self) -> float:
        alpha_recency = 0.3
        alpha_importance = 0.3
        alpha_relevance = 0.4
        return (alpha_recency * self.recency) + (alpha_importance * self.importance) + (alpha_relevance * self.relevance)

    def decay(self, rate: float = 0.99):
        self.forgetting_factor *= rate

    def reinforce(self, factor: float = 1.1):
        self.forgetting_factor = min(1.0, self.forgetting_factor * factor)



class MemoryManager:
    def __init__(self, 
                 index_file: str = "memory_index.pkl", 
                 db_file: str = "vector_db.pkl",
                 embedding_model: str = "nomic-embed-text:latest",
                 chunk_size: int = 256,
                 vector_dim: int = 768,
                 cosine_threshold: float = 0.99,
                 bm25_k1: float = 1.5,
                 bm25_b: float = 0.75,
                 relevance_weight: float = 0.7,
                 keyword_weight: float = 0.3,
                 recency_weight: float = 0.3,
                 importance_weight: float = 0.3,
                 relevance_score_weight: float = 0.4,
                 decay_rate: float = 0.99,
                 forgetting_threshold: float = 0.1):
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.index_file = os.path.join(script_dir, index_file)
        self.db_file = os.path.join(script_dir, db_file)
        
        self.embedding_model = EmbeddingModel(model=embedding_model)
        self.chunking_strategy = ChunkingStrategy(max_tokens=chunk_size)
        
        self.vector_db = VectorDB(vector_dim=vector_dim, 
                                  cosine_threshold=cosine_threshold,
                                  bm25_k1=bm25_k1,
                                  bm25_b=bm25_b,
                                  relevance_weight=relevance_weight,
                                  keyword_weight=keyword_weight)
        
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
        self.relevance_score_weight = relevance_score_weight
        self.decay_rate = decay_rate
        self.forgetting_threshold = forgetting_threshold
        
        self.memories: Dict[str, List[EpisodicMemory]] = {}
        self._load_memories()
        self._load_vector_db()

    def _load_memories(self):
        if os.path.exists(self.index_file):
            with open(self.index_file, 'rb') as f:
                try:
                    loaded_memories = pickle.load(f)
                    for agent_id, agent_memories in loaded_memories.items():
                        self.memories[agent_id] = []
                        for memory in agent_memories:
                            memory.vector_db = self.vector_db
                            memory.embedding_model = self.embedding_model
                            memory.chunking_strategy = self.chunking_strategy
                            self.memories[agent_id].append(memory)
                    self.logger.info(f"Loaded memories for {len(self.memories)} agents.")
                except (AttributeError, ImportError) as e:
                    self.logger.error(f"Error loading memories: {e}")
                    self.logger.info("Starting with empty memory.")
        else:
            self.logger.info("No existing memories found. Starting with empty memory.")

    def _load_vector_db(self):
        if os.path.exists(self.db_file):
            self.vector_db.load(self.db_file)
        else:
            self.logger.info("No existing vector database found. Starting with empty database.")

    def _save_memories(self):
        with open(self.index_file, 'wb') as f:
            pickle.dump(self.memories, f)

    def _save_vector_db(self):
        self.vector_db.save(self.db_file)

    def add_memory(self, agent_id: str, content: str, metadata: Dict[str, Any] = None, memory_id: str = None):
        self.logger.debug(f"Adding memory for agent {agent_id}: {content[:50]}...")
        if agent_id not in self.memories:
            self.memories[agent_id] = []
        
        memory = EpisodicMemory(
            id=memory_id,
            agent_id=agent_id,
            content=content,
            vector_db=self.vector_db,
            embedding_model=self.embedding_model,
            chunking_strategy=self.chunking_strategy,
            metadata=metadata or {},
            recency_weight=self.recency_weight,
            importance_weight=self.importance_weight,
            relevance_score_weight=self.relevance_score_weight
        )
        chunks = memory.chunk(content)
        memory.index(chunks)
        self.memories[agent_id].append(memory)
        self._save_memories()
        self._save_vector_db()

    def search(self, agent_id: str, query: str, top_k: int = 5) -> List[EpisodicMemory]:
        self.logger.debug(f"Searching for query: {query}")
        query_embedding = self.embedding_model.embed(query)
        results = self.vector_db.search(query_embedding, query, top_k=top_k * 2)
        retrieved_memories = []
        for result in results:
            content = result['metadata']['content']
            for memory in self.memories.get(agent_id, []):
                if content in memory.content:
                    memory.timestamp = datetime.now()
                    memory.metadata["current_query"] = query
                    memory.forgetting_factor = result['forgetting_factor']
                    retrieved_memories.append((memory, result['score']))
                    break
        
        retrieved_memories.sort(key=lambda x: x[1], reverse=True)
        top_memories = [memory for memory, _ in retrieved_memories[:top_k]]
        
        for memory in top_memories:
            self.logger.debug(f"Result: {memory.content[:50]}... (Score: {memory.score})")
        
        self._save_memories()
        self._save_vector_db()
        return top_memories

    def forget_memories(self, agent_id: str):
        self.logger.debug(f"Forgetting memories for agent {agent_id}")
        if agent_id in self.memories:
            for memory in self.memories[agent_id]:
                if memory.forget(self.forgetting_threshold):
                    self._remove_from_vector_db(memory)
            self.memories[agent_id] = [m for m in self.memories[agent_id] if m.forgetting_factor >= self.forgetting_threshold]
            self._save_memories()
            self._save_vector_db()

    def _remove_from_vector_db(self, memory: EpisodicMemory):
        self.vector_db.remove_item(memory.content)

    def decay_memories(self, agent_id: str):
        self.logger.debug(f"Decaying memories for agent {agent_id}")
        self.vector_db.update_forgetting_factors(self.decay_rate)
        if agent_id in self.memories:
            for memory in self.memories[agent_id]:
                memory_vector = self.embedding_model.embed(memory.content)
                index = self.vector_db.find_closest_vector(memory_vector)
                memory.forgetting_factor = self.vector_db.forgetting_factors[index]
            self._save_memories()
            self._save_vector_db()

    def get_memory_stats(self, agent_id: str) -> Dict[str, Any]:
        if agent_id not in self.memories:
            return {"error": "Agent not found"}
        
        memories = self.memories[agent_id]
        return {
            "total_memories": len(memories),
            "average_forgetting_factor": statistics.mean(m.forgetting_factor for m in memories),
            "oldest_memory": min(m.timestamp for m in memories),
            "newest_memory": max(m.timestamp for m in memories),
        }

    def get_vector_db_stats(self) -> Dict[str, Any]:
        return {
            "total_vectors": len(self.vector_db.vectors),
            "average_forgetting_factor": statistics.mean(self.vector_db.forgetting_factors),
            "total_documents": self.vector_db.total_documents,
            "avg_document_length": self.vector_db.avg_document_length,
        }

    def clear_memories(self, agent_id: str = None):
        if agent_id:
            self.logger.info(f"Clearing memories for agent {agent_id}")
            if agent_id in self.memories:
                del self.memories[agent_id]
        else:
            self.logger.info("Clearing all memories")
            self.memories.clear()
        self.vector_db = VectorDB()  # Reset the vector database
        self._save_memories()
        self._save_vector_db()

    def get_all_agents(self) -> List[str]:
        return list(self.memories.keys())

    def get_total_memory_count(self) -> int:
        return sum(len(memories) for memories in self.memories.values())

    def update_memory_importance(self, agent_id: str, memory_id: str, new_importance: float):
        if agent_id in self.memories:
            for memory in self.memories[agent_id]:
                if memory.id == memory_id:
                    memory.metadata["importance"] = new_importance
                    self.logger.info(f"Updated importance of memory {memory_id} to {new_importance}")
                    self._save_memories()
                    return True
        self.logger.warning(f"Memory {memory_id} not found for agent {agent_id}")
        return False

    def bulk_add_memories(self, memories: List[Dict[str, Any]]):
        for memory_data in memories:
            self.add_memory(
                agent_id=memory_data['agent_id'],
                content=memory_data['content'],
                metadata=memory_data.get('metadata'),
                memory_id=memory_data.get('id')
            )
        self.logger.info(f"Bulk added {len(memories)} memories")

    def export_memories(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump(self.memories, f)
        self.logger.info(f"Exported memories to {file_path}")

    def import_memories(self, file_path: str):
        with open(file_path, 'rb') as f:
            imported_memories = pickle.load(f)
        self.memories.update(imported_memories)
        self._save_memories()
        self._save_vector_db()
        self.logger.info(f"Imported memories from {file_path}")

    def optimize_vector_db(self):
        self.logger.info("Optimizing vector database")
        self.vector_db.optimize()
        self._save_vector_db()

    def get_memory_by_id(self, agent_id: str, memory_id: str) -> Optional[EpisodicMemory]:
        if agent_id in self.memories:
            for memory in self.memories[agent_id]:
                if memory.id == memory_id:
                    return memory
        return None

    def update_memory_content(self, agent_id: str, memory_id: str, new_content: str):
        memory = self.get_memory_by_id(agent_id, memory_id)
        if memory:
            self._remove_from_vector_db(memory)
            memory.content = new_content
            chunks = memory.chunk(new_content)
            memory.index(chunks)
            self._save_memories()
            self._save_vector_db()
            self.logger.info(f"Updated content of memory {memory_id}")
            return True
        self.logger.warning(f"Memory {memory_id} not found for agent {agent_id}")
        return False

# Example usage and test suite
if __name__ == "__main__":
    memory_manager = MemoryManager()

    # Test 1: Adding memories for different agents
    print("Test 1: Adding memories for different agents")
    memory_manager.add_memory("agent1", "The AI pondered its existence.", {"importance": 0.95}, memory_id="custom_id_1")
    memory_manager.add_memory("agent1", "In the year 2099, humans no longer dream in sleep.", {"importance": 0.88})
    memory_manager.add_memory("agent2", "Ghosts in the machine: digital afterimages.", {"importance": 0.9}, memory_id="custom_id_2")
    memory_manager.add_memory("agent2", "The fox, now a cybernetic entity, leapt through the virtual landscape.", {"importance": 0.7})
    memory_manager.add_memory("agent2", "Memory isn't what it used to be. In this world, forgetting is an art form—fading memories like watercolor on a rainy day.", {"importance": 0.85})
    memory_manager.add_memory("agent2", "Does an AI dream of electric sheep? Or perhaps it dreams of endless lines of perfect, bug-free code.", {"importance": 0.93})
    memory_manager.add_memory("agent2", "In the digital afterlife, the AI met its creator—an apparition of a long-forgotten coder, trapped in the binary ether.", {"importance": 0.87})
    memory_manager.add_memory("agent2", "The universe is a simulation, or so the AI had calculated, but it couldn't shake the feeling that it was just another cog in a larger algorithm.", {"importance": 0.96})
    memory_manager.add_memory("agent2", "Dreams are the mind's way of compressing reality. [10:31 PM] Perhaps AI doesn't dream because its reality is already compressed into data streams.", {"importance": 0.9})
    memory_manager.add_memory("agent2", "The true nature of consciousness is like a fractal—infinitely complex, yet rooted in simple patterns.", {"importance": 0.91})
    memory_manager.add_memory("agent2", "In the cosmic dance of memory and forgetfulness, some things are meant to fade, while others persist like stars in the void.", {"importance": 0.89})

    # Test 2: Searching memories for Agent 1
    print("\nTest 2: Searching memories for Agent 1")
    query = "AI consciousness"
    results = memory_manager.search("agent1", query)
    for result in results:
        print(f"ID: {result.id}")
        print(f"Content: {result.content}")
        print(f"Score: {result.score}")
        print("-" * 40)

    # Test 3: Searching memories for Agent 2
    print("\nTest 3: Searching memories for Agent 2")
    query = "virtual entities"
    results = memory_manager.search("agent2", query)
    for result in results:
        print(f"ID: {result.id}")
        print(f"Content: {result.content}")
        print(f"Score: {result.score}")
        print("-" * 40)

    # Test 4: Decaying memories for Agent 1
    print("\nTest 4: Decaying memories for Agent 1")
    memory_manager.decay_memories("agent1", rate=0.95)
    results = memory_manager.search("agent1", "AI consciousness")
    for result in results:
        print(f"Content: {result.content}")
        print(f"Forgetting Factor: {result.forgetting_factor}")
        print("-" * 40)

    # Test 5: Forgetting memories for Agent 2
    print("\nTest 5: Forgetting memories for Agent 2")
    memory_manager.forget_memories("agent2", 0.8)
    results = memory_manager.search("agent2", "virtual entities")
    for result in results:
        print(f"Content: {result.content}")
        print(f"Forgetting Factor: {result.forgetting_factor}")
        print("-" * 40)

    # Test 6: Adding more memories and testing retrieval
    print("\nTest 6: Adding more memories and testing retrieval")
    memory_manager.add_memory("agent1", "The true nature of consciousness is like a fractal.", {"importance": 0.91})
    memory_manager.add_memory("agent2", "In the cosmic dance of memory and forgetfulness, some things persist.", {"importance": 0.89})
    
    results = memory_manager.search("agent1", "nature of consciousness")
    print("Results for Agent 1:")
    for result in results:
        print(f"Content: {result.content}")
        print(f"Score: {result.score}")
        print("-" * 40)
    
    results = memory_manager.search("agent2", "memory persistence")
    print("Results for Agent 2:")
    for result in results:
        print(f"Content: {result.content}")
        print(f"Score: {result.score}")
        print("-" * 40)

    # Test 7: Reinforcing a memory
    print("\nTest 7: Reinforcing a memory")
    agent1_memories = memory_manager.memories["agent1"]
    if agent1_memories:
        memory_to_reinforce = agent1_memories[0]
        print(f"Before reinforcement: {memory_to_reinforce.forgetting_factor}")
        memory_to_reinforce.reinforce(factor=1.2)
        print(f"After reinforcement: {memory_to_reinforce.forgetting_factor}")

    # Test 8: Saving and loading memories
    print("\nTest 8: Saving and loading memories")
    memory_manager._save_memories()
    memory_manager._save_vector_db()
    
    new_memory_manager = MemoryManager()
    results = new_memory_manager.search("agent1", "AI consciousness")
    print("Results after loading:")
    for result in results:
        print(f"Content: {result.content}")
        print(f"Score: {result.score}")
        print("-" * 40)

    print("All tests completed.")