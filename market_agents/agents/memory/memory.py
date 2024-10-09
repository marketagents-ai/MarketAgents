
"""
AI Memory Management System

This script implements a sophisticated memory management system for AI applications,
simulating human-like memory processes including storage, retrieval, decay, and forgetting.

Main components:
- Memory: Abstract base class for memory objects
- EpisodicMemory: Implementation of Memory, representing individual memories
- EmbeddingModel: Handles text-to-vector embedding
- VectorDB: Vector database for efficient storage and retrieval of memories
- ChunkingStrategy: Breaks text into manageable chunks
- MemoryManager: Orchestrates all memory operations

Key functions:
1. add_memory: Store new memories in the system
2. search: Retrieve relevant memories based on a query
3. forget_memories: Remove less important memories based on a threshold
4. decay_memories: Simulate natural memory decay over time
5. reinforce: Strengthen important or frequently accessed memories

The system uses vector embeddings for semantic similarity, TF-IDF for relevance scoring,
and implements a forgetting factor for memory decay. It's designed to maintain an
efficient and dynamic set of memories, prioritizing important and relevant information.

Usage:
    memory_manager = MemoryManager()
    memory_manager.add_memory("agent1", "New information to remember")
    results = memory_manager.search("agent1", "Query to find relevant memories")

"""

import sys
import os
from typing import Dict, Any, List
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
    def __init__(self, vector_dim: int = 768):
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

    def cosine_check_and_update(self, new_vector: List[float], new_meta: Dict[str, Any], threshold: float = 0.99) -> bool:
        new_vector = np.array(new_vector)
        for i, existing_vector in enumerate(self.vectors):
            similarity = 1 - cosine(new_vector, existing_vector)
            if similarity > threshold:
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

    def calculate_bm25_scores(self, query_terms: List[str], k1: float = 1.5, b: float = 0.75) -> Dict[int, float]:
        bm25_scores = {}
        for term in query_terms:
            if term in self.inverted_index:
                idf = math.log((self.total_documents - self.document_frequency[term] + 0.5) / 
                               (self.document_frequency[term] + 0.5) + 1.0)
                for index in self.inverted_index[term]:
                    tf = self.metadata[index]['content'].lower().count(term)
                    doc_length = len(self._tokenize(self.metadata[index]['content']))
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * doc_length / self.avg_document_length)
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
            hybrid_score = (0.3 * bm25_score + 0.7 * similarity) * ff
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
    @staticmethod
    def chunk(text: str, max_tokens: int = 256, encoding_name: str = 'gpt2') -> List[str]:
        encoding = tiktoken.get_encoding(encoding_name)
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        current_tokens = 0
        for sentence in sentences:
            sentence_tokens = len(encoding.encode(sentence))
            if current_tokens + sentence_tokens > max_tokens:
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
    def __init__(self, index_file: str = "memory_index.pkl", db_file: str = "vector_db.pkl"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.index_file = os.path.join(script_dir, index_file)
        self.db_file = os.path.join(script_dir, db_file)
        self.vector_db = VectorDB()
        self.embedding_model = EmbeddingModel()
        self.chunking_strategy = ChunkingStrategy()
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
                    print(f"Loaded memories for {len(self.memories)} agents.")
                except (AttributeError, ImportError) as e:
                    print(f"Error loading memories: {e}")
                    print("Starting with empty memory.")
        else:
            print("No existing memories found. Starting with empty memory.")

    def _load_vector_db(self):
        if os.path.exists(self.db_file):
            self.vector_db.load(self.db_file)
        else:
            print("No existing vector database found. Starting with empty database.")

    def _save_memories(self):
        with open(self.index_file, 'wb') as f:
            pickle.dump(self.memories, f)

    def _save_vector_db(self):
        self.vector_db.save(self.db_file)

    def add_memory(self, agent_id: str, content: str, metadata: Dict[str, Any] = None, memory_id: str = None):
        if agent_id not in self.memories:
            self.memories[agent_id] = []
        
        memory = EpisodicMemory(
            id=memory_id,
            agent_id=agent_id,
            content=content,
            vector_db=self.vector_db,
            embedding_model=self.embedding_model,
            retrieval_method=self.vector_db.search,
            chunking_strategy=self.chunking_strategy,
            metadata=metadata or {}
        )
        chunks = memory.chunk(content)
        memory.index(chunks)
        self.memories[agent_id].append(memory)
        self._save_memories()
        self._save_vector_db()

    def search(self, agent_id: str, query: str, top_k: int = 5) -> List[EpisodicMemory]:
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
        
        self._save_memories()
        self._save_vector_db()
        return top_memories

    def forget_memories(self, agent_id: str, threshold: float):
        if agent_id in self.memories:
            for memory in self.memories[agent_id]:
                if memory.forget(threshold):
                    self._remove_from_vector_db(memory)
            self.memories[agent_id] = [m for m in self.memories[agent_id] if m.forgetting_factor >= 0.1]
            self._save_memories()
            self._save_vector_db()

    def _remove_from_vector_db(self, memory: EpisodicMemory):
        self.vector_db.remove_item(memory.content)

    def decay_memories(self, agent_id: str, rate: float = 0.99):
        self.vector_db.update_forgetting_factors(rate)
        if agent_id in self.memories:
            for memory in self.memories[agent_id]:
                memory_vector = self.embedding_model.embed(memory.content)
                index = self.vector_db.find_closest_vector(memory_vector)
                memory.forgetting_factor = self.vector_db.forgetting_factors[index]
            self._save_memories()
            self._save_vector_db()

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