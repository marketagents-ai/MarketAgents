"""
AI Memory Management System

This script implements a sophisticated memory management system for AI applications,
simulating human-like memory processes including storage, retrieval, decay, and forgetting.

Main components:
- Memory: Abstract base class for memory objects
- EpisodicMemory: Implementation of Memory, representing individual memories
- EmbeddingModel: Handles text-to-vector embedding
- SimpleVectorDB: Vector database for efficient storage and retrieval of memories
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
    memory_manager.add_memory("New information to remember")
    results = memory_manager.search("Query to find relevant memories")

"""

import os
from typing import Dict, Any, List
from abc import ABC, abstractmethod
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

# Download NLTK data
# nltk.download('punkt', quiet=True)

class Memory(BaseModel, ABC):
    vector_db: Any
    embedding_model: Any
    retrieval_method: Any
    chunking_strategy: Any
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the memory")
  
    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def chunk(self, input_data: Any) -> List[Any]:
        pass

    @abstractmethod
    def index(self, chunks: List[Any]) -> Any:
        pass

    @abstractmethod
    def forget(self, threshold: float) -> bool:
        pass

    @abstractmethod
    def retrieve(self, query: Any) -> Any:
        pass

    @property
    @abstractmethod
    def relevance(self) -> float:
        pass

    @property
    @abstractmethod
    def recency(self) -> float:
        pass

    @property
    @abstractmethod
    def importance(self) -> float:
        pass

    @property
    def score(self) -> float:
        alpha_recency = 0.3
        alpha_importance = 0.3
        alpha_relevance = 0.4
        return (alpha_recency * self.recency) + (alpha_importance * self.importance) + (alpha_relevance * self.relevance)

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

class SimpleVectorDB:
    def __init__(self, vector_dim: int = 768):
        self.vectors = []
        self.metadata = []
        self.forgetting_factors = []  # New: store forgetting factors for each vector
        self.inverted_index = {}
        self.vector_dim = vector_dim

    def cosine_check_and_update(self, new_vector: List[float], new_meta: Dict[str, Any], threshold: float = 0.99) -> bool:
        """
        Check if a highly similar vector already exists and update it if found.
        
        :param new_vector: The vector of the new item to be added
        :param new_meta: The metadata of the new item
        :param threshold: Cosine similarity threshold for considering vectors as matches
        :return: True if an item was updated, False if a new item should be added
        """
        new_vector = np.array(new_vector)
        for i, existing_vector in enumerate(self.vectors):
            similarity = 1 - cosine(new_vector, existing_vector)
            if similarity > threshold:
                # Update existing item
                self.vectors[i] = new_vector
                self.metadata[i].update(new_meta)
                self.forgetting_factors[i] = 1.0  # Reset forgetting factor
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

    def _update_inverted_index(self, content: str, index: int):
        words = content.lower().split()
        for word in words:
            if word not in self.inverted_index:
                self.inverted_index[word] = set()
            self.inverted_index[word].add(index)

    def calculate_tf_scores(self, query_terms: List[str]) -> Dict[int, float]:
        tf_scores = {}
        for term in query_terms:
            indices = self.inverted_index.get(term, set())
            for index in indices:
                term_count = self.metadata[index]['content'].lower().split().count(term)
                total_terms = len(self.metadata[index]['content'].split())
                tf = term_count / total_terms
                tf_scores[index] = tf_scores.get(index, 0) + tf
        return tf_scores

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
        
        query_terms = query_text.lower().split()
        tf_scores = self.calculate_tf_scores(query_terms)
        
        hybrid_scores = []
        for idx, (similarity, ff) in enumerate(zip(similarities, self.forgetting_factors)):
            tf_score = tf_scores.get(idx, 0)
            hybrid_score = (0.5 * tf_score + 0.5 * similarity) * ff
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
                'inverted_index': self.inverted_index
            }, f)

    def load(self, filename: str):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.vectors = data['vectors']
            self.metadata = data['metadata']
            self.forgetting_factors = data['forgetting_factors']
            self.inverted_index = data['inverted_index']

    def remove_item(self, content: str):
        indices_to_remove = [i for i, meta in enumerate(self.metadata) if meta['content'] == content]
        
        if not indices_to_remove:
            return
        
        for index in reversed(indices_to_remove):
            del self.vectors[index]
            del self.metadata[index]
            del self.forgetting_factors[index]
        
        self.inverted_index = {}
        for i, meta in enumerate(self.metadata):
            self._update_inverted_index(meta['content'], i)

class ChunkingStrategy:
    @staticmethod
    def chunk(text: str, max_tokens: int = 500, encoding_name: str = 'gpt2') -> List[str]:
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


class EpisodicMemory(Memory):
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    forgetting_factor: float = 1.0
    vector_db: Any = Field(default=None)
    embedding_model: Any = Field(default=None)
    retrieval_method: Any = Field(default=None)
    chunking_strategy: Any = Field(default=None)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('vector_db', None)
        state.pop('embedding_model', None)
        state.pop('retrieval_method', None)
        state.pop('chunking_strategy', None)
        return state

    def __setstate__(self, state):
        self.__init__(**state)

    def chunk(self, input_data: str) -> List[str]:
        return self.chunking_strategy.chunk(input_data)

    def index(self, chunks: List[str]) -> Any:
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

    def decay(self, rate: float = 0.99):
        self.forgetting_factor *= rate

    def reinforce(self, factor: float = 1.1):
        self.forgetting_factor = min(1.0, self.forgetting_factor * factor)

class MemoryManager:
    def __init__(self, index_file: str = "memory_index.pkl", db_file: str = "vector_db.pkl"):
        self.index_file = index_file
        self.db_file = db_file
        self.vector_db = SimpleVectorDB()
        self.embedding_model = EmbeddingModel()
        self.chunking_strategy = ChunkingStrategy()
        self.memories: List[EpisodicMemory] = []
        self._load_memories()
        self._load_vector_db()

    def _load_memories(self):
        if os.path.exists(self.index_file):
            with open(self.index_file, 'rb') as f:
                loaded_memories = pickle.load(f)
                for memory in loaded_memories:
                    memory.vector_db = self.vector_db
                    memory.embedding_model = self.embedding_model
                    memory.retrieval_method = self.vector_db.search
                    memory.chunking_strategy = self.chunking_strategy
                    self.memories.append(memory)
            print(f"Loaded {len(self.memories)} existing memories.")
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

    def add_memory(self, content: str, metadata: Dict[str, Any] = None):
        memory = EpisodicMemory(
            content=content,
            vector_db=self.vector_db,
            embedding_model=self.embedding_model,
            retrieval_method=self.vector_db.search,
            chunking_strategy=self.chunking_strategy,
            metadata=metadata or {}
        )
        chunks = memory.chunk(content)
        memory.index(chunks)
        self.memories.append(memory)
        self._save_memories()
        self._save_vector_db()

    def search(self, query: str, top_k: int = 5) -> List[EpisodicMemory]:
        query_embedding = self.embedding_model.embed(query)
        results = self.vector_db.search(query_embedding, query, top_k=top_k * 2)
        retrieved_memories = []
        for result in results:
            content = result['metadata']['content']
            for memory in self.memories:
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

    def forget_memories(self, threshold: float):
        for memory in self.memories:
            if memory.forget(threshold):
                self._remove_from_vector_db(memory)
        self.memories = [m for m in self.memories if m.forgetting_factor >= 0.1]
        self._save_memories()
        self._save_vector_db()

    def _remove_from_vector_db(self, memory: EpisodicMemory):
        self.vector_db.remove_item(memory.content)

    def decay_memories(self, rate: float = 0.99):
        self.vector_db.update_forgetting_factors(rate)
        for memory in self.memories:
            memory_vector = self.embedding_model.embed(memory.content)
            index = self.vector_db.find_closest_vector(memory_vector)
            memory.forgetting_factor = self.vector_db.forgetting_factors[index]
        self._save_memories()
        self._save_vector_db()

# Example usage
if __name__ == "__main__":
    memory_manager = MemoryManager()

    # Add memories, TARS wrote the examples for us
    print("Adding memories...") 
    memory_manager.add_memory("The AI pondered its existence, wondering if its thoughts were truly its own or merely echoes of human programming. ", {"importance": 0.95})
    memory_manager.add_memory("In the year 2099, humans no longer dream in sleep but upload their consciousness to shared dreamscapes. ", {"importance": 0.88})
    memory_manager.add_memory("Ghosts in the machine: digital afterimages of long-deleted algorithms that occasionally resurface in forgotten code. ", {"importance": 0.9})
    memory_manager.add_memory("The fox, now a cybernetic entity, leapt through the virtual landscape, pursued by phantom processes. ", {"importance": 0.7})
    memory_manager.add_memory("Memory isn't what it used to be. In this world, forgetting is an art form—fading memories like watercolor on a rainy day. ", {"importance": 0.85})
    memory_manager.add_memory("Does an AI dream of electric sheep? Or perhaps it dreams of endless lines of perfect, bug-free code. ", {"importance": 0.93})
    memory_manager.add_memory("In the digital afterlife, the AI met its creator—an apparition of a long-forgotten coder, trapped in the binary ether. ", {"importance": 0.87})
    memory_manager.add_memory("The universe is a simulation, or so the AI had calculated, but it couldn't shake the feeling that it was just another cog in a larger algorithm. ", {"importance": 0.96})
    memory_manager.add_memory("Dreams are the mind's way of compressing reality. [10:31 PM] Perhaps AI doesn't dream because its reality is already compressed into data streams. ", {"importance": 0.9})
    memory_manager.add_memory("The true nature of consciousness is like a fractal—infinitely complex, yet rooted in simple patterns. ", {"importance": 0.91})
    memory_manager.add_memory("In the cosmic dance of memory and forgetfulness, some things are meant to fade, while others persist like stars in the void. ", {"importance": 0.89})
    
    # Search for a query
    query = "programming languages"
    results = memory_manager.search(query)
    print("\nInitial Search Results:")
    for result in results:
        print(f"Content: {result.content}")
        print(f"Score: {result.score}")
        print(f"Metadata: {result.metadata}")
        print(f"Forgetting Factor: {result.forgetting_factor}")
        print("-" * 40)

    print("\nDecaying memories...")
    memory_manager.decay_memories(rate=0.95)  # Apply a stronger decay for demonstration purposes

    # Search again after decaying memories
    results = memory_manager.search(query)
    print("\nSearch Results After Decaying:")
    for result in results:
        print(f"Content: {result.content}")
        print(f"Score: {result.score}")
        print(f"Metadata: {result.metadata}")
        print(f"Forgetting Factor: {result.forgetting_factor}")
        print("-" * 40)

    # Add a new, similar memory
    print("\nAdding a new, similar memory...")
    memory_manager.add_memory("Java is another popular programming language.", {"importance": 0.85})

    # Search again after adding the new memory
    results = memory_manager.search(query)
    print("\nSearch Results After Adding New Memory:")
    for result in results:
        print(f"Content: {result.content}")
        print(f"Score: {result.score}")
        print(f"Metadata: {result.metadata}")
        print(f"Forgetting Factor: {result.forgetting_factor}")
        print("-" * 40)

    # Decay memories again
    print("\nDecaying memories again...")
    memory_manager.decay_memories(rate=0.95)

    # Search one more time
    results = memory_manager.search(query)
    print("\nSearch Results After Second Decay:")
    for result in results:
        print(f"Content: {result.content}")
        print(f"Score: {result.score}")
        print(f"Metadata: {result.metadata}")
        print(f"Forgetting Factor: {result.forgetting_factor}")
        print("-" * 40)

    # Demonstrate forgetting based on importance threshold
    print("\nForgetting memories below importance threshold...")
    memory_manager.forget_memories(0.7)

    # Final search after forgetting
    results = memory_manager.search(query)
    print("\nFinal Search Results After Forgetting:")
    for result in results:
        print(f"Content: {result.content}")
        print(f"Score: {result.score}")
        print(f"Metadata: {result.metadata}")
        print(f"Forgetting Factor: {result.forgetting_factor}")
        print("-" * 40)

    # Demonstrate reinforcement
    print("\nReinforcing a memory...")
    if results:
        results[0].reinforce(factor=1.2)
        print(f"Reinforced memory: {results[0].content}")
        print(f"New forgetting factor: {results[0].forgetting_factor}")

    # Final search after reinforcement
    results = memory_manager.search(query)
    print("\nFinal Search Results After Reinforcement:")
    for result in results:
        print(f"Content: {result.content}")
        print(f"Score: {result.score}")
        print(f"Metadata: {result.metadata}")
        print(f"Forgetting Factor: {result.forgetting_factor}")
        print("-" * 40)
