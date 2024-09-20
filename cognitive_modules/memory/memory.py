# sketch out line for possible integration with main, for inspiration only, we need to chat about how I align this with the existing framework

import pickle
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, computed_field
from datetime import datetime
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

class AgentIO(BaseModel):
    input: str
    output: str
    timestamp: datetime

class AgentMemoryState(BaseModel):
    sliding_window: List[AgentIO] = Field(default_factory=list)
    max_window_size: int = 10
    thoughts: Optional[str] = None
    experience: Optional[str] = None
    trajectory: Optional[str] = None
    action: Optional[str] = None

    def add_io(self, input_str: str, output_str: str):
        new_io = AgentIO(input=input_str, output=output_str, timestamp=datetime.now())
        self.sliding_window.append(new_io)
        if len(self.sliding_window) > self.max_window_size:
            self.sliding_window.pop(0)

    def update_state(self, thoughts: str, experience: str, trajectory: str, action: str):
        self.thoughts = thoughts
        self.experience = experience
        self.trajectory = trajectory
        self.action = action

class PickleDB:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.ensure_db_exists()

    def ensure_db_exists(self):
        if not os.path.exists(self.file_path):
            self.create_empty_db()

    def create_empty_db(self):
        empty_data = {
            "index": {},
            "metadata": {}
        }
        with open(self.file_path, 'wb') as file:
            pickle.dump(empty_data, file)

    def load(self) -> Dict[str, Any]:
        with open(self.file_path, 'rb') as file:
            return pickle.load(file)

    def save(self, data: Dict[str, Any]):
        with open(self.file_path, 'wb') as file:
            pickle.dump(data, file)

    def update(self, update_func) -> Tuple[Dict[str, Any], Any]:
        data = self.load()
        updated_data, result = update_func(data)
        self.save(updated_data)
        return updated_data, result

class Memory(BaseModel, ABC):
    vector_db: Any
    embedding_model: Any
    retrieval_method: Any
    chunking_strategy: Any
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the memory")
  
    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def chunk(self, input_data: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
        pass

    @abstractmethod
    def index(self, chunks: List[str], embeddings: List[np.ndarray]) -> None:
        pass

    @abstractmethod
    def forget(self, threshold: float) -> bool:
        pass

    @abstractmethod
    def retrieve(self, query: Any) -> Any:
        pass

    @computed_field
    @property
    @abstractmethod
    def relevance(self) -> float:
        pass

    @computed_field
    @property
    @abstractmethod
    def recency(self) -> float:
        pass

    @computed_field
    @property
    @abstractmethod
    def importance(self) -> float:
        pass

    @computed_field
    @property
    def score(self) -> float:
        alpha_recency = 0.3
        alpha_importance = 0.3
        alpha_relevance = 0.4
        return (alpha_recency * self.recency) + (alpha_importance * self.importance) + (alpha_relevance * self.relevance)

class InvertedIndexMemory(Memory):
    index_file: str = Field(..., description="Path to the pickle file storing the inverted index")
    agent_memory: AgentMemoryState = Field(default_factory=AgentMemoryState)
    db: PickleDB = None

    def __init__(self, **data):
        super().__init__(**data)
        self.db = PickleDB(self.index_file)

    def process_agent_memory(self):
        all_text = " ".join([f"{io.input} {io.output}" for io in self.agent_memory.sliding_window])
        chunks = self.chunking_strategy.chunk(all_text)
        embeddings = self.embedding_model.embed_many(chunks)
        self.index(chunks, embeddings)

        for field in ['thoughts', 'experience', 'trajectory', 'action']:
            value = getattr(self.agent_memory, field)
            if value:
                self.index([value], [self.embedding_model.embed(value)])

    def update_agent_memory(self, new_state: AgentMemoryState):
        self.agent_memory = new_state
        self.process_agent_memory()

    def chunk(self, input_data: str, max_tokens: int = 512, overlap: int = 50) -> List[str]:
        return self.chunking_strategy.chunk(input_data, max_tokens, overlap)

    def index(self, chunks: List[str], embeddings: List[np.ndarray]) -> None:
        def update_index(data):
            index = data["index"]
            metadata = data["metadata"]
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                words = chunk.lower().split()
                chunk_id = f"chunk_{len(metadata)}"
                
                metadata[chunk_id] = {
                    "text": chunk,
                    "embedding": embedding.tolist(),
                    "timestamp": datetime.now().isoformat()
                }
                
                for position, word in enumerate(words):
                    if word not in index:
                        index[word] = []
                    index[word].append((chunk_id, position))
            
            return data, None

        self.db.update(update_index)
        self.vector_db.add_documents(chunks, embeddings)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.retrieval_method.retrieve(query, top_k)
        self._last_query_results = results
        
        # Combine TF-IDF and embedding-based relevance scores
        combined_results = []
        similarities = self.vector_db.get_similarity(self.embedding_model.embed(query))
        for (text, tfidf_score), emb_score in zip(results, similarities):
            combined_score = 0.5 * tfidf_score + 0.5 * emb_score  # Equal weighting, adjust as needed
            combined_results.append((text, combined_score))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x[1], reverse=True)
        return combined_results[:top_k]

    def forget(self, threshold: float) -> bool:
        def forget_old_chunks(data):
            index = data["index"]
            metadata = data["metadata"]
            
            current_time = datetime.now()
            chunks_to_remove = [
                chunk_id for chunk_id, chunk_data in metadata.items()
                if (current_time - datetime.fromisoformat(chunk_data["timestamp"])).days > threshold
            ]
            
            if chunks_to_remove:
                for chunk_id in chunks_to_remove:
                    del metadata[chunk_id]
                    for word in index:
                        index[word] = [(cid, pos) for cid, pos in index[word] if cid != chunk_id]
                return data, True
            return data, False

        _, forgotten = self.db.update(forget_old_chunks)
        if forgotten:
            # Rebuild the vector_db with the remaining documents
            remaining_chunks = [chunk_data["text"] for chunk_data in self.db.load()["metadata"].values()]
            remaining_embeddings = [np.array(chunk_data["embedding"]) for chunk_data in self.db.load()["metadata"].values()]
            self.vector_db.rebuild(remaining_chunks, remaining_embeddings)
        return forgotten

    @property
    def relevance(self) -> float:
        if not hasattr(self, '_last_query_results') or not self._last_query_results:
            return 0.0
        total_score = sum(score for _, score in self._last_query_results)
        avg_score = total_score / len(self._last_query_results)
        return max(0.0, min(1.0, avg_score))

    @property
    def recency(self) -> float:
        data = self.db.load()
        metadata = data["metadata"]
        if not metadata:
            return 0.0
        
        current_time = datetime.now()
        avg_age = sum((current_time - datetime.fromisoformat(chunk["timestamp"])).total_seconds() 
                      for chunk in metadata.values()) / len(metadata)
        
        max_age = 30 * 24 * 60 * 60  # 30 days in seconds
        normalized_recency = 1 - min(avg_age / max_age, 1)
        return normalized_recency

    @property
    def importance(self) -> float:
        data = self.db.load()
        metadata = data["metadata"]
        if not metadata:
            return 0.0
        
        avg_length = sum(len(chunk["text"].split()) for chunk in metadata.values()) / len(metadata)
        max_length = 1000  # Assuming 1000 words as maximum important length
        normalized_importance = min(avg_length / max_length, 1)
        return normalized_importance

class OpenAIEmbeddingModel:
    def __init__(self, api_key, base_url=None, model="text-embedding-ada-002"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model

    def embed(self, text):
        response = self.client.embeddings.create(
            model=self.model,
            input=[text]
        )
        return np.array(response.data[0].embedding)

    def embed_many(self, texts):
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [np.array(item.embedding) for item in response.data]

class EmbeddingVectorDB:
    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add_documents(self, documents, embeddings):
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)

    def rebuild(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings

    def get_similarity(self, query_embedding):
        # Return a list of similarities instead of a single value
        return cosine_similarity([query_embedding], self.embeddings)[0]

class EmbeddingRetrievalMethod:
    def __init__(self, vector_db, embedding_model):
        self.vector_db = vector_db
        self.embedding_model = embedding_model

    def retrieve(self, query, top_k=5):
        query_embedding = self.embedding_model.embed(query)
        similarities = self.vector_db.get_similarity(query_embedding)
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [(self.vector_db.documents[i], similarities[i]) for i in top_indices]

class SimpleChunkingStrategy:
    def chunk(self, text, max_tokens=512, overlap=50):
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_tokens - overlap):
            chunk = ' '.join(words[i:i + max_tokens])
            chunks.append(chunk)
        return chunks

# Main script
if __name__ == "__main__":
    vector_db = EmbeddingVectorDB()
    embedding_model = OpenAIEmbeddingModel(
        api_key='ollama',
        base_url='http://localhost:11434/v1/',
        model="all-minilm"
    )
    retrieval_method = EmbeddingRetrievalMethod(vector_db, embedding_model)
    chunking_strategy = SimpleChunkingStrategy()

    memory = InvertedIndexMemory(
        vector_db=vector_db,
        embedding_model=embedding_model,
        retrieval_method=retrieval_method,
        chunking_strategy=chunking_strategy,
        index_file="memory_index.pkl"
    )

    # Update agent memory
    agent_memory = AgentMemoryState()
    agent_memory.add_io("What's the weather like?", "It's sunny today.")
    agent_memory.update_state(
        thoughts="I should check the forecast for the week",
        experience="Users often ask about weather",
        trajectory="Provide more detailed weather information",
        action="Check weekly forecast"
    )
    memory.update_agent_memory(agent_memory)

    # Retrieve information
    results = memory.retrieve("weather forecast")
    print("Retrieved results:")
    for text, score in results:
        print(f"Text: {text}")
        print(f"Score: {score:.2f}")
        print("---")

    # Calculate and print memory scores
    print(f"Relevance score: {memory.relevance:.2f}")
    print(f"Recency score: {memory.recency:.2f}")
    print(f"Importance score: {memory.importance:.2f}")
    print(f"Overall memory score: {memory.score:.2f}")

    # Demonstrate forgetting
    forgotten = memory.forget(7)  # Forget chunks older than 7 days
    print(f"Chunks forgotten: {forgotten}")
