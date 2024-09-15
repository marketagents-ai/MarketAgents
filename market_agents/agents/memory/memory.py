from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field, computed_field
from datetime import datetime

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
