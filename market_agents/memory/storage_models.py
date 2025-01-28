import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from uuid import UUID

from pydantic import BaseModel, Field


class RetrievedMemory(BaseModel):
    text: str
    similarity: float
    context: str = ""


class MemoryObject(BaseModel):
    """A single step of cognitive memory."""
    memory_id: UUID = Field(default_factory=uuid.uuid4)
    agent_id: str
    cognitive_step: str
    content: str
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def serialize_metadata(self) -> Optional[str]:
        """Serialize metadata to JSON string, handling datetime objects"""
        if not self.metadata:
            return None

        def serialize_value(v):
            if isinstance(v, datetime):
                return v.isoformat()
            if hasattr(v, 'serialize_json'):
                return json.loads(v.serialize_json())
            if hasattr(v, 'model_dump'):
                return v.model_dump()
            if isinstance(v, dict):
                return {k: serialize_value(v2) for k, v2 in v.items()}
            if isinstance(v, list):
                return [serialize_value(v2) for v2 in v]
            return v

        serialized_metadata = {k: serialize_value(v) for k, v in self.metadata.items()}
        return json.dumps(serialized_metadata)


class CognitiveStep(BaseModel):
    """
    A single step stored *inside* an EpisodicMemoryObject.
    """
    step_type: str = Field(..., description="Type of cognitive step (e.g., 'perception')")
    content: Dict[str, Any] = Field(..., description="Content of the step")


class EpisodicMemoryObject(BaseModel):
    """
    An entire 'episode' of task execution.
    It aggregates multiple steps (CognitiveStep),
    plus optional fields like total_reward, strategy_update, etc.
    """
    memory_id: UUID = Field(default_factory=uuid.uuid4)
    agent_id: str
    task_query: str
    cognitive_steps: List[CognitiveStep] = Field(default_factory=list)
    total_reward: Optional[float] = None
    strategy_update: Optional[List[str]] = None
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = None
    similarity: Optional[float] = None

class KnowledgeObject(BaseModel):
    knowledge_id: UUID = Field(default_factory=uuid.uuid4)
    content: str
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

class KnowledgeChunk(BaseModel):
    text: str
    start: int
    end: int
    embedding: Optional[List[float]] = None
    knowledge_id: Optional[UUID] = None

class CognitiveMemoryParams(BaseModel):
    """Parameters for querying cognitive memory."""
    limit: Optional[int] = Field(default=10, description="Maximum number of memories to return")
    cognitive_step: Optional[Union[str, List[str]]] = Field(default=None, description="Filter by cognitive step type(s)")
    metadata_filters: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata filters")
    start_time: Optional[datetime] = Field(default=None, description="Start time for temporal filtering")
    end_time: Optional[datetime] = Field(default=None, description="End time for temporal filtering")

class EpisodicMemoryParams(BaseModel):
    """Parameters for querying episodic memory."""
    limit: Optional[int] = Field(default=5, description="Maximum number of memories to return")
    metadata_filters: Optional[Dict] = Field(default=None, description="Additional metadata filters")
    start_time: Optional[datetime] = Field(default=None, description="Start time for temporal filtering")
    end_time: Optional[datetime] = Field(default=None, description="End time for temporal filtering")
    query: Optional[str] = Field(default=None, description="Search query string")
    top_k: Optional[int] = Field(default=5, description="Number of top results to return")

class KnowledgeQueryParams(BaseModel):
    """Parameters for querying knowledge base."""
    query: str = Field(..., description="Search query string")
    top_k: Optional[int] = Field(default=5, description="Number of top results to return")
    table_prefix: Optional[str] = Field(default=None, description="Optional table prefix for multi-tenant setups")

class IngestKnowledgeRequest(BaseModel):
    """Request model for knowledge ingestion."""
    text: str = Field(..., description="Text content to ingest")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata for the knowledge entry")
    table_prefix: Optional[str] = Field(default=None, description="Optional table prefix for multi-tenant setups")

class CreateTablesRequest(BaseModel):
    """Request model for table creation."""
    table_type: str = Field(..., description="Type of tables to create (cognitive/episodic/knowledge)")
    agent_id: Optional[str] = Field(default=None, description="Agent ID for memory tables")
    table_prefix: Optional[str] = Field(default=None, description="Optional table prefix for multi-tenant setups")

class AIRequest(BaseModel):
    request_id: str
    agent_id: Optional[str] = None
    prompt: str
    response: Any
    metadata: Dict[str, Any] = {}
    created_at: datetime