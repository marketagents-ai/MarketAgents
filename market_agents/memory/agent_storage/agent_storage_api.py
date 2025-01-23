from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from uuid import UUID
import logging

from market_agents.memory.agent_storage.memory_service import MemoryService
from market_agents.memory.memory import MemoryObject, EpisodicMemoryObject

# Request Models
class CognitiveMemoryParams(BaseModel):
    limit: Optional[int] = 10
    cognitive_step: Optional[Union[str, List[str]]] = None
    metadata_filters: Optional[Dict[str, Any]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

class EpisodicMemoryParams(BaseModel):
    query: str  # Removed agent_id as it comes from path
    top_k: Optional[int] = 5

class KnowledgeQueryParams(BaseModel):
    query: str
    top_k: Optional[int] = 5
    table_prefix: Optional[str] = None

class IngestKnowledgeRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None
    table_prefix: Optional[str] = None

class CreateTablesRequest(BaseModel):
    table_type: str
    agent_id: Optional[str] = None
    table_prefix: Optional[str] = None

class MemoryAPI:
    def __init__(self, memory_service: MemoryService):
        self.router = APIRouter()
        self.memory_service = memory_service
        self.logger = logging.getLogger("memory_api")
        self._register_routes()

    def _register_routes(self):
        # Memory Management Endpoints
        @self.router.post("/memory/cognitive")
        async def store_cognitive_memory(memory: MemoryObject):  # Removed self
            """Store a cognitive memory item."""
            try:
                created_at = await self.memory_service.store_cognitive_memory(memory)
                return {"status": "success", "created_at": created_at}
            except Exception as e:
                self.logger.error(f"Error storing cognitive memory: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/memory/episodic")
        async def store_episodic_memory(episode: EpisodicMemoryObject):  # Removed self
            """Store an episodic memory."""
            try:
                await self.memory_service.store_episodic_memory(episode)
                return {"status": "success"}
            except Exception as e:
                self.logger.error(f"Error storing episodic memory: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/memory/cognitive/{agent_id}")
        async def get_cognitive_memory(
            agent_id: str,
            params: CognitiveMemoryParams = Depends()  # Correct query param handling
        ):
            """Retrieve cognitive memories for an agent using query parameters."""
            try:
                memories = await self.memory_service.get_cognitive_memory(
                    agent_id=agent_id,
                    limit=params.limit,
                    cognitive_step=params.cognitive_step,
                    metadata_filters=params.metadata_filters,
                    start_time=params.start_time,
                    end_time=params.end_time
                )
                return {"memories": [mem.model_dump() for mem in memories]}
            except Exception as e:
                self.logger.error(f"Error retrieving cognitive memory: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/memory/episodic/{agent_id}")
        async def get_episodic_memory(
            agent_id: str,
            params: EpisodicMemoryParams = Depends()  # Corrected params
        ):
            """Retrieve episodic memories for an agent using semantic search."""
            try:
                episodes = await self.memory_service.get_episodic_memory(
                    agent_id=agent_id,
                    query=params.query,
                    top_k=params.top_k
                )
                return {"episodes": [ep.model_dump() for ep in episodes]}
            except Exception as e:
                self.logger.error(f"Error retrieving episodic memory: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.delete("/memory/{agent_id}")
        async def clear_agent_memory(
            agent_id: str,
            memory_type: Optional[str] = None  # Query parameter
        ):
            """Clear agent's memory of specified type (cognitive/episodic)."""
            try:
                deleted_count = await self.memory_service.clear_agent_memory(
                    agent_id=agent_id,
                    memory_type=memory_type
                )
                return {
                    "status": "success",
                    "agent_id": agent_id,
                    "memory_type": memory_type or "all",
                    "deleted_count": deleted_count
                }
            except Exception as e:
                self.logger.error(f"Error clearing agent memory: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        # Knowledge Base Endpoints
        @self.router.post("/knowledge/ingest")
        async def ingest_knowledge(request: IngestKnowledgeRequest):  # Using request model
            """Ingest text into the knowledge base."""
            try:
                knowledge_id = await self.memory_service.ingest_knowledge(
                    text=request.text,
                    metadata=request.metadata,
                    table_prefix=request.table_prefix or "default"
                )
                return {
                    "status": "success",
                    "knowledge_id": knowledge_id
                }
            except Exception as e:
                self.logger.error(f"Error ingesting knowledge: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/knowledge/search")
        async def search_knowledge(params: KnowledgeQueryParams = Depends()):
            """Search the knowledge base using semantic similarity."""
            try:
                results = await self.memory_service.search_knowledge(
                    query=params.query,
                    top_k=params.top_k,
                    table_prefix=params.table_prefix
                )
                return {"matches": results}
            except Exception as e:
                self.logger.error(f"Error searching knowledge base: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/memory/episodic/search")
        async def search_episodic_memory(
            agent_id: str,
            top_k: int,
            query: str
        ):
            """Search episodic memory using semantic similarity."""
            try:
                results = await self.memory_service.search_episodic_memory(
                    agent_id=agent_id,
                    top_k=top_k,
                    query=query
                )
                return {"matches": results}
            except Exception as e:
                self.logger.error(f"Error searching episodic memory: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.router.get("/memory/cognitive/search")
        async def search_cognitive_memory(
            agent_id: str,
            top_k: int,
            query: str
        ):
            """Search cognitive memory using semantic similarity."""
            try:
                results = await self.memory_service.search_cognitive_memory(
                    agent_id=agent_id,
                    top_k=top_k,
                    query=query
                )
                return {"matches": results}
            except Exception as e:
                self.logger.error(f"Error searching cognitive memory: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.delete("/knowledge/{knowledge_id}")
        async def delete_knowledge(
            knowledge_id: UUID,
            table_prefix: Optional[str] = None
        ):
            """Delete a specific knowledge entry and its chunks."""
            try:
                deleted = await self.memory_service.delete_knowledge(
                    knowledge_id=knowledge_id,
                    table_prefix=table_prefix or "default"
                )
                return {
                    "status": "success",
                    "knowledge_id": knowledge_id,
                    "deleted": deleted
                }
            except Exception as e:
                self.logger.error(f"Error deleting knowledge: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        # Table Management Endpoint
        @self.router.post("/tables/create")
        async def create_tables(request: CreateTablesRequest):  # Using request model
            """Create memory or knowledge base tables."""
            try:
                await self.memory_service.create_tables(
                    table_type=request.table_type,
                    agent_id=request.agent_id,
                    table_prefix=request.table_prefix
                )
                return {
                    "status": "success",
                    "table_type": request.table_type,
                    "agent_id": request.agent_id,
                    "table_prefix": request.table_prefix
                }
            except Exception as e:
                self.logger.error(f"Error creating tables: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))