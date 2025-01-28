import asyncio
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from contextlib import asynccontextmanager
import uvicorn
from typing import Optional
from uuid import UUID
import logging

from market_agents.memory.agent_storage.storage_service import StorageService
from market_agents.memory.memory_models import (
    MemoryObject,
    EpisodicMemoryObject,
    CognitiveMemoryParams,
    EpisodicMemoryParams,
    KnowledgeQueryParams,
    IngestKnowledgeRequest,
    CreateTablesRequest
)


class AgentStorageAPI:
    def __init__(self, storage_service: StorageService):
        self.router = APIRouter()
        self.storage_service = storage_service
        self.logger = logging.getLogger("agent_storage_api")
        self._register_routes()

    def _register_routes(self):
        @self.router.post("/memory/cognitive")
        async def store_cognitive_memory(memory: MemoryObject):
            """Store a single-step (cognitive) memory item."""
            try:
                created_at = await self.storage_service.store_cognitive_memory(memory)
                return {"status": "success", "created_at": created_at}
            except Exception as e:
                self.logger.error(f"Error storing cognitive memory: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.post("/memory/episodic")
        async def store_episodic_memory(episode: EpisodicMemoryObject):
            """Store an entire (episodic) memory."""
            try:
                await self.storage_service.store_episodic_memory(episode)
                return {"status": "success"}
            except Exception as e:
                self.logger.error(f"Error storing episodic memory: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/memory/cognitive/sql/{agent_id}")
        async def get_cognitive_memory_sql(
            agent_id: str,
            params: CognitiveMemoryParams = Depends()
        ):
            """
            Retrieve cognitive memories (SQL/relational mode) for an agent,
            using time range, metadata filters, etc.
            """
            try:
                memories = await self.storage_service.get_cognitive_memory(
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

        @self.router.get("/memory/episodic/sql/{agent_id}")
        async def get_episodic_memory_sql(
            agent_id: str,
            params: EpisodicMemoryParams = Depends()
        ):
            """
            Retrieve episodic memories (SQL/relational mode) for an agent,
            using time range, metadata filters, etc.
            """
            try:
                episodes = await self.storage_service.get_episodic_memory(
                    agent_id=agent_id,
                    limit=params.limit,
                    metadata_filters=params.metadata_filters,
                    start_time=params.start_time,
                    end_time=params.end_time
                )
                return {"episodes": [ep.model_dump() for ep in episodes]}
            except Exception as e:
                self.logger.error(f"Error retrieving episodic memory: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/memory/cognitive/vector/{agent_id}")
        async def search_cognitive_memory_vector(
            agent_id: str,
            query: str,
            top_k: int = 5
        ):
            """Search cognitive memory (vector/embedding mode)."""
            try:
                results = await self.storage_service.search_cognitive_memory(
                    agent_id=agent_id,
                    top_k=top_k,
                    query=query
                )
                return {"matches": results}
            except Exception as e:
                self.logger.error(f"Error searching cognitive memory: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.get("/memory/episodic/vector/{agent_id}")
        async def search_episodic_memory_vector(
            agent_id: str,
            query: str,
            top_k: int = 5
        ):
            """Search episodic memory (vector/embedding mode)."""
            try:
                results = await self.storage_service.search_episodic_memory(
                    agent_id=agent_id,
                    query=query,
                    top_k=top_k
                )
                return {"matches": results}
            except Exception as e:
                self.logger.error(f"Error searching episodic memory: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.delete("/memory/{agent_id}")
        async def clear_agent_memory(
            agent_id: str,
            memory_type: Optional[str] = None
        ):
            """Clear agent's memory of specified type (cognitive/episodic)."""
            try:
                deleted_count = await self.storage_service.clear_agent_memory(
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

        @self.router.post("/knowledge/ingest")
        async def ingest_knowledge(request: IngestKnowledgeRequest):
            """Ingest text into the knowledge base."""
            try:
                knowledge_id = await self.storage_service.ingest_knowledge(
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
                results = await self.storage_service.search_knowledge(
                    query=params.query,
                    top_k=params.top_k,
                    table_prefix=params.table_prefix
                )
                return {"matches": results}
            except Exception as e:
                self.logger.error(f"Error searching knowledge base: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.router.delete("/knowledge/{knowledge_id}")
        async def delete_knowledge(
            knowledge_id: UUID,
            table_prefix: Optional[str] = None
        ):
            """Delete a specific knowledge entry and its chunks."""
            try:
                deleted = await self.storage_service.delete_knowledge(
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

        @self.router.post("/tables/create")
        async def create_tables(request: CreateTablesRequest):
            """Create memory or knowledge base tables."""
            try:
                await self.storage_service.create_tables(
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for database connection."""
    # Startup: initialize database
    await app.state.storage_service.db.initialize()
    yield
    # Shutdown: cleanup
    await app.state.storage_service.db.close()


def create_app(storage_service: StorageService) -> FastAPI:
    app = FastAPI(
        title="Agent Storage API",
        lifespan=lifespan
    )

    app.state.storage_service = storage_service
    api = AgentStorageAPI(storage_service)
    app.include_router(api.router)

    return app


if __name__ == "__main__":
    from market_agents.memory.embedding import MemoryEmbedder
    from market_agents.memory.config import load_config_from_yaml
    from market_agents.memory.agent_storage.setup_db import AsyncDatabase
    from pathlib import Path

    # Load config
    config_path = Path(__file__).parent.parent / "memory_config.yaml"
    config = load_config_from_yaml(str(config_path))

    # Initialize embedding service
    embedding_service = MemoryEmbedder(config)

    # Create database instance (but don't initialize connection yet)
    db = AsyncDatabase(config)

    # Initialize storage service
    storage_service = StorageService(
        db=db,
        embedding_service=embedding_service,
        config=config
    )

    app = create_app(storage_service)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )