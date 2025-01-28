import json
import aiohttp
import logging
from uuid import UUID
from datetime import datetime
from typing import Optional, Dict, Any, List

from market_agents.memory.memory_models import (
    CognitiveMemoryParams,
    EpisodicMemoryParams,
    KnowledgeQueryParams,
    IngestKnowledgeRequest,
    CreateTablesRequest,
    EpisodicMemoryObject,
    MemoryObject
)
from market_agents.memory.config import AgentStorageConfig, load_config_from_yaml

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class AgentStorageAPIUtils:
    def __init__(
        self,
        config: AgentStorageConfig | str,
        logger: Optional[logging.Logger] = None
    ):
        if isinstance(config, str):
            self.config = load_config_from_yaml(config)
        elif isinstance(config, AgentStorageConfig):
            self.config = config
        else:
            raise ValueError("config must be either a path or AgentStorageConfig instance")
        
        self.api_url = self.config.storage_api_url
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info(f"Initializing Agent Storage API Utils with URL: {self.api_url}")
        self.json_encoder = CustomJSONEncoder

    async def check_api_health(self) -> bool:
        """Check if the Agent Storage API is healthy."""
        try:
            self.logger.info(f"Checking Agent Storage API health at {self.api_url}/health")
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/health", timeout=5) as resp:
                    if resp.status == 200:
                        self.logger.info("Agent Storage API is healthy")
                        return True
                    else:
                        self.logger.error(f"Agent Storage API health check failed: {resp.status}")
                        return False
        except aiohttp.ClientError as e:
            self.logger.error(f"Connection error to Agent Storage API: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Could not connect to Agent Storage API: {e}")
            return False

    async def store_cognitive_memory(self, memory: MemoryObject) -> Dict[str, Any]:
        try:
            memory_dict = memory.model_dump(exclude_none=True)
            async with aiohttp.ClientSession(json_serialize=lambda x: json.dumps(x, cls=self.json_encoder)) as session:
                async with session.post(
                    f"{self.api_url}/memory/cognitive",
                    json=memory_dict
                ) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:
            self.logger.error(f"Error storing cognitive memory: {e}")
            raise

    async def store_episodic_memory(self, episode: EpisodicMemoryObject) -> Dict[str, Any]:
        try:
            episode_dict = episode.model_dump(exclude_none=True)
            async with aiohttp.ClientSession(json_serialize=lambda x: json.dumps(x, cls=self.json_encoder)) as session:
                async with session.post(
                    f"{self.api_url}/memory/episodic",
                    json=episode_dict
                ) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:
            self.logger.error(f"Error storing episodic memory: {e}")
            raise

    async def get_cognitive_memory_sql(
        self,
        agent_id: str,
        params: CognitiveMemoryParams
    ) -> List[MemoryObject]:
        """Get cognitive memories using SQL query."""
        try:
            query_params = {
                "limit": params.limit,
                "cognitive_step": params.cognitive_step,
                "metadata_filters": json.dumps(params.metadata_filters) if params.metadata_filters else None,
                "start_time": params.start_time.isoformat() if params.start_time else None,
                "end_time": params.end_time.isoformat() if params.end_time else None
            }
            query_params = {k: v for k, v in query_params.items() if v is not None}

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/memory/cognitive/sql/{agent_id}",
                    params=query_params
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    memories_data = data.get("memories", [])
                    return [MemoryObject.model_validate(mem_dict) for mem_dict in memories_data]
        except Exception as e:
            self.logger.error(f"Error getting cognitive memory: {e}")
            raise

    async def get_episodic_memory_sql(self, agent_id: str, params: EpisodicMemoryParams) -> Dict[str, Any]:
        """
        Calls GET /memory/episodic/sql/{agent_id} for standard SQL retrieval
        using time range, metadata filters, etc.
        """
        try:
            query_params = params.model_dump(exclude_unset=True)
            # Possibly convert start/end times, metadata here if needed
            if query_params.get('start_time'):
                query_params['start_time'] = query_params['start_time'].isoformat()
            if query_params.get('end_time'):
                query_params['end_time'] = query_params['end_time'].isoformat()
            if query_params.get('metadata_filters'):
                query_params['metadata_filters'] = json.dumps(query_params['metadata_filters'])

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/memory/episodic/sql/{agent_id}",
                    params=query_params
                ) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:
            self.logger.error(f"Error retrieving episodic memory (SQL): {e}")
            raise

    async def get_cognitive_memory_vector(self, agent_id: str, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Calls GET /memory/cognitive/vector/{agent_id} for embedding-based retrieval."""
        try:
            params = {"query": query, "top_k": top_k}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/memory/cognitive/vector/{agent_id}",
                    params=params
                ) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:
            self.logger.error(f"Error searching cognitive memory (vector): {e}")
            raise

    async def get_episodic_memory_vector(self, agent_id: str, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Calls GET /memory/episodic/vector/{agent_id} for embedding-based retrieval."""
        try:
            params = {"query": query, "top_k": top_k}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/memory/episodic/vector/{agent_id}",
                    params=params
                ) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:
            self.logger.error(f"Error searching episodic memory (vector): {e}")
            raise

    async def ingest_knowledge(self, request: IngestKnowledgeRequest) -> Dict[str, Any]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/knowledge/ingest",
                    json=request.model_dump()
                ) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:
            self.logger.error(f"Error ingesting knowledge: {e}")
            raise

    async def search_knowledge(self, params: KnowledgeQueryParams) -> Dict[str, Any]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/knowledge/search",
                    params=params.model_dump(exclude_unset=True)
                ) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:
            self.logger.error(f"Error searching knowledge: {e}")
            raise

    async def delete_knowledge(self, knowledge_id: UUID, table_prefix: Optional[str] = None) -> Dict[str, Any]:
        try:
            params = {"table_prefix": table_prefix}
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.api_url}/knowledge/{knowledge_id}",
                    params=params
                ) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:
            self.logger.error(f"Error deleting knowledge: {e}")
            raise

    async def create_tables(self, request: CreateTablesRequest) -> Dict[str, Any]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/tables/create",
                    json=request.model_dump()
                ) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            raise

    async def clear_agent_memory(self, agent_id: str, memory_type: Optional[str] = None) -> Dict[str, Any]:
        try:
            params = {"memory_type": memory_type}
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    f"{self.api_url}/memory/{agent_id}",
                    params=params
                ) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:
            self.logger.error(f"Error clearing agent memory: {e}")
            raise