import aiohttp
import logging
from uuid import UUID
from typing import Optional, Dict, Any, List
from market_agents.memory.agent_storage.agent_storage_api import (
    CognitiveMemoryParams, EpisodicMemoryParams, KnowledgeQueryParams,
    IngestKnowledgeRequest, CreateTablesRequest, EpisodicMemoryObject,
)
from market_agents.memory.memory_models import MemoryObject


class AgentStorageAPIUtils:
    def __init__(self, api_url: str, logger: logging.Logger):
        self.api_url = api_url
        self.logger = logger
        self.logger.info(f"Initializing Agent Storage API Utils with URL: {api_url}")

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

    # Cognitive Memory Endpoints
    async def store_cognitive_memory(self, memory: MemoryObject) -> Dict[str, Any]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.api_url}/memory/cognitive", json=memory.dict()) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:
            self.logger.error(f"Error storing cognitive memory: {e}")
            raise

    async def get_cognitive_memory(self, agent_id: str, params: CognitiveMemoryParams) -> Dict[str, Any]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/memory/cognitive/{agent_id}", params=params.dict(exclude_unset=True)) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:
            self.logger.error(f"Error retrieving cognitive memory: {e}")
            raise

    async def search_cognitive_memory(self, agent_id: str, top_k: int, query: str) -> Dict[str, Any]:
        try:
            params = {"top_k": top_k, "query": query}
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/memory/cognitive/search", params=params) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:
            self.logger.error(f"Error searching cognitive memory: {e}")
            raise

    # Episodic Memory Endpoints
    async def store_episodic_memory(self, episode: EpisodicMemoryObject) -> Dict[str, Any]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.api_url}/memory/episodic", json=episode.dict()) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:
            self.logger.error(f"Error storing episodic memory: {e}")
            raise

    async def get_episodic_memory(self, agent_id: str, params: EpisodicMemoryParams) -> Dict[str, Any]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/memory/episodic/{agent_id}", params=params.dict(exclude_unset=True)) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:
            self.logger.error(f"Error retrieving episodic memory: {e}")
            raise

    async def search_episodic_memory(self, agent_id: str, top_k: int, query: str) -> Dict[str, Any]:
        try:
            params = {"top_k": top_k, "query": query}
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/memory/episodic/search", params=params) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:
            self.logger.error(f"Error searching episodic memory: {e}")
            raise

    # Knowledge Base Endpoints
    async def ingest_knowledge(self, request: IngestKnowledgeRequest) -> Dict[str, Any]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.api_url}/knowledge/ingest", json=request.dict()) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:
            self.logger.error(f"Error ingesting knowledge: {e}")
            raise

    async def search_knowledge(self, params: KnowledgeQueryParams) -> Dict[str, Any]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/knowledge/search", params=params.dict(exclude_unset=True)) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:
            self.logger.error(f"Error searching knowledge: {e}")
            raise

    async def delete_knowledge(self, knowledge_id: UUID, table_prefix: Optional[str] = None) -> Dict[str, Any]:
        try:
            params = {"table_prefix": table_prefix}
            async with aiohttp.ClientSession() as session:
                async with session.delete(f"{self.api_url}/knowledge/{knowledge_id}", params=params) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:
            self.logger.error(f"Error deleting knowledge: {e}")
            raise

    # Table Management Endpoints
    async def create_tables(self, request: CreateTablesRequest) -> Dict[str, Any]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.api_url}/tables/create", json=request.dict()) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            raise

    # Agent Memory Management
    async def clear_agent_memory(self, agent_id: str, memory_type: Optional[str] = None) -> Dict[str, Any]:
        try:
            params = {"memory_type": memory_type}
            async with aiohttp.ClientSession() as session:
                async with session.delete(f"{self.api_url}/memory/{agent_id}", params=params) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except Exception as e:
            self.logger.error(f"Error clearing agent memory: {e}")
            raise
