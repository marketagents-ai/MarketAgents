import logging
from typing import Dict, Any, Optional, List

import aiohttp

from market_agents.memory.knowledge_base import KnowledgeObject
from market_agents.memory.memory import MemoryObject, EpisodicMemoryObject
from market_agents.orchestrators.group_chat.groupchat_api import CognitiveMemoryParams, EpisodicMemoryParams, \
    KnowledgeQueryParams


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


    async def store_memory(self, memory_object: MemoryObject) -> bool:
      """Store a cognitive memory object."""
      url = f"{self.api_url}/memory/cognitive"
      try:
          async with aiohttp.ClientSession() as session:
              async with session.post(url, json=memory_object.dict()) as resp:
                  if resp.status == 200:
                      self.logger.info(f"Stored cognitive memory for agent {memory_object.agent_id}")
                      return True
                  else:
                    error_detail = await resp.text()
                    self.logger.error(f"Failed to store cognitive memory for agent {memory_object.agent_id}: {resp.status}, {error_detail}")
                    return False
      except Exception as e:
          self.logger.error(f"Error storing cognitive memory: {e}")
          return False

    async def store_episodic_memory(self, memory_object: EpisodicMemoryObject) -> bool:
      """Store an episodic memory object."""
      url = f"{self.api_url}/memory/episodic"
      try:
          async with aiohttp.ClientSession() as session:
              async with session.post(url, json=memory_object.dict()) as resp:
                  if resp.status == 200:
                      self.logger.info(f"Stored episodic memory for agent {memory_object.agent_id}")
                      return True
                  else:
                    error_detail = await resp.text()
                    self.logger.error(f"Failed to store episodic memory for agent {memory_object.agent_id}: {resp.status}, {error_detail}")
                    return False
      except Exception as e:
          self.logger.error(f"Error storing episodic memory: {e}")
          return False

    async def retrieve_cognitive_memory(self, agent_id: str, params: CognitiveMemoryParams = CognitiveMemoryParams()) -> Optional[List[Dict]]:
      """Retrieve cognitive memory for an agent."""
      url = f"{self.api_url}/memory/cognitive/{agent_id}"
      try:
          async with aiohttp.ClientSession() as session:
              async with session.get(url, params=params.dict(exclude_none=True)) as resp:
                  if resp.status == 200:
                      memory_data = await resp.json()
                      self.logger.info(f"Retrieved cognitive memory for agent {agent_id}")
                      return memory_data
                  else:
                      error_detail = await resp.text()
                      self.logger.error(f"Failed to retrieve cognitive memory for agent {agent_id}: {resp.status}, {error_detail}")
                      return None
      except Exception as e:
          self.logger.error(f"Error retrieving cognitive memory: {e}")
          return None


    async def retrieve_episodes(self, agent_id: str, query: str, top_k: int) -> Optional[List[Dict]]:
        """Retrieve episodic memory episodes."""
        url = f"{self.api_url}/memory/episodic/{agent_id}"
        params = EpisodicMemoryParams(agent_id=agent_id, query=query, top_k=top_k).dict()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        episodes = await resp.json()
                        self.logger.info(f"Retrieved episodic memory for agent {agent_id}")
                        return episodes
                    else:
                        error_detail = await resp.text()
                        self.logger.error(f"Failed to retrieve episodic memory for agent {agent_id}: {resp.status}, {error_detail}")
                        return None
        except Exception as e:
            self.logger.error(f"Error retrieving episodic memory: {e}")
            return None

    async def delete_memory(self, agent_id: str) -> bool:
        """Delete all memory for an agent."""
        url = f"{self.api_url}/memory/{agent_id}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(url) as resp:
                    if resp.status == 200:
                        self.logger.info(f"Deleted memory for agent {agent_id}")
                        return True
                    else:
                        error_detail = await resp.text()
                        self.logger.error(f"Failed to delete memory for agent {agent_id}: {resp.status}, {error_detail}")
                        return False
        except Exception as e:
            self.logger.error(f"Error deleting memory: {e}")
            return False


    async def ingest_knowledge(self, knowledge_object: KnowledgeObject) -> bool:
        """Ingest knowledge into the knowledge base."""
        url = f"{self.api_url}/knowledge/ingest"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=knowledge_object.dict()) as resp:
                    if resp.status == 200:
                        self.logger.info(f"Ingested knowledge with id {knowledge_object.knowledge_id}")
                        return True
                    else:
                        error_detail = await resp.text()
                        self.logger.error(f"Failed to ingest knowledge with id {knowledge_object.knowledge_id}: {resp.status}, {error_detail}")
                        return False
        except Exception as e:
            self.logger.error(f"Error ingesting knowledge: {e}")
            return False

    async def search_knowledge(self, query: str, top_k: int, table_prefix: Optional[str] = None) -> Optional[List[Dict]]:
        """Search the knowledge base."""
        url = f"{self.api_url}/knowledge/search"
        params = KnowledgeQueryParams(query=query, top_k=top_k, table_prefix=table_prefix).model_dump(exclude_none=True)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        results = await resp.json()
                        self.logger.info(f"Searched knowledge base with query: {query}")
                        return results
                    else:
                        error_detail = await resp.text()
                        self.logger.error(f"Failed to search knowledge base with query {query}: {resp.status}, {error_detail}")
                        return None
        except Exception as e:
            self.logger.error(f"Error searching knowledge base: {e}")
            return None

    async def delete_knowledge(self, knowledge_id: str) -> bool:
      """Delete knowledge from knowledge base."""
      url = f"{self.api_url}/knowledge/{knowledge_id}"
      try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(url) as resp:
                    if resp.status == 200:
                        self.logger.info(f"Deleted knowledge with ID {knowledge_id}")
                        return True
                    else:
                         error_detail = await resp.text()
                         self.logger.error(f"Failed to delete knowledge with ID {knowledge_id}: {resp.status}, {error_detail}")
                         return False
      except Exception as e:
            self.logger.error(f"Error deleting knowledge: {e}")
            return False

