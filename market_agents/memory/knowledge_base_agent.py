from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from market_agents.memory.agent_storage.agent_storage_api import IngestKnowledgeRequest, KnowledgeQueryParams
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.memory_models import RetrievedMemory


class KnowledgeBaseAgent(BaseModel):
    agent_storage_utils: AgentStorageAPIUtils
    retrieved_knowledge: List[RetrievedMemory] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    async def store(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        return await self.agent_storage_utils.ingest_knowledge(IngestKnowledgeRequest(metadata=metadata, text=text))

    async def retrieve(self, query: str, table_prefix: str):
        return await self.agent_storage_utils.search_knowledge(KnowledgeQueryParams(query=query, table_prefix=table_prefix))