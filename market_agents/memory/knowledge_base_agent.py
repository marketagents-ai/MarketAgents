from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from market_agents.memory.memory_models import RetrievedMemory
from market_agents.memory.knowledge_base import MarketKnowledgeBase

class KnowledgeBaseAgent(BaseModel):
    market_kb: MarketKnowledgeBase
    retrieved_knowledge: List[RetrievedMemory] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    async def store(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        return await self.market_kb.ingest_knowledge(text=text, metadata=metadata)

    async def retrieve(self, query: str):
        return await self.market_kb.search(query=query)