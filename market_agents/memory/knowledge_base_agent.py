from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from market_agents.memory.storage_models import RetrievedMemory
from market_agents.memory.knowledge_base import MarketKnowledgeBase

class KnowledgeBaseAgent(BaseModel):
    """
    A Pydantic model representing a knowledge base agent.
    """
    id: Optional[str] = Field(
        None, 
        description="Agent ID or reference for the knowledge base agent"
    )
    market_kb: MarketKnowledgeBase = Field(
        ...,
        description="The market knowledge base instance used by the agent"
    )
    retrieved_knowledge: List[RetrievedMemory] = Field(
        default_factory=list,
        description="List of retrieved knowledge entries"
    )

    class Config:
        arbitrary_types_allowed = True

    async def store(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Store a document in the knowledge base.
        """
        return await self.market_kb.ingest_knowledge(text=text, metadata=metadata)

    async def retrieve(self, query: str, top_k: int):
        """
        Retrieve knowledge from the knowledge base using a query.
        """
        return await self.market_kb.search(query=query, top_k=top_k)