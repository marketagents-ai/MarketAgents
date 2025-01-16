from typing import Dict, List, Optional, Any
from uuid import UUID
from pydantic import BaseModel, Field

from market_agents.memory.knowledge_base import MarketKnowledgeBase
from market_agents.memory.vector_search import RetrievedMemory
from market_agents.memory.vector_search import MemoryRetriever

class KnowledgeBaseAgent(BaseModel):
    market_kb: MarketKnowledgeBase
    retriever: MemoryRetriever
    knowledge_bases: Dict[str, MarketKnowledgeBase] = Field(default_factory=dict)
    retrieved_knowledge: List[RetrievedMemory] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def store(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> UUID:
        return self.market_kb.ingest_knowledge(text, metadata)

    def retrieve(self, query: str, table_prefix: str) -> List[RetrievedMemory]:
        return self.retriever.search_knowledge_base(table_prefix, query, self.market_kb.config.top_k)