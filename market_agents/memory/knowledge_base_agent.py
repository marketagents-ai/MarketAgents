from typing import Dict, List, Optional, Any
from uuid import UUID

from market_agents.memory.knowledge_base import MarketKnowledgeBase
from market_agents.memory.vector_search import RetrievedMemory
from market_agents.memory.vector_search import MemoryRetriever


class KnowledgeBaseAgent:
    def __init__(self, market_knowledge_base: MarketKnowledgeBase, retriever: MemoryRetriever):
        self.knowledge_bases: Dict[str, MarketKnowledgeBase] = {}
        self.retrieved_knowledge: List[RetrievedMemory] = []

        self.marketKnowledgeBase = market_knowledge_base
        self.retriever = retriever


    def store(self, text: str, metadata: Optional[Dict[str, Any]] = None)-> UUID:
        return self.marketKnowledgeBase.ingest_knowledge(text, metadata)


    def retrieve(self, query: str, table_prefix:str)-> List[RetrievedMemory]:
        return self.retriever.search_knowledge_base(table_prefix, query, self.marketKnowledgeBase.config.top_k)