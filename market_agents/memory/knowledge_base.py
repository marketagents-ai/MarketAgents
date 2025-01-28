from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
import re
from uuid import UUID

from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.memory_models import (
    CreateTablesRequest,
    IngestKnowledgeRequest,
    KnowledgeQueryParams,
    RetrievedMemory,
    KnowledgeChunk
)
from market_agents.memory.config import AgentStorageConfig

class KnowledgeChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[KnowledgeChunk]:
        pass

class MarketKnowledgeBase:
    """
    A base class for agent knowledge bases built with embeddings for semantic search.
    Uses AgentStorageAPIUtils for all storage operations.
    """

    def __init__(self, 
                 config: AgentStorageConfig,
                 table_prefix: str,
                 chunking_method: Optional[KnowledgeChunker] = None):
        self.config = config
        self.agent_storage_utils = AgentStorageAPIUtils(config)
        self.chunking_method = chunking_method
        self.table_prefix = table_prefix

    async def initialize(self):
        """Initialize knowledge base tables"""
        await self.agent_storage_utils.create_tables(
            CreateTablesRequest(
                table_type="knowledge",
                table_prefix=self.table_prefix
            )
        )

    async def ingest_knowledge(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> UUID:
        """Process and store a document in the knowledge base."""
        response = await self.agent_storage_utils.ingest_knowledge(
            IngestKnowledgeRequest(
                text=text,
                metadata=metadata,
                table_prefix=self.table_prefix
            )
        )
        return UUID(response["knowledge_id"])

    async def search(self, query: str, top_k: int = 5) -> List[RetrievedMemory]:
        """Search the knowledge base using semantic similarity."""
        results = await self.agent_storage_utils.search_knowledge(
            KnowledgeQueryParams(
                query=query,
                top_k=top_k,
                table_prefix=self.table_prefix
            )
        )
        return [RetrievedMemory(**match) for match in results["matches"]]

    async def clear(self):
        """Clear all knowledge entries."""
        await self.agent_storage_utils.clear_agent_memory(
            self.table_prefix,
            "knowledge"
        )

class SemanticChunker(KnowledgeChunker):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def chunk(self, text: str) -> List[KnowledgeChunk]:
        text = re.sub(r'\n{3,}', '\n\n', text)

        splits = [
            r'(?<=\.)\s*\n\n(?=[A-Z])',
            r'(?<=\n)\s*#{1,6}\s+[A-Z]',
            r'(?<=\n)[A-Z][^a-z]*?:\s*\n',
            r'(?<=\n\n)\s*(?:[-•\*]|\d+\.)\s+',
            r'(?<=\.)\s{2,}(?=[A-Z])',
            r'(?<=[\.\?\!])\s+(?=[A-Z])'
        ]

        pattern = '|'.join(splits)
        segments = re.split(pattern, text.strip())

        chunks = []
        current_chunk = []
        current_length = 0
        current_pos = 0
        chunk_start = 0

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            if current_length + len(segment) > self.max_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(KnowledgeChunk(
                    text=chunk_text,
                    start=chunk_start,
                    end=chunk_start + len(chunk_text)
                ))
                current_chunk = []
                current_length = 0
                chunk_start = current_pos

            current_chunk.append(segment)
            current_length += len(segment)
            current_pos += len(segment) + 1

            if current_length >= self.min_size:
                chunk_text = ' '.join(current_chunk)
                chunks.append(KnowledgeChunk(
                    text=chunk_text,
                    start=chunk_start,
                    end=chunk_start + len(chunk_text)
                ))
                current_chunk = []
                current_length = 0
                chunk_start = current_pos

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(KnowledgeChunk(
                text=chunk_text,
                start=chunk_start,
                end=chunk_start + len(chunk_text)
            ))

        return chunks