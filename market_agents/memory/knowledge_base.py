from abc import ABC, abstractmethod
import json
import logging
import uuid
import re

from uuid import UUID
from datetime import datetime
from typing import List
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from market_agents.memory.embedding import MemoryEmbedder
from market_agents.memory.setup_db import DatabaseConnection

class KnowledgeObject(BaseModel):
    knowledge_id: UUID = Field(default_factory=uuid.uuid4)
    content: str
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

class KnowledgeChunk(BaseModel):
    text: str
    start: int
    end: int
    embedding: Optional[List[float]] = None
    knowledge_id: Optional[UUID] = None

class KnowledgeChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[KnowledgeChunk]:
        pass

class MarketKnowledgeBase:
    """
    A base class for agent knowledge bases built with embeddings for semantic search.
    Dynamically handles agent- or knowledge base-specific tables.
    """

    def __init__(self, config, db_conn: DatabaseConnection, embedding_service: MemoryEmbedder, table_prefix: str, chunking_method: Optional[KnowledgeChunker] = None):
        self.config = config
        self.db = db_conn
        self.embedding_service = embedding_service
        self.chunking_method = chunking_method
        self.table_prefix = table_prefix
        self.knowledge_objects_table = f"{table_prefix}_knowledge_objects"
        self.knowledge_chunks_table = f"{table_prefix}_knowledge_chunks"
        if not self.check_table_exists(db_conn, table_prefix):
            self.db.create_knowledge_base_tables(table_prefix)

    @classmethod
    def check_table_exists(cls, db_conn: DatabaseConnection, table_prefix: str) -> bool:
        """Check if knowledge base tables exist"""
        try:
            db_conn.connect()
            db_conn.cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                );
            """, (f"{table_prefix}_knowledge_objects",))
            
            return db_conn.cursor.fetchone()[0]
        except Exception as e:
            logging.error(f"Error checking knowledge base existence: {str(e)}")
            return False

    def ingest_knowledge(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> UUID:
        """Process a document: chunk, embed, and store as a KnowledgeObject."""
        chunks = self._chunk(text)
        
        # Process embeddings in batches according to config
        all_embeddings = []
        chunk_texts = [c.text for c in chunks]
        
        for i in range(0, len(chunk_texts), self.config.batch_size):
            batch = chunk_texts[i:i + self.config.batch_size]
            logging.info(f"Getting embedding for batch:\n{batch}")
            batch_embeddings = self.embedding_service.get_embeddings(batch)
            all_embeddings.extend(batch_embeddings)

        for chunk, emb in zip(chunks, all_embeddings):
            chunk.embedding = emb

        knowledge_id = self._save_knowledge_and_chunks(text, chunks, metadata)
        return knowledge_id

    def _chunk(self, text: str) -> List[KnowledgeChunk]:
        if self.chunking_method:
            return self.chunking_method.chunk(text)
        else:
            default_chunker = SemanticChunker(
                min_size=self.config.min_chunk_size,
                max_size=self.config.max_chunk_size
            )
            return default_chunker.chunk(text)

    def _save_knowledge_and_chunks(self, document_text: str, chunks: List[KnowledgeChunk], metadata: Optional[Dict[str, Any]]) -> UUID:
        """Save knowledge object and chunks to the dynamically named tables."""
        knowledge_id = uuid.uuid4()
        self.db.connect()
        try:
            # Insert into the knowledge objects table
            self.db.cursor.execute(f"""
                INSERT INTO {self.knowledge_objects_table} (knowledge_id, content, metadata)
                VALUES (%s, %s, %s)
                RETURNING created_at;
            """, (str(knowledge_id), document_text, json.dumps(metadata) if metadata else json.dumps({})))
            created_at = self.db.cursor.fetchone()[0]

            # Insert chunks into the knowledge chunks table
            for chunk in chunks:
                self.db.cursor.execute(f"""
                    INSERT INTO {self.knowledge_chunks_table} (knowledge_id, text, start_pos, end_pos, embedding)
                    VALUES (%s, %s, %s, %s, %s);
                """, (str(knowledge_id), chunk.text, chunk.start, chunk.end, chunk.embedding))

            self.db.conn.commit()
            return knowledge_id
        except Exception as e:
            self.db.conn.rollback()
            raise e

    def clear_knowledge_base(self):
        """Clear all knowledge entries from the dynamically named tables."""
        self.db.connect()
        try:
            self.db.cursor.execute(f"DELETE FROM {self.knowledge_chunks_table};")
            self.db.cursor.execute(f"DELETE FROM {self.knowledge_objects_table};")
            self.db.conn.commit()
        except Exception as e:
            self.db.conn.rollback()
            raise e
        
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